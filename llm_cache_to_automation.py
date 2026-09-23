"""LLM-based converter: action cache → Optexity automation JSON.

Uses the LLM (via litellm/bedrock) with the Pydantic schema injected as context
so the model understands exactly what to produce.  Validates output with
Automation.model_validate(); on ValidationError it feeds the error back to the
LLM for up to MAX_REPAIR rounds.  Falls back to rule-based cache_to_automation
if all repairs fail.

Usage:
    from llm_cache_to_automation import llm_convert

    automation = llm_convert(
        cache_path="action_cache.json",
        input_parameters={"destination_city": ["Gurgaon"], ...},
        start_url="https://www.booking.com",
    )
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_REPAIR = 3  # self-repair iterations on ValidationError

# ---------------------------------------------------------------------------
# Schema extraction — no hardcoding, sourced directly from Pydantic models
# ---------------------------------------------------------------------------

def _get_focused_schema() -> str:
    """Return a compact JSON schema covering only the action types the LLM needs.

    We omit the full Automation schema (60 defs, very noisy) and instead give the
    LLM a curated subset: the wrapper shape + the five interaction types it will use.
    """
    from optexity.schema.actions.interaction_action import (
        ClickElementAction,
        GoToUrlAction,
        InputTextAction,
        ScrollAction,
        SelectOptionAction,
    )

    wrapper = {
        "automation": {
            "url": "string — start URL",
            "parameters": {
                "input_parameters": "object — same as provided",
                "generated_parameters": {},
            },
            "nodes": "array of action_node objects (see below)",
        },
        "action_node": {
            "type": "action_node",
            "interaction_action": {
                "NOTE": "exactly ONE of the fields below must be set; others omitted",
                "click_element": "ClickElementAction schema",
                "input_text": "InputTextAction schema",
                "select_option": "SelectOptionAction schema",
                "scroll": "ScrollAction schema",
                "go_to_url": "GoToUrlAction schema",
            },
            "expect_new_tab": "bool — set true when click opens a new tab (target=_blank)",
        },
        "ClickElementAction": _trim_schema(ClickElementAction.model_json_schema()),
        "InputTextAction": _trim_schema(InputTextAction.model_json_schema()),
        "SelectOptionAction": _trim_schema(SelectOptionAction.model_json_schema()),
        "ScrollAction": _trim_schema(ScrollAction.model_json_schema()),
        "GoToUrlAction": _trim_schema(GoToUrlAction.model_json_schema()),
    }
    return json.dumps(wrapper, indent=2)


def _trim_schema(schema: dict) -> dict:
    """Keep only 'properties' and 'required' — drop noisy allOf/anyOf/$defs."""
    return {k: v for k, v in schema.items() if k in ("properties", "required", "title")}


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are an expert at converting browser-automation action caches into Optexity \
automation JSON.  You produce valid JSON that matches the provided schema exactly.  \
Never add fields not in the schema.  Return ONLY the JSON object, no prose."""


def _build_prompt(
    cache: dict,
    input_parameters: dict,
    start_url: str,
    schema_str: str,
    prior_error: str | None = None,
    failed_history: str | None = None,
) -> str:
    deterministic = cache.get("deterministic_actions_list", [])

    # Compact element summary for each action
    action_lines = []
    for i, act in enumerate(deterministic):
        el = act.get("element") or {}
        attrs = el.get("attributes", {})
        summary = {
            "i": i,
            "type": act.get("action_type"),
            "params": act.get("action_params", {}),
            "tag": el.get("tag_name", ""),
            "ax_name": el.get("ax_name", ""),
            "id": attrs.get("id", ""),
            "data-test": attrs.get("data-test", ""),
            "data-testid": attrs.get("data-testid", ""),
            "name": attrs.get("name", ""),
            "placeholder": attrs.get("placeholder", ""),
            "role": attrs.get("role", ""),
            "aria-label": attrs.get("aria-label", ""),
            "target": attrs.get("target", ""),
        }
        # Strip empty fields
        summary = {k: v for k, v in summary.items() if v}
        action_lines.append(json.dumps(summary))

    actions_str = "\n".join(action_lines)

    repair_block = ""
    if prior_error:
        repair_block = f"""
The previous attempt produced invalid JSON.  Validation error:
{prior_error}

Fix the error and return the corrected JSON.
"""

    history_block = ""
    if failed_history:
        history_block = f"""
## Failed Node History (locators that were tried and FAILED in previous rounds — do NOT reuse these)
{failed_history}

For each listed node, generate a DIFFERENT locator strategy than what was tried.
"""

    return f"""{repair_block}{history_block}
Convert the following browser action cache into a valid Optexity automation JSON.

## Schema
{schema_str}

## Input Parameters (use {{key[index]}} syntax for variable substitution)
{json.dumps(input_parameters, indent=2)}

## Start URL
{start_url}

## Deterministic Actions (one per line, JSON)
{actions_str}

## Rules
1. Map each action to ONE node with the matching interaction_action type.
2. Use stable Playwright locators in `command`:
   - Prefer: data-testid > id > name > placeholder > role+ax_name > aria-label > xpath
   - For data-testid selectors that match multiple elements append .nth(0)
   - Format: `locator("#id")`, `get_by_role("button", name="Search")`, etc.
3. Replace literal param values with {{key[index]}} refs in both `command` and `input_text`.
4. Set `prompt_instructions` to a short human-readable description of the action.
5. If a click action is followed by a text input on the same element, keep both nodes.
6. Set `expect_new_tab: true` when target="_blank".
7. Set `optional: true` on popup/overlay dismiss nodes.
8. For scroll actions use {{"down": true}} or {{"down": false}}.
9. The `url` field must be the start URL.

Return ONLY a valid JSON object.
"""


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _call_llm(prompt: str) -> str:
    """Call the bedrock model via litellm and return the raw text response."""
    import litellm

    response = litellm.completion(
        model="bedrock/us.anthropic.claude-sonnet-4-6",
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=4096,
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict | None:
    """Extract the first JSON object from LLM output (handles markdown fences)."""
    # Strip markdown fence
    fenced = re.search(r"```(?:json)?\s*(\{.*?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)

    # Find outermost { ... }
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass
    return None


# ---------------------------------------------------------------------------
# Validate + self-repair
# ---------------------------------------------------------------------------

def _validate(data: dict) -> tuple[object | None, str | None]:
    """Validate against Automation schema.  Returns (instance, None) or (None, error)."""
    from optexity.schema.automation import Automation

    try:
        return Automation.model_validate(data), None
    except Exception as e:
        return None, str(e)[:800]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def llm_convert(
    cache_path: str | Path,
    input_parameters: dict | None = None,
    start_url: str | None = None,
    failed_history: str | None = None,
) -> dict:
    """Build a deterministic Optexity automation from an action cache using LLM.

    failed_history: compact multi-line string of previously-failed (node, locator, error)
    entries so the LLM avoids repeating the same broken locators.

    Falls back to rule-based cache_to_automation on failure.
    Returns a plain dict (not a Pydantic model).
    """
    cache_path = Path(cache_path)
    with open(cache_path) as f:
        cache = json.load(f)

    url = start_url or cache.get("start_url", "")
    params = input_parameters or {}
    schema_str = _get_focused_schema()

    prior_error: str | None = None
    for attempt in range(MAX_REPAIR + 1):
        if attempt > 0:
            logger.info(f"[LLM builder] repair attempt {attempt}/{MAX_REPAIR}")

        prompt = _build_prompt(cache, params, url, schema_str, prior_error, failed_history)
        try:
            raw = _call_llm(prompt)
        except Exception as e:
            logger.error(f"[LLM builder] LLM call failed: {e}")
            break

        data = _extract_json(raw)
        if data is None:
            prior_error = "Response did not contain a valid JSON object."
            logger.warning(f"[LLM builder] attempt {attempt}: no JSON found")
            continue

        instance, error = _validate(data)
        if instance is not None:
            logger.info(f"[LLM builder] validated on attempt {attempt}")
            return data

        prior_error = error
        logger.warning(f"[LLM builder] attempt {attempt}: validation failed — {error}")

    # All attempts exhausted — fall back to rule-based
    logger.warning("[LLM builder] falling back to rule-based converter")
    from cache_to_automation import convert_cache_to_automation

    return convert_cache_to_automation(cache_path, params, url)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="LLM-based cache → automation converter")
    parser.add_argument("--cache", required=True, help="Path to action_cache.json")
    parser.add_argument("--params", default=None, help="JSON file with input_parameters")
    parser.add_argument("--output", default="automation_llm.json", help="Output path")
    parser.add_argument("--url", default=None, help="Override start URL")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    input_parameters: dict = {}
    if args.params:
        with open(args.params) as f:
            p = json.load(f)
        input_parameters = (
            p.get("input_parameters")
            or p.get("parameters", {}).get("input_parameters", {})
            or p
        )

    automation = llm_convert(args.cache, input_parameters, args.url)

    with open(args.output, "w") as f:
        json.dump(automation, f, indent=2)

    n = len(automation.get("nodes", []))
    print(f"Generated {n} nodes → {args.output}")

    from optexity.schema.automation import Automation

    try:
        Automation.model_validate(automation)
        print("Schema validation: PASSED")
    except Exception as e:
        print(f"Schema validation: FAILED — {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
