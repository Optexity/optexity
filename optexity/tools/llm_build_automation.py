"""CLI: LLM (Gemini) auto-builder for cached Automation JSON (Phase 6 / Bonus A).

Prompts Gemini with (a) the filtered Phase 1/2 export, (b) the Automation
pydantic JSON schema, and (c) 1-2 valid example automations. Validates each
reply with ``Automation.model_validate`` and, on failure, feeds the error back
for up to ``--max-attempts`` retries. See ``optexity_cursor_plan.md`` Phase 6.

Bonus B (iterative verify/repair loop) is intentionally not implemented.

    python -m optexity.tools.llm_build_automation \\
      --input cached_run_<ts>.json \\
      --output test_automation_cached_llm.json \\
      --ground-truth test_automation_cached.json \\
      --base-automation test_automation.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import litellm
from browser_use.cache.step_filter import filter_steps
from pydantic import ValidationError

from optexity.schema.automation import Automation
from optexity.utils.llm_settings import llm_settings, resolve_llm_api_key

logger = logging.getLogger(__name__)

_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)

# Compact valid examples so the model sees the oneof shape without pasting the
# entire hand-built file twice. Locators/values here are illustrative only;
# the filtered export is the source of truth for the run being built.
_MINI_EXAMPLE: dict[str, Any] = {
    "url": "https://example.com/form",
    "browser_channel": "chromium",
    "backend": "browser-use",
    "parameters": {
        "input_parameters": {"first_name": ["myname"]},
        "generated_parameters": {},
    },
    "nodes": [
        {
            "type": "action_node",
            "interaction_action": {
                "input_text": {
                    "command": "locator(\"[name='email']\").first",
                    "prompt_instructions": "Fill 'email' with 'a@b.com'",
                    "input_text": "a@b.com",
                    "press_enter": False,
                }
            },
            "end_sleep_time": 5.0,
        },
        {
            "type": "action_node",
            "interaction_action": {
                "click_element": {
                    "command": "locator(\"[data-title='Next']\").first",
                    "prompt_instructions": "Click 'Next'",
                }
            },
            "end_sleep_time": 5.0,
        },
    ],
    "automation_description": "Minimal valid cached automation example.",
}


def _extract_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object from a model reply (raw or fenced)."""
    text = text.strip()
    fence = _JSON_FENCE.search(text)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise ValueError("model reply contained no JSON object")
    return json.loads(text[start : end + 1])


def _node_fingerprint(automation: Automation) -> list[dict[str, Any]]:
    """Compact per-node summary for ground-truth diffing."""
    out: list[dict[str, Any]] = []
    for node in automation.nodes:
        if getattr(node, "type", None) != "action_node":
            out.append({"type": getattr(node, "type", None)})
            continue
        ia = node.interaction_action
        kind = None
        command = None
        value = None
        press_enter = None
        if ia is None:
            out.append({"type": "action_node", "kind": None})
            continue
        if ia.input_text is not None:
            kind = "input_text"
            command = ia.input_text.command
            value = ia.input_text.input_text
            press_enter = ia.input_text.press_enter
        elif ia.click_element is not None:
            kind = "click"
            command = ia.click_element.command
        elif ia.select_option is not None:
            kind = "select"
            command = ia.select_option.command
            value = ia.select_option.select_values
        elif ia.go_to_url is not None:
            kind = "go_to_url"
            value = ia.go_to_url.url
        out.append(
            {
                "kind": kind,
                "command": command,
                "value": value,
                "press_enter": press_enter,
                "end_sleep_time": node.end_sleep_time,
            }
        )
    return out


def compare_to_ground_truth(
    llm_automation: Automation, ground_truth: Automation
) -> list[str]:
    """Return human-readable diff lines (empty => structurally aligned)."""
    diffs: list[str] = []
    if llm_automation.url != ground_truth.url:
        diffs.append(f"url: llm={llm_automation.url!r} gt={ground_truth.url!r}")

    llm_fp = _node_fingerprint(llm_automation)
    gt_fp = _node_fingerprint(ground_truth)
    if len(llm_fp) != len(gt_fp):
        diffs.append(f"node_count: llm={len(llm_fp)} gt={len(gt_fp)}")

    for i, (a, b) in enumerate(zip(llm_fp, gt_fp)):
        for key in ("kind", "command", "value", "press_enter"):
            if a.get(key) != b.get(key):
                diffs.append(f"node[{i}].{key}: llm={a.get(key)!r} gt={b.get(key)!r}")
    for i in range(min(len(llm_fp), len(gt_fp)), max(len(llm_fp), len(gt_fp))):
        side = "llm" if i < len(llm_fp) else "gt"
        extra = llm_fp[i] if side == "llm" else gt_fp[i]
        diffs.append(f"node[{i}]: only in {side}: {extra}")
    return diffs


def _build_prompt(
    *,
    kept: list[dict[str, Any]],
    schema: dict[str, Any],
    examples: list[dict[str, Any]],
    url: str,
    input_parameters: dict[str, Any],
    browser_channel: str,
    backend: str,
    prior_error: str | None,
) -> str:
    hard_rules = """
Hard rules:
1. NEVER invent bracket indices like [67]. Locators must use stable identity from the export
   (id, name, placeholder, aria-label, data-* attrs, role+text, else xpath only if in_shadow_dom is false).
2. NEVER emit xpath for in_shadow_dom=true elements.
3. Replay ONLY the kept filtered steps (plus go_to_url only if clearly required by export page_url jumps).
4. For input_text: set command, prompt_instructions, input_text from typed_value.
5. For click: set command + prompt_instructions.
6. If caused_navigation or press_enter is true, set end_sleep_time to 5.0 on that action_node
   (Optexity maps this to wait_for_load_state("load"); do NOT add sleep_action / time.sleep).
7. Preserve input_parameters exactly as given.
8. Return ONLY a single JSON object matching the Automation schema — no markdown commentary.
""".strip()

    parts = [
        "You are building a deterministic Optexity cached Automation JSON from a filtered browser-use export.",
        hard_rules,
        f"Target url: {url}",
        f"browser_channel: {browser_channel}",
        f"backend: {backend}",
        "input_parameters JSON:",
        json.dumps(input_parameters, indent=2),
        "Filtered export steps (source of truth for locators/values):",
        json.dumps(kept, indent=2),
        "Automation pydantic JSON schema:",
        json.dumps(schema),
        "Valid example automation(s):",
        json.dumps(examples, indent=2),
    ]
    if prior_error:
        parts.extend(
            [
                "Your previous JSON failed Automation.model_validate. Fix ONLY the validation issues.",
                f"Validation error:\n{prior_error}",
            ]
        )
    parts.append("Emit the full Automation JSON object now.")
    return "\n\n".join(parts)


def call_gemini(prompt: str, model: str) -> str:
    api_key = resolve_llm_api_key(model)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
    }
    if api_key:
        kwargs["api_key"] = api_key
    # litellm gemini route prefers GEMINI_API_KEY; mirror GOOGLE_API_KEY if needed.
    if model.startswith("gemini/") and api_key and not os.environ.get("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = api_key

    logger.info(f"LLM request model={model} prompt_chars={len(prompt)}")
    response = litellm.completion(**kwargs)
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("empty LLM response content")
    return content


def build_with_retries(
    *,
    kept: list[dict[str, Any]],
    url: str,
    input_parameters: dict[str, Any],
    browser_channel: str,
    backend: str,
    examples: list[dict[str, Any]],
    model: str,
    max_attempts: int,
) -> Automation:
    schema = Automation.model_json_schema()
    prior_error: str | None = None
    last_raw: str | None = None

    for attempt in range(1, max_attempts + 1):
        logger.info(f"attempt {attempt}/{max_attempts}")
        print(f"ATTEMPT {attempt}/{max_attempts}", flush=True)
        prompt = _build_prompt(
            kept=kept,
            schema=schema,
            examples=examples,
            url=url,
            input_parameters=input_parameters,
            browser_channel=browser_channel,
            backend=backend,
            prior_error=prior_error,
        )
        try:
            raw = call_gemini(prompt, model=model)
            last_raw = raw
            logger.info(f"attempt {attempt}: got reply chars={len(raw)}")
            print(f"ATTEMPT {attempt}: reply_chars={len(raw)}", flush=True)
            data = _extract_json_object(raw)
            automation = Automation.model_validate(data)
            logger.info(f"attempt {attempt}: Automation.model_validate OK ({len(automation.nodes)} nodes)")
            print(
                f"ATTEMPT {attempt}: Automation.model_validate OK nodes={len(automation.nodes)}",
                flush=True,
            )
            return automation
        except (ValidationError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
            prior_error = str(exc)
            logger.warning(f"attempt {attempt}: FAILED: {prior_error}")
            print(f"ATTEMPT {attempt}: FAILED: {prior_error}", flush=True)

    raise SystemExit(
        f"ERROR: failed to produce a schema-valid Automation after {max_attempts} attempts. "
        f"Last error: {prior_error!r}. Last raw reply starts: {(last_raw or '')[:500]!r}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Phase 1 export JSON (cached_run_<ts>.json)")
    parser.add_argument("--output", default="test_automation_cached_llm.json")
    parser.add_argument(
        "--ground-truth",
        default="test_automation_cached.json",
        help="Phase 3 hand-built cached automation to diff against",
    )
    parser.add_argument("--base-automation", default="test_automation.json")
    parser.add_argument("--example", action="append", default=[], help="extra example Automation JSON path (repeatable)")
    parser.add_argument("--url", default=None)
    parser.add_argument("--model", default=None, help="litellm model string (default: LLM_MODEL / gemini flash-lite)")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--diff-report", default="llm_vs_handbuilt_diff.md")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)
    logger.setLevel(logging.INFO)

    records = json.loads(Path(args.input).read_text(encoding="utf-8"))
    kept, discarded = filter_steps(records)
    logger.info(f"filter_steps: total={len(records)} kept={len(kept)} discarded={len(discarded)}")

    url = args.url
    browser_channel = "chromium"
    backend = "browser-use"
    input_parameters: dict[str, Any] = {}
    base_path = Path(args.base_automation)
    if base_path.exists():
        base = json.loads(base_path.read_text(encoding="utf-8"))
        if url is None:
            url = base["url"]
        browser_channel = base.get("browser_channel", browser_channel)
        backend = base.get("backend", backend)
        input_parameters = (base.get("parameters") or {}).get("input_parameters") or {}
    if url is None:
        parser.error(f"--url not given and base automation not found at {base_path}")

    examples: list[dict[str, Any]] = [_MINI_EXAMPLE]
    # Second few-shot example: prefer a *different* site's hand-built cache so
    # the model sees real shape without being handed the ground-truth answer.
    gt_path = Path(args.ground_truth)
    alt_example = Path("test_automation_2_cached.json")
    if alt_example.exists() and alt_example.resolve() != gt_path.resolve():
        examples.append(json.loads(alt_example.read_text(encoding="utf-8")))
    for extra in args.example:
        examples.append(json.loads(Path(extra).read_text(encoding="utf-8")))

    model = args.model or llm_settings.LLM_MODEL
    automation = build_with_retries(
        kept=kept,
        url=url,
        input_parameters=input_parameters,
        browser_channel=browser_channel,
        backend=backend,
        examples=examples,
        model=model,
        max_attempts=args.max_attempts,
    )

    out_path = Path(args.output)
    out_path.write_text(json.dumps(automation.model_dump(), indent=2), encoding="utf-8")
    logger.info(f"wrote {out_path}")

    if gt_path.exists():
        ground = Automation.model_validate(json.loads(gt_path.read_text(encoding="utf-8")))
        diffs = compare_to_ground_truth(automation, ground)
        report_path = Path(args.diff_report)
        lines = [
            "# LLM auto-builder vs hand-built ground truth",
            "",
            f"- input export: `{args.input}`",
            f"- LLM output: `{out_path}`",
            f"- ground truth: `{gt_path}`",
            f"- model: `{model}`",
            "",
        ]
        if not diffs:
            lines.append("No structural diffs in url / node kind / command / value / press_enter.")
            logger.info("ground-truth compare: MATCH")
        else:
            lines.append(f"Found {len(diffs)} diff(s):")
            lines.append("")
            for d in diffs:
                lines.append(f"- {d}")
            logger.info(f"ground-truth compare: {len(diffs)} diff(s)")
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info(f"wrote diff report {report_path}")
        print(report_path.read_text(encoding="utf-8"))
    else:
        logger.warning(f"ground truth not found at {gt_path}; skipped compare")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
