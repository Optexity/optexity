"""LLM auto-builder for cached Optexity automations (Bonus A).

This module lets the LLM convert an agent step cache into a valid Optexity
automation JSON, paired with Pydantic validation and a validation-retry
loop so the output always validates.

Usage::

    python -m optexity.utils.llm_build_automation \\
        --cache agent_step_cache.json \\
        --out test_automation_cached_llm.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from browser_use.agent.step_cache import (
    AgentStepCache,
    CachedElement,
    playwright_command,
)
from browser_use.llm.messages import UserMessage
from optexity.inference.models.chat_litellm import build_agent_llm
from optexity.inference.models.llm_model import parse_json_from_completion
from optexity.schema.automation import Automation
from pydantic import ValidationError

logger = logging.getLogger(__name__)


def _load_cache(cache_path):
    return json.loads(Path(cache_path).read_text())


def _element_command(element):
    """Rebuild the locator via the authoritative builder shared with the
    deterministic converter (browser_use.agent.step_cache.playwright_command),
    so both converters emit byte-identical commands from the same cache."""
    if not isinstance(element, dict) or not element:
        return None
    try:
        return playwright_command(CachedElement.model_validate(element))
    except Exception:
        return None


def _steps_for_llm(cache):
    """Extract deterministic steps from cache in LLM-friendly format."""
    steps = []
    for s in cache.get("steps", []):
        if s.get("classification") != "deterministic":
            continue
        element = s.get("element", {})
        params = s.get("action_params", {})
        action_name = s.get("action_name", s.get("action", ""))
        command = s.get("command") or _element_command(element)
        input_text = s.get("input_text") or params.get("text")
        steps.append(
            {
                "index": s.get("step_index", s.get("index")),
                "url": s.get("url", ""),
                "action": action_name,
                "element": element,
                "command": command,
                "input_text": input_text,
                "prompt_instructions": s.get("prompt_instructions", ""),
                "classification": s.get("classification", ""),
                "reason": s.get("reason", ""),
            }
        )
    # Special-case nodes (e.g. clicks on inline-plaintext documents, which
    # must become script-based downloads instead of plain clicks) are
    # compiled by the shared rule engine and handed to the LLM verbatim, so
    # that knowledge lives in exactly one place.
    try:
        ref_nodes = AgentStepCache.model_validate(cache).to_optexity_automation_dict(
            cache.get("url") or cache.get("start_url", "")
        )["nodes"]
    except Exception:
        ref_nodes = []
    script_nodes = [n for n in ref_nodes if n.get("python_script_action")]
    if script_nodes:
        candidates = [
            i
            for i, s in enumerate(steps)
            if s["action"] == "click"
            and any(
                ext
                in str((s.get("element") or {}).get("attributes", {}).get("href", ""))
                for ext in (".txt", ".md", ".csv", ".log")
            )
        ]
        for node, i in zip(script_nodes, candidates):
            steps[i]["node"] = node
    return steps


async def llm_build_automation(cache, *, llm=None, max_retries=3):
    """Convert a step cache into a validated Automation via the LLM."""
    if llm is None:
        llm = build_agent_llm()

    url = cache.get("url") or cache.get("start_url", "")
    steps = _steps_for_llm(cache)
    steps_json = json.dumps(steps, indent=2)
    user_prompt = f"""Task: {cache.get('task', '')}
Start URL: {url}
Deterministic steps ({len(steps)}):
{steps_json}

Produce an Automation JSON object. Required top-level keys: url (string),
parameters (object: {{"input_parameters": {{}}, "generated_parameters": {{}}}}),
nodes (list). Map the steps 1:1: exactly one node per given step, in the
given order - do not invent, merge, reorder, or skip steps. Each node:
{{"type": "action_node", "interaction_action":
{{...}}}} with exactly ONE action set: input_text {{command, input_text,
prompt_instructions}}, click_element {{command, prompt_instructions}},
or go_to_url {{url, new_tab}}. A step with action "navigate" becomes
go_to_url using that step's own url as the target. Use the steps'
pre-computed command values verbatim as the command fields. Steps that
carry a "node" field are pre-compiled special cases: emit that exact node
object for that step instead of an interaction_action."""

    validation_error = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info("LLM build attempt %d/%d", attempt, max_retries)
            # NOTE: output_format is deliberately NOT passed. Structured
            # output makes ChatLiteLLM run the Automation schema through
            # browser-use SchemaOptimizer, whose recursive ref flattening
            # blows the stack on a schema this large (RecursionError). The
            # model returns plain JSON and Pydantic validation below remains
            # the correctness gate.
            result = await llm.ainvoke([UserMessage(content=user_prompt)])
            raw = result.completion
            if isinstance(raw, str):
                try:
                    automation = Automation.model_validate_json(raw)
                except ValidationError:
                    automation = parse_json_from_completion(raw, Automation)
                if isinstance(automation, dict):
                    automation = Automation.model_validate(automation)
            else:
                automation = raw
            logger.info("LLM build succeeded on attempt %d", attempt)
            return automation
        except Exception as e:
            validation_error = e
            logger.warning("Attempt %d failed: %s", attempt, e)
            user_prompt += (
                "\n\nPrevious attempt failed: "
                + type(e).__name__
                + ": "
                + str(e)
                + "\nPlease fix the output and produce a valid Automation."
            )

    raise ValueError(
        f"LLM build failed after {max_retries} attempts: {validation_error}"
    )


def main(argv=None):
    """CLI: build automation from cache via LLM."""
    parser = argparse.ArgumentParser(description="Build automation from cache via LLM.")
    parser.add_argument("--cache", required=True, help="Path to step cache JSON.")
    parser.add_argument("--out", required=True, help="Output automation JSON path.")
    parser.add_argument("--max-retries", type=int, default=3, help="Max LLM attempts.")
    parser.add_argument("--model", default=None, help="Override LLM model.")
    args = parser.parse_args(argv)

    cache = _load_cache(args.cache)
    llm = build_agent_llm(args.model) if args.model else build_agent_llm()
    automation = asyncio.run(
        llm_build_automation(cache, llm=llm, max_retries=args.max_retries)
    )
    out_path = Path(args.out)
    out_path.write_text(automation.model_dump_json(indent=2, exclude_none=True))
    print(f"Wrote automation to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
