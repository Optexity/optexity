"""CLI: build a schema-valid cached Automation JSON from a Phase 1/2 export.

Reads the action records exported by the browser-use fork's
`browser_use/cache/action_export.py` (Phase 1), runs them through the
rule-based filter in `browser_use/cache/step_filter.py` (Phase 2), resolves a
stable Playwright locator for each kept step, and emits a schema-valid
`Automation` JSON that replays without any LLM calls. See
`optexity_cursor_plan.md` Phase 3/T3.

Locator tiers (first match wins, in this order):
  1. id                                -> locator("#<id>") or locator('[id="<id>"]')
  2. name                              -> locator('[name="<name>"]')
  3. placeholder / aria-label / data-testid attribute
  4. role + visible text               -> get_by_role("<role>", name="<text>")
  5. xpath (ONLY if not in_shadow_dom) -> locator("xpath=<xpath>")
Shadow-DOM elements that miss tiers 1-4 fail loudly (LocatorResolutionError)
instead of falling back to xpath, which does not pierce shadow roots in
Playwright (hard rule 2 in optexity_cursor_plan.md).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

from browser_use.cache.step_filter import filter_steps

from optexity.schema.actions.interaction_action import (
    ClickElementAction,
    GoToUrlAction,
    InputTextAction,
    InteractionAction,
    SelectOptionAction,
)
from optexity.schema.automation import ActionNode, Automation, Parameters

logger = logging.getLogger(__name__)

_CSS_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")
_MASKED_VALUE = "***MASKED***"


class LocatorResolutionError(RuntimeError):
    """Raised when no safe locator tier matches an element."""


def _attr_selector(name: str, value: str) -> str:
    return f'[{name}={value!r}]' if "'" not in value else f'[{name}="{value}"]'


def resolve_locator(element: dict[str, Any] | None) -> tuple[str, str, str]:
    """Return (playwright_command, tier, source_attribute) for one element.

    `tier`/`source_attribute` are for --trace only; `playwright_command` is
    what gets embedded in the emitted automation's `command` field.
    """
    if not element:
        raise LocatorResolutionError("element block is empty; cannot build a locator")

    elem_id = element.get("id")
    if elem_id:
        selector = f"#{elem_id}" if _CSS_IDENT.match(elem_id) else _attr_selector("id", elem_id)
        return f"locator({selector!r}).first", "tier1:id", f"id={elem_id!r}"

    name = element.get("name")
    if name:
        return (
            f"locator({_attr_selector('name', name)!r}).first",
            "tier2:name",
            f"name={name!r}",
        )

    placeholder = element.get("placeholder")
    if placeholder:
        return (
            f"locator({_attr_selector('placeholder', placeholder)!r}).first",
            "tier3:placeholder",
            f"placeholder={placeholder!r}",
        )

    aria_label = element.get("aria_label")
    if aria_label:
        return (
            f"locator({_attr_selector('aria-label', aria_label)!r}).first",
            "tier3:aria-label",
            f"aria_label={aria_label!r}",
        )

    data_testid = (element.get("data_attrs") or {}).get("data-testid")
    if data_testid:
        return (
            f"locator({_attr_selector('data-testid', data_testid)!r}).first",
            "tier3:data-testid",
            f"data_attrs.data-testid={data_testid!r}",
        )

    role = element.get("role")
    visible_text = element.get("visible_text")
    if role and visible_text:
        return (
            f"get_by_role({role!r}, name={visible_text!r}).first",
            "tier4:role+text",
            f"role={role!r} visible_text={visible_text!r}",
        )

    if element.get("in_shadow_dom"):
        raise LocatorResolutionError(
            "element is in_shadow_dom and has no id/name/placeholder/aria-label/"
            "data-testid/role+text match; refusing to emit xpath because xpath "
            "does not pierce shadow roots in Playwright (hard rule 2)"
        )

    x_path = element.get("x_path")
    if x_path:
        return f"locator({('xpath=' + x_path)!r}).first", "tier5:xpath", f"x_path={x_path!r}"

    raise LocatorResolutionError(f"no locator tier matched element: {element!r}")


def _prompt_for(action_type: str, element: dict[str, Any] | None, typed_value: str | None) -> str:
    label = None
    if element:
        label = element.get("aria_label") or element.get("visible_text") or element.get("name") or element.get("id")
    label = label or "the target element"
    if action_type == "input_text":
        return f"Fill '{label}' with '{typed_value}'"
    if action_type == "click":
        return f"Click '{label}'"
    if action_type == "select":
        return f"Select '{typed_value}' in '{label}'"
    return f"Interact with '{label}'"


def build_action_node(record: dict[str, Any], trace: list[str]) -> ActionNode:
    action_type = record["action_type"]
    element = record.get("element")
    typed_value = record.get("typed_value")
    step_index = record.get("step_index")

    if action_type == "navigate":
        interaction_action = InteractionAction(go_to_url=GoToUrlAction(url=typed_value))
        trace.append(f"step={step_index} action=navigate -> go_to_url(url={typed_value!r}) source=typed_value")
    else:
        if typed_value == _MASKED_VALUE:
            logger.warning(
                f"step={step_index}: typed_value is masked ({_MASKED_VALUE}); the emitted "
                "command will literally type that placeholder. Replace it by hand with a "
                "SecureParameter (see optexity/schema/automation.py:SecureParameter) before "
                "using this cached automation."
            )

        command, tier, source = resolve_locator(element)
        prompt_instructions = _prompt_for(action_type, element, typed_value)
        trace.append(f"step={step_index} action={action_type} -> {tier} -> {command} source={source}")

        if action_type == "input_text":
            interaction_action = InteractionAction(
                input_text=InputTextAction(
                    command=command, input_text=typed_value, prompt_instructions=prompt_instructions
                )
            )
        elif action_type == "click":
            interaction_action = InteractionAction(
                click_element=ClickElementAction(command=command, prompt_instructions=prompt_instructions)
            )
        elif action_type == "select":
            interaction_action = InteractionAction(
                select_option=SelectOptionAction(
                    command=command,
                    select_values=[typed_value] if typed_value is not None else None,
                    prompt_instructions=prompt_instructions,
                )
            )
        else:
            raise LocatorResolutionError(f"unsupported action_type for cached replay: {action_type!r}")

    node_kwargs: dict[str, Any] = {"type": "action_node", "interaction_action": interaction_action}
    if record.get("caused_navigation"):
        # ActionNode already sleeps `end_sleep_time` seconds after every action
        # (schema default 5.0s, capped at 30s) - that existing field is the
        # wait mechanism, so post-navigation settling needs no new node type
        # and no sleep_action/sleep() call. Set it explicitly here (even
        # though 5.0 is already the default) so the wait is visible in the
        # emitted JSON and in --trace instead of relying on an implicit default.
        node_kwargs["end_sleep_time"] = 5.0
        trace.append(f"step={step_index}: caused_navigation=True -> end_sleep_time=5.0 (built-in post-action wait)")

    return ActionNode(**node_kwargs)


def build_automation(
    kept: list[dict[str, Any]], url: str, browser_channel: str, backend: str, trace: list[str]
) -> Automation:
    nodes = [build_action_node(record, trace) for record in kept]
    return Automation(
        url=url,
        browser_channel=browser_channel,
        backend=backend,
        parameters=Parameters(input_parameters={}, generated_parameters={}),
        nodes=nodes,
        automation_description="Deterministic cache built by build_cached_automation.py from a filtered browser-use export.",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Phase 1 export JSON (cached_run_<ts>.json)")
    parser.add_argument("--output", default="test_automation_cached.json", help="output path for the cached Automation JSON")
    parser.add_argument(
        "--base-automation",
        default="test_automation.json",
        help="source of url/browser_channel/backend defaults (the agentic test_automation.json)",
    )
    parser.add_argument("--url", default=None, help="override the automation url instead of reading --base-automation")
    parser.add_argument("--trace", action="store_true", help="print step -> chosen locator -> source attribute")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    input_path = Path(args.input)
    records = json.loads(input_path.read_text(encoding="utf-8"))

    kept, discarded = filter_steps(records)
    logger.info(f"filter_steps: total={len(records)} kept={len(kept)} discarded={len(discarded)}")

    url = args.url
    browser_channel = "chromium"
    backend = "browser-use"
    base_path = Path(args.base_automation)
    if url is None:
        if not base_path.exists():
            parser.error(f"--url not given and base automation not found at {base_path}")
        base = json.loads(base_path.read_text(encoding="utf-8"))
        url = base["url"]
        browser_channel = base.get("browser_channel", browser_channel)
        backend = base.get("backend", backend)

    trace: list[str] = []
    try:
        automation = build_automation(kept, url=url, browser_channel=browser_channel, backend=backend, trace=trace)
    except LocatorResolutionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        Automation.model_validate(automation.model_dump())
    except Exception as exc:  # pydantic ValidationError, printed verbatim for the operator
        print(f"ERROR: cached automation failed Automation.model_validate: {exc}", file=sys.stderr)
        return 1

    output_path = Path(args.output)
    output_path.write_text(json.dumps(automation.model_dump(), indent=2), encoding="utf-8")
    logger.info(f"wrote {len(automation.nodes)} node(s) to {output_path}")

    if args.trace:
        print("\n--- trace: step -> tier -> command -> source ---")
        for line in trace:
            print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
