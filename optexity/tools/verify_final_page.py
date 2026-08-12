"""Verify a multi-step cached automation landed on the expected final page.

Unlike verify_form_fill (field values), this asserts final URL and/or a known
on-page signal after replaying the cached Automation via /inference — or, for
a standalone check, by navigating directly to --expect-url. See
`optexity_cursor_plan.md` Phase 5/T5.

Standalone mode (no inference; useful for locator smoke tests):

    python -m optexity.tools.verify_final_page \\
      --expect-url-substr /stocks/nvda/financials \\
      --expect-text "Financials"

Post-inference mode reads the live browser is not available (browser closes),
so the intended T5 check is: after each cached /inference success, run this
script in --goto mode against the known final URL the automation should have
reached, OR use --replay-automation to drive Playwright with the cached
commands and then assert URL/text.

    python -m optexity.tools.verify_final_page \\
      --automation test_automation_2_cached.json \\
      --expect-url-substr /stocks/nvda/financials \\
      --expect-text "Financials"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright

from optexity.schema.automation import Automation


def _action_commands(automation: Automation) -> list[tuple[str, str, dict[str, Any]]]:
    """Return [(kind, command_or_url, extras), ...] for replayable nodes."""
    out: list[tuple[str, str, dict[str, Any]]] = []
    for i, node in enumerate(automation.nodes):
        if getattr(node, "type", None) != "action_node":
            continue
        ia = node.interaction_action
        if ia is None:
            continue
        if ia.go_to_url is not None and ia.go_to_url.url:
            out.append(("go_to_url", ia.go_to_url.url, {"end_sleep_time": node.end_sleep_time}))
        elif ia.click_element is not None and ia.click_element.command:
            out.append(
                (
                    "click",
                    ia.click_element.command,
                    {"end_sleep_time": node.end_sleep_time},
                )
            )
        elif ia.input_text is not None and ia.input_text.command:
            out.append(
                (
                    "input_text",
                    ia.input_text.command,
                    {
                        "text": ia.input_text.input_text or "",
                        "press_enter": bool(ia.input_text.press_enter),
                        "end_sleep_time": node.end_sleep_time,
                    },
                )
            )
        elif ia.select_option is not None and ia.select_option.command:
            vals = ia.select_option.select_values or []
            out.append(
                (
                    "select",
                    ia.select_option.command,
                    {
                        "values": vals,
                        "end_sleep_time": node.end_sleep_time,
                    },
                )
            )
        else:
            raise SystemExit(f"node {i}: no replayable interaction (click/input/select/go_to_url)")
    return out


def replay_and_verify(
    automation_path: Path,
    expect_url_substr: str,
    expect_text: str | None,
    headless: bool,
    timeout_ms: int,
) -> int:
    data = json.loads(automation_path.read_text(encoding="utf-8"))
    automation = Automation.model_validate(data)
    steps = _action_commands(automation)
    if not steps:
        print(f"ERROR: no replayable nodes in {automation_path}", file=sys.stderr)
        return 1

    print(f"start_url={automation.url}")
    print(f"replaying {len(steps)} step(s) from {automation_path}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page()
        page.goto(automation.url, wait_until="domcontentloaded", timeout=timeout_ms)

        for kind, command, extras in steps:
            sleep_s = float(extras.get("end_sleep_time") or 0)
            if kind == "go_to_url":
                print(f"  go_to_url {command}")
                page.goto(command, wait_until="domcontentloaded", timeout=timeout_ms)
            elif kind == "click":
                print(f"  click {command}")
                locator = eval(f"page.{command}")
                locator.scroll_into_view_if_needed(timeout=timeout_ms)
                # Prefer navigation-aware click when the action is expected to navigate.
                try:
                    with page.expect_navigation(wait_until="domcontentloaded", timeout=timeout_ms):
                        locator.click(timeout=timeout_ms)
                except Exception:
                    locator.click(timeout=timeout_ms)
                    page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
            elif kind == "input_text":
                text = extras["text"]
                print(f"  input {command!r} <- {text!r} press_enter={extras.get('press_enter')}")
                locator = eval(f"page.{command}")
                locator.scroll_into_view_if_needed(timeout=timeout_ms)
                locator.fill(text, timeout=timeout_ms)
                if extras.get("press_enter"):
                    locator.press("Enter")
            elif kind == "select":
                print(f"  select {command}")
                locator = eval(f"page.{command}")
                locator.select_option(extras.get("values") or [], timeout=timeout_ms)
            if sleep_s > 0:
                page.wait_for_timeout(int(sleep_s * 1000))

        final_url = page.url
        body_text = page.locator("body").inner_text(timeout=timeout_ms)
        browser.close()

    print(f"final_url={final_url}")
    ok_url = expect_url_substr.lower() in final_url.lower()
    print(f"  [{'OK' if ok_url else 'FAIL'}] expect_url_substr={expect_url_substr!r}")
    ok_text = True
    if expect_text:
        ok_text = expect_text.lower() in body_text.lower()
        print(f"  [{'OK' if ok_text else 'FAIL'}] expect_text={expect_text!r}")

    if ok_url and ok_text:
        print("\nPASS: final page matches expected URL/content")
        return 0
    print("\nFAIL: final page did not match expectations", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--automation", default="test_automation_2_cached.json")
    parser.add_argument(
        "--expect-url-substr",
        default="/stocks/nvda/financials",
        help="substring that must appear in the final URL (case-insensitive)",
    )
    parser.add_argument(
        "--expect-text",
        default="Financials",
        help="substring that must appear in the final page body (empty to skip)",
    )
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--timeout-ms", type=int, default=60000)
    args = parser.parse_args(argv)

    path = Path(args.automation)
    if not path.exists():
        print(f"ERROR: automation file not found: {path}", file=sys.stderr)
        return 1
    expect_text = args.expect_text or None
    return replay_and_verify(
        path,
        expect_url_substr=args.expect_url_substr,
        expect_text=expect_text,
        headless=not args.headed,
        timeout_ms=args.timeout_ms,
    )


if __name__ == "__main__":
    raise SystemExit(main())
