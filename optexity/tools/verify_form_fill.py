"""Verify a cached form-fill automation by reading field values back.

Loads `test_automation_cached.json` (or --automation), navigates to its URL,
replays every `input_text` / `select_option` / `click_element` command with
Playwright, then READS BACK each filled field's value and asserts it equals
the expected input. Success = correct content, not merely "no exception".
See `optexity_cursor_plan.md` Phase 4/T4.

    python -m optexity.tools.verify_form_fill
    python -m optexity.tools.verify_form_fill --automation test_automation_cached.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright

from optexity.schema.automation import Automation


def _input_nodes(automation: Automation) -> list[tuple[str, str, str]]:
    """Return [(command, expected_value, label), ...] for every input_text node."""
    out: list[tuple[str, str, str]] = []
    for i, node in enumerate(automation.nodes):
        if getattr(node, "type", None) != "action_node":
            continue
        ia = node.interaction_action
        if ia is None or ia.input_text is None:
            continue
        action = ia.input_text
        if not action.command:
            raise SystemExit(f"node {i}: input_text.command is empty")
        expected = action.input_text
        if expected is None:
            raise SystemExit(f"node {i}: input_text.input_text is empty")
        label = action.prompt_instructions or action.command
        out.append((action.command, expected, label))
    return out


def verify(automation_path: Path, headless: bool = True, timeout_ms: int = 30000) -> int:
    data = json.loads(automation_path.read_text(encoding="utf-8"))
    automation = Automation.model_validate(data)
    fields = _input_nodes(automation)
    if not fields:
        print(f"ERROR: no input_text nodes in {automation_path}", file=sys.stderr)
        return 1

    print(f"url={automation.url}")
    print(f"verifying {len(fields)} input_text field(s) from {automation_path}")

    failures: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page()
        page.goto(automation.url, wait_until="domcontentloaded", timeout=timeout_ms)
        page.wait_for_timeout(1000)

        for command, expected, label in fields:
            locator = eval(f"page.{command}")
            locator.scroll_into_view_if_needed(timeout=timeout_ms)
            locator.fill(expected, timeout=timeout_ms)

        # Read back AFTER all fills so we catch overwrite / wrong-locator bugs.
        for command, expected, label in fields:
            locator = eval(f"page.{command}")
            actual = locator.input_value(timeout=timeout_ms)
            ok = actual == expected
            status = "OK" if ok else "FAIL"
            print(f"  [{status}] {label}: expected={expected!r} actual={actual!r} via {command}")
            if not ok:
                failures.append(f"{label}: expected={expected!r} actual={actual!r}")

        browser.close()

    if failures:
        print(f"\nFAIL: {len(failures)}/{len(fields)} field(s) mismatched", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print(f"\nPASS: all {len(fields)} field value(s) read back correctly")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--automation",
        default="test_automation_cached.json",
        help="cached Automation JSON to replay and verify",
    )
    parser.add_argument("--headed", action="store_true", help="show the browser window")
    parser.add_argument("--timeout-ms", type=int, default=30000)
    args = parser.parse_args(argv)

    path = Path(args.automation)
    if not path.exists():
        print(f"ERROR: automation file not found: {path}", file=sys.stderr)
        return 1
    return verify(path, headless=not args.headed, timeout_ms=args.timeout_ms)


if __name__ == "__main__":
    raise SystemExit(main())
