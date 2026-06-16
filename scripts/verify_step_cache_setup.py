#!/usr/bin/env python3
"""Verify step-cache setup for the take-home assignment."""

from __future__ import annotations

import importlib.util
import inspect
import subprocess
import sys


def _pkg_installed(name: str) -> bool:
    return (
        subprocess.run(
            [sys.executable, "-m", "pip", "show", name],
            capture_output=True,
        ).returncode
        == 0
    )


def main() -> int:
    print(f"Python: {sys.executable}")
    errors: list[str] = []
    warnings: list[str] = []

    if _pkg_installed("optexity-browser-use"):
        errors.append(
            "optexity-browser-use is installed and shadows your browser-use fork.\n"
            "      Fix:\n"
            "        pip uninstall -y optexity-browser-use\n"
            "        pip install -e /Users/rreddy/Documents/repana/browser-use"
        )

    try:
        spec = importlib.util.find_spec("browser_use")
        origin = getattr(spec, "origin", None) if spec else None
        if origin and "site-packages/browser_use" in origin.replace("\\", "/"):
            warnings.append(
                f"browser_use loads from site-packages ({origin}), not your fork.\n"
                "      Re-run: pip install -e /Users/rreddy/Documents/repana/browser-use"
            )
    except Exception:
        pass

    try:
        from optexity.learning.step_cache import build_optexity_automation

        print("OK  optexity.learning.step_cache")
        _ = build_optexity_automation
    except Exception as e:
        errors.append(f"optexity.learning: {e}")

    try:
        import browser_use

        print(f"OK  browser_use ({browser_use.__file__})")
    except Exception as e:
        errors.append(f"browser_use: {e}")
        browser_use = None  # type: ignore
    else:
        try:
            from browser_use.learning.step_cache import save_step_cache

            print("OK  browser_use.learning.step_cache (fork)")
            _ = save_step_cache
        except Exception as e:
            errors.append(
                f"browser_use.learning missing: {e}\n"
                "      pip install -e /Users/rreddy/Documents/repana/browser-use"
            )

        try:
            from browser_use.agent.service import Agent

            if "step_cache_path" in inspect.signature(Agent.__init__).parameters:
                print("OK  Agent.step_cache_path")
            else:
                errors.append("Agent missing step_cache_path — browser-use fork not active")
        except Exception as e:
            errors.append(f"Agent inspect failed: {e}")

    # Import chain used by worker.py (needs env vars for settings)
    try:
        from optexity.inference.core.interaction import handle_agentic_task as _m

        print("OK  worker import chain (handle_agentic_task)")
        _ = _m
    except Exception as e:
        if "OPTEXITY_API_KEY" in str(e) or "DEPLOYMENT" in str(e):
            print("OK  worker import chain (settings env vars not set — expected)")
        else:
            errors.append(f"worker import chain: {e}")

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  ⚠ {w}")

    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  ✗ {e}")
        return 1

    import os

    if not os.getenv("GOOGLE_API_KEY"):
        print("\nNote: GOOGLE_API_KEY is not set — agentic browser-use runs will fail until it is set.")

    print("\n✓ Setup OK. Run inference with THIS python:")
    print("  cd /Users/rreddy/Documents/repana/optexity")
    print("  source .venv/bin/activate")
    print("  export OPTEXITY_API_KEY=... GOOGLE_API_KEY=... DEPLOYMENT=dev")
    print("  optexity inference --port 9000 --child_process_id 0")
    print("\nWith test_automation.json present + DEPLOYMENT=dev, /inference uses")
    print("local test_automation.json (any endpoint_name/input_parameters work).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
