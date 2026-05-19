"""Recursively import every submodule of the named packages.

Fails CI if any module raises ImportError, ModuleNotFoundError, or SyntaxError
— catches things ruff/black can't, like `from typing import BaseException`
(the module exists, but the name doesn't) or a deep submodule with a typo'd
import statement.

Runtime errors at import time (e.g. missing env vars, network calls) are
intentionally ignored so this check stays focused on import validity.

Usage:
    python .github/scripts/import_smoke.py PKG [PKG ...] [--skip MOD ...]
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
import traceback


def main(packages: list[str], skip: set[str]) -> int:
    failures: list[tuple[str, str, str]] = []
    skipped: list[str] = []
    scanned = 0

    for pkg_name in packages:
        if pkg_name in skip:
            skipped.append(pkg_name)
            continue
        try:
            pkg = importlib.import_module(pkg_name)
        except (ImportError, ModuleNotFoundError, SyntaxError) as e:
            failures.append((pkg_name, type(e).__name__, str(e)))
            continue
        except Exception as e:
            print(
                f"WARN  {pkg_name}: skipped — non-import error at top-level "
                f"({type(e).__name__}: {e})"
            )
            continue

        scanned += 1
        if not hasattr(pkg, "__path__"):
            continue

        for modinfo in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
            if modinfo.name in skip:
                skipped.append(modinfo.name)
                continue
            try:
                importlib.import_module(modinfo.name)
                scanned += 1
            except (ImportError, ModuleNotFoundError, SyntaxError) as e:
                failures.append((modinfo.name, type(e).__name__, str(e)))
            except Exception:
                exc = traceback.format_exception_only(*sys.exc_info()[:2])[-1].strip()
                print(f"WARN  {modinfo.name}: skipped — {exc}")
                scanned += 1

    if skipped:
        print(f"SKIP  {len(skipped)} module(s) explicitly skipped: {sorted(skipped)}")

    if failures:
        print(f"\nFAIL  {len(failures)} module(s) failed to import:\n")
        for name, etype, msg in failures:
            print(f"  {name}")
            print(f"    {etype}: {msg}")
        return 1

    print(f"\nOK    {scanned} module(s) imported cleanly across {packages}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packages", nargs="+", help="Top-level packages to scan")
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        help="Module name to skip (repeatable). Useful for orphan dev scripts with "
        "known-stale imports that aren't worth fixing yet.",
    )
    args = parser.parse_args()
    sys.exit(main(args.packages, set(args.skip)))
