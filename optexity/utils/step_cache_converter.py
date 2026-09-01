"""Build a deterministic Optexity automation from a browser-use agent step cache.

The browser-use fork records every step an agent takes into a step cache
(the memory layer, see ``browser_use.agent.step_cache``). This module converts
the deterministic subset of those steps into an Optexity automation validated
against the ``Automation`` schema, so the task can be replayed without any LLM
reasoning.

Usage:
    python -m optexity.utils.step_cache_converter \
        --cache agent_step_cache.json \
        --out test_automation_cached.json

    # optionally override the start url (defaults to the first cached url):
    python -m optexity.utils.step_cache_converter --cache cache.json --url https://...
"""

import argparse
import json
import sys
from pathlib import Path

from browser_use.agent.step_cache import AgentStepCache

from optexity.schema.automation import Automation


def build_cached_automation(
    cache_path: str | Path,
    url: str | None = None,
) -> Automation:
    """Load a step cache and build a schema-validated deterministic automation."""
    cache = AgentStepCache.load_from_file(cache_path)
    automation_dict = cache.to_optexity_automation_dict(url=url)
    # Pydantic validation against the Optexity schema: invalid commands fail fast
    return Automation.model_validate(automation_dict)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert a browser-use agent step cache into a deterministic Optexity automation"
    )
    parser.add_argument(
        "--cache", required=True, help="Path to the agent step cache JSON file"
    )
    parser.add_argument(
        "--out",
        default="test_automation_cached.json",
        help="Where to write the deterministic automation (default: test_automation_cached.json)",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Start URL override (defaults to the first URL seen during the cached run)",
    )
    args = parser.parse_args(argv)

    cache = AgentStepCache.load_from_file(args.cache)
    print(cache.summary())
    for step in cache.redundant_steps():
        print(f"  [skip] step {step.step_index}: {step.action_name} — {step.reason}")

    automation = build_cached_automation(args.cache, url=args.url)
    # Write the minimal dict (post-validation) to keep the file close to the
    # hand-written automation format rather than a full model_dump.
    minimal = cache.to_optexity_automation_dict(url=args.url)
    Path(args.out).write_text(
        json.dumps(minimal, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(automation.nodes)} deterministic nodes to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
