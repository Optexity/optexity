"""Learning loop demo for the Optexity take-home assignment.

Workflow:
1. Run the agentic automation once (test_automation.json) — browser-use explores with LLM.
2. Step cache is written to the task log directory (step_cache.json).
3. cached_automation.json is auto-generated from the cache.
4. Copy cached_automation.json to test_automation_cached.json and re-run for a
   deterministic, low-token execution.

Usage (from repo root, with env activated):
    python -m optexity.examples.step_cache_learning agentic
    python -m optexity.examples.step_cache_learning cached
    python -m optexity.examples.step_cache_learning build-cache /path/to/step_cache.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from optexity.learning.step_cache import build_optexity_automation

from optexity.inference.core.run_automation import run_automation
from optexity.schema.automation import Automation
from optexity.schema.task import Task

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENTIC_AUTOMATION = REPO_ROOT / "test_automation.json"
CACHED_AUTOMATION = REPO_ROOT / "test_automation_cached.json"


def _load_automation(path: Path) -> Automation:
    with open(path, encoding="utf-8") as f:
        return Automation.model_validate(json.load(f))


def _build_task(automation: Automation) -> Task:
    return Task(
        task_id=str(uuid.uuid4()),
        user_id=str(uuid.uuid4()),
        recording_id=str(uuid.uuid4()),
        automation=automation,
        input_parameters={},
        unique_parameter_names=[],
        created_at=datetime.now(timezone.utc),
        status="queued",
    )


async def run_agentic() -> None:
    logger.info("Running agentic automation from %s", AGENTIC_AUTOMATION)
    task = _build_task(_load_automation(AGENTIC_AUTOMATION))
    await run_automation(task, 0)
    logger.info("Done. Inspect task logs for step_cache.json and cached_automation.json")


async def run_cached() -> None:
    if not CACHED_AUTOMATION.exists():
        raise FileNotFoundError(
            f"{CACHED_AUTOMATION} not found. Run the agentic flow first, then copy "
            "cached_automation.json from task logs or use `build-cache`."
        )
    logger.info("Running cached deterministic automation from %s", CACHED_AUTOMATION)
    task = _build_task(_load_automation(CACHED_AUTOMATION))
    await run_automation(task, 0)


def build_cached_automation(cache_path: Path, output_path: Path | None = None) -> Path:
    output = output_path or CACHED_AUTOMATION
    with open(cache_path, encoding="utf-8") as f:
        cache_data = json.load(f)
    automation = build_optexity_automation(cache_data)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(automation, f, indent=2)
    logger.info("Wrote %d node(s) to %s", len(automation.get("nodes", [])), output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Step cache learning demo")
    parser.add_argument(
        "command",
        choices=["agentic", "cached", "build-cache"],
        help="agentic=LLM run, cached=deterministic replay, build-cache=convert step_cache.json",
    )
    parser.add_argument("cache_path", nargs="?", help="Path to step_cache.json for build-cache")
    parser.add_argument("--output", help="Output path for build-cache")
    args = parser.parse_args()

    if args.command == "agentic":
        asyncio.run(run_agentic())
    elif args.command == "cached":
        asyncio.run(run_cached())
    else:
        if not args.cache_path:
            parser.error("build-cache requires a path to step_cache.json")
        build_cached_automation(Path(args.cache_path), Path(args.output) if args.output else None)


if __name__ == "__main__":
    main()
