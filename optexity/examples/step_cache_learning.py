"""Iterative cache-learning demo for the Optexity take-home assignment.

Workflow:
1. Start from a simple `agentic_task` automation (browser-use + LLM).
2. browser-use writes `step_cache.json` in task logs.
3. Optexity converts that cache to deterministic `cached_automation.json`.
4. Re-run in a loop and persist per-iteration artifacts/metrics.

Usage (from repo root, with env activated):
    python -m optexity.examples.step_cache_learning iterate --iterations 3
    python -m optexity.examples.step_cache_learning agentic
    python -m optexity.examples.step_cache_learning cached
    python -m optexity.examples.step_cache_learning build-cache /path/to/step_cache.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv

from optexity.inference.core.run_automation import run_automation
from optexity.inference.infra.actual_browser import ActualBrowser
from optexity.learning.step_cache import build_optexity_automation
from optexity.schema.automation import Automation
from optexity.schema.task import Task
from optexity.utils.settings import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENTIC_AUTOMATION = REPO_ROOT / "test_automation.json"
CACHED_AUTOMATION = REPO_ROOT / "test_automation_cached.json"
ITERATION1_AUTOMATION = REPO_ROOT / "test_automation_iteration1.json"
ITERATION_OUTPUT_DIR = REPO_ROOT / "cache_iterations"

_env_path = os.getenv("ENV_PATH")
if _env_path:
    load_dotenv(_env_path)
else:
    load_dotenv(REPO_ROOT / ".env")

if not os.getenv("GOOGLE_API_KEY"):
    for alias in ("GEMINI_API_KEY", "GOOGLE_GENAI_API_KEY"):
        value = os.getenv(alias)
        if value:
            os.environ["GOOGLE_API_KEY"] = value
            break

if not os.getenv("OPTEXITY_API_KEY") and os.getenv("API_KEY"):
    os.environ["OPTEXITY_API_KEY"] = os.environ["API_KEY"]


def _validate_runtime_env(require_llm: bool) -> None:
    missing: list[str] = []
    if not (os.getenv("OPTEXITY_API_KEY") or os.getenv("API_KEY")):
        missing.append("OPTEXITY_API_KEY (or API_KEY)")
    if not os.getenv("DEPLOYMENT"):
        missing.append("DEPLOYMENT")
    if require_llm and not os.getenv("GOOGLE_API_KEY"):
        missing.append("GOOGLE_API_KEY")
    if missing:
        raise RuntimeError(
            "Missing required environment variable(s): "
            + ", ".join(missing)
            + ". Add them to your shell, set ENV_PATH to an env file, or create "
            + f"{REPO_ROOT / '.env'}."
        )


def _load_automation(path: Path) -> Automation:
    with open(path, encoding="utf-8") as f:
        return Automation.model_validate(json.load(f))


def _build_task(automation: Automation) -> Task:
    return Task(
        task_id=str(uuid.uuid4()),
        user_id=str(uuid.uuid4()),
        recording_id=str(uuid.uuid4()),
        endpoint_name="local_step_cache_learning",
        automation=automation,
        input_parameters={},
        secure_parameters={},
        unique_parameter_names=[],
        created_at=datetime.now(timezone.utc),
        status="queued",
        api_key=settings.OPTEXITY_API_KEY,
        company_id="local-dev-company",
        local_test_override=True,
    )


async def _run_task(task: Task) -> float:
    unique_child_arn = f"step-cache-{task.task_id[:8]}"
    child_process_id = int(os.getenv("OPTEXITY_CHILD_PROCESS_ID", "0"))
    actual_browser = ActualBrowser(
        channel=task.automation.browser_channel,
        unique_child_arn=unique_child_arn,
        port=9222 + child_process_id,
        headless=os.getenv("OPTEXITY_HEADLESS", "false").lower() == "true",
        is_dedicated=False,
        use_proxy=task.use_proxy,
        os_emulation=task.automation.os_emulation,
        allow_cookies=task.automation.allow_cookies,
    )
    start = perf_counter()
    try:
        await actual_browser.start()
        if actual_browser.cdp_url is None:
            raise RuntimeError("Browser started but CDP URL is missing")
        await run_automation(
            task=task,
            unique_child_arn=unique_child_arn,
            child_process_id=child_process_id,
            cdp_url=actual_browser.cdp_url,
        )
    finally:
        await actual_browser.stop(graceful=True)
    return perf_counter() - start


def _latest_cached_automation_path(task: Task) -> Path | None:
    if not task.logs_directory.exists():
        return None
    step_dirs = sorted(task.logs_directory.glob("step_*"))
    for step_dir in reversed(step_dirs):
        candidate = step_dir / "cached_automation.json"
        if candidate.exists():
            return candidate
    return None


async def run_agentic() -> None:
    _validate_runtime_env(require_llm=True)
    logger.info("Running agentic automation from %s", ITERATION1_AUTOMATION)
    task = _build_task(_load_automation(ITERATION1_AUTOMATION))
    elapsed = await _run_task(task)
    logger.info("Done in %.2fs. Inspect %s", elapsed, task.logs_directory)


async def run_cached() -> None:
    _validate_runtime_env(require_llm=False)
    if not CACHED_AUTOMATION.exists():
        raise FileNotFoundError(
            f"{CACHED_AUTOMATION} not found. Run the agentic flow first, then copy "
            "cached_automation.json from task logs or use `build-cache`."
        )
    logger.info("Running cached deterministic automation from %s", CACHED_AUTOMATION)
    task = _build_task(_load_automation(CACHED_AUTOMATION))
    elapsed = await _run_task(task)
    logger.info("Done in %.2fs", elapsed)


def build_cached_automation(cache_path: Path, output_path: Path | None = None) -> Path:
    output = output_path or CACHED_AUTOMATION
    with open(cache_path, encoding="utf-8") as f:
        cache_data = json.load(f)
    automation = build_optexity_automation(cache_data)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(automation, f, indent=2)
    logger.info("Wrote %d node(s) to %s", len(automation.get("nodes", [])), output)
    return output


def _automation_requires_llm(automation: Automation) -> bool:
    for node in automation.nodes:
        if node.interaction_action and node.interaction_action.agentic_task:
            return True
    return False


async def run_iterative_learning(
    iterations: int,
    seed_automation_path: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_automation = _load_automation(seed_automation_path)
    current_automation_path = seed_automation_path
    run_summary: list[dict[str, str | int | float | None]] = []
    _validate_runtime_env(require_llm=_automation_requires_llm(seed_automation))

    logger.info(
        "Starting iterative learning loop: iterations=%d seed=%s",
        iterations,
        seed_automation_path,
    )

    for i in range(1, iterations + 1):
        input_automation_path = current_automation_path
        logger.info("Iteration %d/%d using %s", i, iterations, input_automation_path)
        task = _build_task(_load_automation(input_automation_path))
        elapsed_seconds = await _run_task(task)
        cached_automation_path = _latest_cached_automation_path(task)

        iteration_cached_output = output_dir / f"iteration_{i}_cached.json"
        nodes_count: int | None = None
        if cached_automation_path:
            with open(cached_automation_path, encoding="utf-8") as in_f:
                cached_data = json.load(in_f)
            nodes_count = len(cached_data.get("nodes", []))
            if nodes_count == 0:
                logger.warning(
                    "Iteration %d produced an empty cached automation; keeping previous automation.",
                    i,
                )
            else:
                with open(iteration_cached_output, "w", encoding="utf-8") as out_f:
                    json.dump(cached_data, out_f, indent=2)
                current_automation_path = iteration_cached_output
                logger.info(
                    "Iteration %d cached automation saved to %s (%d nodes)",
                    i,
                    iteration_cached_output,
                    nodes_count,
                )
        else:
            logger.warning(
                "Iteration %d produced no cached_automation.json. Reusing previous automation.",
                i,
            )

        run_summary.append(
            {
                "iteration": i,
                "input_automation": str(input_automation_path),
                "cached_output": str(iteration_cached_output)
                if cached_automation_path and nodes_count
                else None,
                "elapsed_seconds": round(elapsed_seconds, 3),
                "cached_nodes_count": nodes_count,
                "task_logs_directory": str(task.logs_directory),
            }
        )

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)
    logger.info("Iterative learning complete. Summary written to %s", summary_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Step cache learning demo")
    parser.add_argument(
        "command",
        choices=["agentic", "cached", "build-cache", "iterate"],
        help="agentic=LLM run, cached=deterministic replay, build-cache=convert step_cache.json",
    )
    parser.add_argument("cache_path", nargs="?", help="Path to step_cache.json for build-cache")
    parser.add_argument("--output", help="Output path for build-cache")
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of loop iterations for the iterate command.",
    )
    parser.add_argument(
        "--seed",
        default=str(ITERATION1_AUTOMATION),
        help="Seed automation path for iterate command.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ITERATION_OUTPUT_DIR),
        help="Directory where iterative cached automations are written.",
    )
    args = parser.parse_args()

    if args.command == "agentic":
        asyncio.run(run_agentic())
    elif args.command == "cached":
        asyncio.run(run_cached())
    elif args.command == "build-cache":
        if not args.cache_path:
            parser.error("build-cache requires a path to step_cache.json")
        build_cached_automation(Path(args.cache_path), Path(args.output) if args.output else None)
    else:
        if args.iterations < 1:
            parser.error("--iterations must be >= 1")
        asyncio.run(
            run_iterative_learning(
                iterations=args.iterations,
                seed_automation_path=Path(args.seed),
                output_dir=Path(args.output_dir),
            )
        )


if __name__ == "__main__":
    main()
