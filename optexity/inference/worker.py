import asyncio
import os
import sys
import traceback

from optexity.inference.core.run_automation import run_automation
from optexity.private_nodes import load_plugins
from optexity.schema.enums import ExitCodes
from optexity.schema.task import Task

# The parent reads this file to report a crash; stderr only reaches the container log.
WORKER_ERROR_FILE_ENV = "WORKER_ERROR_FILE"
WORKER_ERROR_MAX_CHARS = 1000


def _write_crash_error(exc: BaseException) -> None:
    path = os.environ.get(WORKER_ERROR_FILE_ENV)
    if not path:
        return
    try:
        with open(path, "w") as f:
            f.write(f"{type(exc).__name__}: {exc}"[:WORKER_ERROR_MAX_CHARS])
    except OSError as e:
        # The parent falls back to the exit code, so only note it.
        print(f"Could not write worker error file {path}: {e}", file=sys.stderr)


def _force_exit(code: int) -> None:
    """Exit immediately, ignoring leftover non-daemon threads (Playwright/Chrome).

    ``sys.exit`` can hang if browser teardown left non-daemon threads alive; the
    parent then hits ``max_timeout_in_minutes`` and overwrites a successful
    completion as killed. Task post-processing already finished inside
    ``run_automation`` before this is called.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


async def main():
    try:
        # Nodes execute in this process, so private_node handlers must be registered
        # here — registering them in the parent service would not reach the executor.
        load_plugins()

        task = Task.model_validate_json(sys.argv[1])
        unique_child_arn = sys.argv[2]
        child_process_id = int(sys.argv[3])
        cdp_url = sys.argv[4]
        max_tries = int(sys.argv[5]) if len(sys.argv) > 5 else 1

        outcome = await run_automation(
            task,
            unique_child_arn,
            child_process_id,
            cdp_url=cdp_url,
            max_tries=max_tries,
        )
    except Exception as e:
        traceback.print_exc()
        _write_crash_error(e)
        _force_exit(ExitCodes.WORKER_CRASHED.value)

    if outcome == ExitCodes.BROWSER_UNRESPONSIVE:
        _force_exit(ExitCodes.BROWSER_UNRESPONSIVE.value)
    if task.status == "success":
        _force_exit(ExitCodes.SUCCESS.value)
    if task.status == "killed":
        _force_exit(ExitCodes.AUTOMATION_KILLED.value)
    _force_exit(ExitCodes.AUTOMATION_FAILED.value)


if __name__ == "__main__":
    asyncio.run(main())
