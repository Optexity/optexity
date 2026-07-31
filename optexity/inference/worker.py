import asyncio
import os
import sys

from optexity.inference.core.run_automation import run_automation
from optexity.schema.enums import ExitCodes
from optexity.schema.task import Task


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
    task = Task.model_validate_json(sys.argv[1])
    unique_child_arn = sys.argv[2]
    child_process_id = int(sys.argv[3])
    cdp_url = sys.argv[4]
    max_tries = int(sys.argv[5]) if len(sys.argv) > 5 else 1

    try:
        await run_automation(
            task,
            unique_child_arn,
            child_process_id,
            cdp_url=cdp_url,
            max_tries=max_tries,
        )
    except Exception:
        _force_exit(ExitCodes.WORKER_CRASHED.value)

    if task.status == "success":
        _force_exit(ExitCodes.SUCCESS.value)
    if task.status == "killed":
        _force_exit(ExitCodes.AUTOMATION_KILLED.value)
    _force_exit(ExitCodes.AUTOMATION_FAILED.value)


if __name__ == "__main__":
    asyncio.run(main())
