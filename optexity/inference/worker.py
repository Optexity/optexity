import asyncio
import json
import sys

from optexity.inference.core.run_automation import run_automation
from optexity.schema.enums import ExitCodes
from optexity.schema.task import Task


async def main():
    task = Task.model_validate_json(sys.argv[1])
    unique_child_arn = sys.argv[2]
    child_process_id = int(sys.argv[3])
    cdp_url = sys.argv[4] if sys.argv[4] != "None" else None
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
        sys.exit(ExitCodes.WORKER_CRASHED.value)

    if task.status == "success":
        sys.exit(ExitCodes.SUCCESS.value)
    if task.status == "killed":
        sys.exit(ExitCodes.AUTOMATION_KILLED.value)
    sys.exit(ExitCodes.AUTOMATION_FAILED.value)


if __name__ == "__main__":
    asyncio.run(main())
