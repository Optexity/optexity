import asyncio
import json
import sys

from optexity.inference.core.run_automation import run_automation
from optexity.schema.task import Task


async def main():
    task = Task.model_validate_json(sys.argv[1])
    unique_child_arn = sys.argv[2]
    child_process_id = int(sys.argv[3])

    await run_automation(task, unique_child_arn, child_process_id)


if __name__ == "__main__":
    asyncio.run(main())
