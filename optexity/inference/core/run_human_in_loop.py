import asyncio
import logging
from urllib.parse import urljoin

import httpx

from optexity.exceptions import HumanInLoopTimeoutException
from optexity.schema.actions.misc_action import HumanInLoopAction
from optexity.schema.memory import Memory
from optexity.schema.task import Task
from optexity.utils.settings import settings

logger = logging.getLogger(__name__)


async def run_human_in_loop_action(
    human_in_loop_action: HumanInLoopAction,
    task: Task,
    memory: Memory,
) -> None:
    await notify_human_in_loop(task, memory)

    elapsed = 0
    interval = 2.0
    while elapsed < human_in_loop_action.max_wait_time:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"http://localhost:{settings.CHILD_PORT_OFFSET}/human_in_loop_status",
                params={
                    "unique_child_arn": memory.unique_child_arn,
                    "task_id": task.task_id,
                },
            )
            data = resp.json()
            if data.get("completed"):
                return

        await asyncio.sleep(interval)
        elapsed += interval

    raise HumanInLoopTimeoutException(
        f"Human-in-loop timeout: no completion signal received after {human_in_loop_action.max_wait_time} seconds for task {task.task_id}."
    )


async def notify_human_in_loop(task: Task, memory: Memory):
    url = urljoin(settings.SERVER_URL, settings.HUMAN_IN_LOOP_ENDPOINT)
    headers = {
        "x-api-key": task.api_key,
    }
    body = {
        "unique_child_arn": memory.unique_child_arn,
        "task_id": task.task_id,
    }

    logger.info(
        "Calling human-in-the-loop endpoint for task %s and unique_child_arn %s",
        task.task_id,
        memory.unique_child_arn,
    )

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=body)
            response.raise_for_status()
            logger.debug("Human-in-loop response: %s", response.text)
    except Exception as e:
        logger.error("Error calling human-in-loop endpoint: %s", e)
