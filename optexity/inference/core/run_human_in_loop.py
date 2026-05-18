import asyncio
import logging
import os
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
    """
    Pause the automation for human takeover.

    1. Notifies opcloud (which emails the task owner a link to the live stream).
    2. Polls child_process.py's /hitl_status endpoint every 2 seconds until
       the human signals completion or max_wait_time elapses.
    3. Raises HumanInLoopTimeoutException if no completion signal arrives in
       time (the caller's retry/fail logic then handles the task outcome).
    """
    await _notify_human_in_loop(task, memory)

    child_fastapi_port = int(
        os.environ.get("CHILD_FASTAPI_PORT", str(settings.CHILD_PORT_OFFSET))
    )
    status_url = f"http://localhost:{child_fastapi_port}/hitl_status"

    elapsed = 0.0
    interval = 2.0
    async with httpx.AsyncClient(timeout=5.0) as client:
        while elapsed < human_in_loop_action.max_wait_time:
            try:
                resp = await client.get(status_url, params={"task_id": task.task_id})
                if resp.json().get("completed"):
                    logger.info(
                        "HITL completed for task %s after %.0f s",
                        task.task_id,
                        elapsed,
                    )
                    return
            except Exception as e:
                logger.warning(
                    "HITL status poll error for task %s: %s", task.task_id, e
                )

            await asyncio.sleep(interval)
            elapsed += interval

    raise HumanInLoopTimeoutException(
        f"Human-in-loop timeout: no completion signal received after "
        f"{human_in_loop_action.max_wait_time} seconds for task {task.task_id}."
    )


async def _notify_human_in_loop(task: Task, memory: Memory) -> None:
    url = urljoin(settings.SERVER_URL, settings.HUMAN_IN_LOOP_ENDPOINT)
    headers = {"x-api-key": task.api_key}
    body = {"task_id": task.task_id}

    logger.info(
        "Notifying opcloud of HITL for task %s (unique_child_arn=%s)",
        task.task_id,
        memory.unique_child_arn,
    )
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(url, headers=headers, json=body)
        response.raise_for_status()
        logger.debug("HITL notify response: %s", response.text)
