import asyncio
import logging

from optexity.inference.infra.browser import Browser
from optexity.schema.actions.misc_action import FailStateAction, SleepAction
from optexity.schema.memory import Memory
from optexity.schema.task import Task

logger = logging.getLogger(__name__)


async def run_sleep_action(sleep_action: SleepAction):
    logger.debug(
        f"---------Running sleep action {sleep_action.model_dump_json()}---------"
    )
    await asyncio.sleep(sleep_action.sleep_time)


async def run_fail_state_action(
    fail_state_action: FailStateAction, memory: Memory, browser: Browser, task: Task
):
    logger.debug(
        f"---------Running fail state action {fail_state_action.model_dump_json()}---------"
    )
    raise Exception(fail_state_action.failure_message)
