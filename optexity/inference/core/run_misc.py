import asyncio
import logging

from optexity.schema.actions.misc_action import SleepAction

logger = logging.getLogger(__name__)


async def run_sleep_action(sleep_action: SleepAction):
    logger.debug(
        f"---------Running sleep action {sleep_action.model_dump_json()}---------"
    )
    await asyncio.sleep(sleep_action.sleep_time)
