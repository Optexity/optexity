import asyncio
import logging

from optexity.inference.infra.browser import Browser
from optexity.schema.actions.misc_action import (
    FailStateAction,
    SetVariableAction,
    SleepAction,
)
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


async def run_set_variable_action(
    set_variable_action: SetVariableAction, memory: Memory
):
    logger.debug(
        f"---------Running set variable action {set_variable_action.model_dump_json()}---------"
    )

    if set_variable_action.expression is not None:
        try:
            result = eval(set_variable_action.expression)  # noqa: S307
        except Exception as e:
            logger.error(
                f"Failed to eval expression '{set_variable_action.expression}': {e}"
            )
            raise
    else:
        result = set_variable_action.value

    memory.variables.generated_variables[set_variable_action.name] = [result]
    logger.info(f"Set variable '{set_variable_action.name}' = {result}")
