import logging

from optexity.inference.core.interaction.handle_command import (
    command_based_action_with_retry,
)
from optexity.inference.infra.browser import Browser
from optexity.schema.actions.interaction_action import CheckAction, UncheckAction
from optexity.schema.memory import Memory
from optexity.schema.task import Task

logger = logging.getLogger(__name__)


async def handle_check_element(
    check_element_action: CheckAction,
    task: Task,
    memory: Memory,
    browser: Browser,
    max_timeout_seconds_per_try: float,
    max_tries: int,
):

    if check_element_action.command and not check_element_action.skip_command:
        last_error = await command_based_action_with_retry(
            check_element_action,
            browser,
            memory,
            task,
            max_tries,
            max_timeout_seconds_per_try,
        )

        if last_error is None:
            return


async def handle_uncheck_element(
    uncheck_element_action: UncheckAction,
    task: Task,
    memory: Memory,
    browser: Browser,
    max_timeout_seconds_per_try: float,
    max_tries: int,
):

    if uncheck_element_action.command and not uncheck_element_action.skip_command:
        last_error = await command_based_action_with_retry(
            uncheck_element_action,
            browser,
            memory,
            task,
            max_tries,
            max_timeout_seconds_per_try,
        )

        if last_error is None:
            return
