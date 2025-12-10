import logging

from optexity.inference.core.interaction.utils import command_based_action_with_retry
from optexity.inference.infra.browser import Browser
from optexity.schema.actions.interaction_action import CheckAction
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

    if check_element_action.command:
        last_error = await command_based_action_with_retry(
            lambda: check_locator(
                check_element_action, browser, max_timeout_seconds_per_try, memory, task
            ),
            check_element_action.command,
            max_tries,
            max_timeout_seconds_per_try,
            check_element_action.assert_locator_presence,
        )

        if last_error is None:
            return


async def check_locator(
    check_element_action: CheckAction,
    browser: Browser,
    max_timeout_seconds_per_try: float,
    memory: Memory,
    task: Task,
):
    locator = await browser.get_locator_from_command(check_element_action.command)
    await locator.check(no_wait_after=True, timeout=max_timeout_seconds_per_try * 1000)
