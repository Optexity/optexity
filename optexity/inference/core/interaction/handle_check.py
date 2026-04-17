import logging

from optexity.exceptions import ElementNotFoundInAxtreeException
from optexity.inference.core.interaction.handle_command import (
    command_based_action_with_retry,
)
from optexity.inference.core.interaction.utils import (
    get_index_from_prompt,
    update_screenshot_with_highlight,
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

    if not check_element_action.skip_prompt:
        logger.debug(
            f"Executing prompt-based action: {check_element_action.__class__.__name__}"
        )
        await check_element_index(check_element_action, browser, memory, task)


async def check_element_index(
    check_action: CheckAction,
    browser: Browser,
    memory: Memory,
    task: Task,
):
    try:
        index = await get_index_from_prompt(
            memory, check_action.prompt_instructions, browser, task
        )
        if index is None:
            return

        try:
            await update_screenshot_with_highlight(browser, memory, index)
        except Exception as e:
            logger.error(
                f"Error in updating screenshot with highlight in check_element_index: {e}"
            )

        logger.debug(f"Checking element with index: {index}")
        action_model = browser.backend_agent.ActionModel(
            **{"click": {"index": int(index), "button": "left"}}
        )
        await browser.backend_agent.multi_act([action_model])
    except ElementNotFoundInAxtreeException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in check_element_index: {e}")
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

    if not uncheck_element_action.skip_prompt:
        logger.debug(
            f"Executing prompt-based action: {uncheck_element_action.__class__.__name__}"
        )
        await uncheck_element_index(uncheck_element_action, browser, memory, task)


async def uncheck_element_index(
    uncheck_action: UncheckAction,
    browser: Browser,
    memory: Memory,
    task: Task,
):
    try:
        index = await get_index_from_prompt(
            memory, uncheck_action.prompt_instructions, browser, task
        )
        if index is None:
            return

        try:
            await update_screenshot_with_highlight(browser, memory, index)
        except Exception as e:
            logger.error(
                f"Error in updating screenshot with highlight in uncheck_element_index: {e}"
            )

        logger.debug(f"Unchecking element with index: {index}")
        action_model = browser.backend_agent.ActionModel(
            **{"click": {"index": int(index), "button": "left"}}
        )
        await browser.backend_agent.multi_act([action_model])
    except ElementNotFoundInAxtreeException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in uncheck_element_index: {e}")
        return
