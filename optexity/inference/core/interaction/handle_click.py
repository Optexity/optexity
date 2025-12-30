import logging

from optexity.inference.core.interaction.handle_command import (
    command_based_action_with_retry,
)
from optexity.inference.core.interaction.utils import (
    get_index_from_prompt,
    handle_download,
)
from optexity.inference.infra.browser import Browser
from optexity.schema.actions.interaction_action import ClickElementAction
from optexity.schema.memory import Memory
from optexity.schema.task import Task

logger = logging.getLogger(__name__)


async def handle_click_element(
    click_element_action: ClickElementAction,
    task: Task,
    memory: Memory,
    browser: Browser,
    max_timeout_seconds_per_try: float,
    max_tries: int,
):

    if click_element_action.command and not click_element_action.skip_command:
        last_error = await command_based_action_with_retry(
            click_element_action,
            browser,
            memory,
            task,
            max_tries,
            max_timeout_seconds_per_try,
        )

        if last_error is None:
            return

    if not click_element_action.skip_prompt:
        logger.debug(
            f"Executing prompt-based action: {click_element_action.__class__.__name__}"
        )
        await click_element_index(click_element_action, browser, memory, task)


async def click_element_index(
    click_element_action: ClickElementAction,
    browser: Browser,
    memory: Memory,
    task: Task,
):

    try:
        index = await get_index_from_prompt(
            memory, click_element_action.prompt_instructions, browser
        )
        if index is None:
            return

        async def _actual_click_element():
            action_model = browser.backend_agent.ActionModel(
                **{"click": {"index": index}}
            )
            await browser.backend_agent.multi_act([action_model])

        if click_element_action.expect_download:
            await handle_download(
                _actual_click_element,
                memory,
                browser,
                task,
                click_element_action.download_filename,
            )
        else:
            await _actual_click_element()
    except Exception as e:
        logger.error(f"Error in click_element_index: {e}")
        return
