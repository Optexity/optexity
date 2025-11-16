import logging

from optexity.inference.core.interaction.utils import (
    command_based_action_with_retry,
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

    if click_element_action.command:
        last_error = await command_based_action_with_retry(
            lambda: click_locator(
                click_element_action, browser, max_timeout_seconds_per_try, memory, task
            ),
            click_element_action.command,
            max_tries,
            max_timeout_seconds_per_try,
            click_element_action.assert_locator_presence,
        )

        if last_error is None:
            return

    if not click_element_action.skip_prompt:
        await click_element_index(click_element_action, browser, memory, task)


async def click_locator(
    click_element_action: ClickElementAction,
    browser: Browser,
    max_timeout_seconds_per_try: float,
    memory: Memory,
    task: Task,
):
    async def _actual_click():
        locator = await browser.get_locator_from_command(click_element_action.command)
        if click_element_action.double_click:
            await locator.dblclick(
                no_wait_after=True, timeout=max_timeout_seconds_per_try * 1000
            )
        else:
            await locator.click(
                no_wait_after=True, timeout=max_timeout_seconds_per_try * 1000
            )

    if click_element_action.expect_download:
        await handle_download(
            _actual_click, memory, browser, task, click_element_action.download_filename
        )
    else:
        await _actual_click()


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
