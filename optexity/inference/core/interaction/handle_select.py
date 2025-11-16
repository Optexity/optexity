import logging

from optexity.inference.core.interaction.utils import (
    command_based_action_with_retry,
    get_index_from_prompt,
    handle_download,
)
from optexity.inference.infra.browser import Browser
from optexity.schema.actions.interaction_action import SelectOptionAction
from optexity.schema.memory import Memory
from optexity.schema.task import Task

logger = logging.getLogger(__name__)


async def handle_select_option(
    select_option_action: SelectOptionAction,
    task: Task,
    memory: Memory,
    browser: Browser,
    max_timeout_seconds_per_try: float,
    max_tries: int,
):

    if select_option_action.command:
        last_error = await command_based_action_with_retry(
            lambda: select_option_locator(
                select_option_action, browser, max_timeout_seconds_per_try, memory, task
            ),
            select_option_action.command,
            max_tries,
            max_timeout_seconds_per_try,
            select_option_action.assert_locator_presence,
        )

        if last_error is None:
            return

    if not select_option_action.skip_prompt:
        await select_option_index(select_option_action, browser, memory, task)


async def select_option_locator(
    select_option_action: SelectOptionAction,
    browser: Browser,
    max_timeout_seconds_per_try: float,
    memory: Memory,
    task: Task,
):
    async def _actual_select_option():
        locator = await browser.get_locator_from_command(select_option_action.command)
        await locator.select_option(
            select_option_action.select_values,
            no_wait_after=True,
            timeout=max_timeout_seconds_per_try * 1000,
        )

    if select_option_action.expect_download:
        await handle_download(
            _actual_select_option,
            memory,
            browser,
            task,
            select_option_action.download_filename,
        )
    else:
        await _actual_select_option()


async def select_option_index(
    select_option_action: SelectOptionAction,
    browser: Browser,
    memory: Memory,
    task: Task,
):

    try:
        index = await get_index_from_prompt(
            memory, select_option_action.prompt_instructions, browser
        )
        if index is None:
            return

        async def _actual_select_option():
            action_model = browser.backend_agent.ActionModel(
                **{
                    "select_dropdown": {
                        "index": int(index),
                        "text": select_option_action.select_values[0],
                    }
                }
            )
            await browser.backend_agent.multi_act([action_model])

        if select_option_action.expect_download:
            await handle_download(
                _actual_select_option,
                memory,
                browser,
                task,
                select_option_action.download_filename,
            )
        else:
            await _actual_select_option()
    except Exception as e:
        logger.error(f"Error in select_option_index: {e}")
        return
