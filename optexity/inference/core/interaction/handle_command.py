import asyncio
import logging

from playwright.async_api import Locator

from optexity.exceptions import AssertLocatorPresenceException
from optexity.inference.core.interaction.handle_select_locator import (
    select_option_locator,
)
from optexity.inference.core.interaction.utils import handle_download
from optexity.inference.infra.browser import Browser
from optexity.schema.actions.interaction_action import (
    CheckAction,
    ClickElementAction,
    InputTextAction,
    SelectOptionAction,
    UncheckAction,
    UploadFileAction,
)
from optexity.schema.memory import BrowserState, Memory
from optexity.schema.task import Task

logger = logging.getLogger(__name__)


async def command_based_action_with_retry(
    action: (
        ClickElementAction
        | InputTextAction
        | SelectOptionAction
        | CheckAction
        | UploadFileAction
        | UncheckAction
    ),
    browser: Browser,
    memory: Memory,
    task: Task,
    max_tries: int,
    max_timeout_seconds_per_try: float,
):

    if action.command is None or action.skip_command:
        return

    last_error = None

    logger.debug(f"Executing command-based action: {action.__class__.__name__}")

    for try_index in range(max_tries):
        last_error = None
        try:
            # https://playwright.dev/docs/actionability
            locator = await browser.get_locator_from_command(action.command)
            if try_index == 0:
                try:
                    await locator.wait_for(
                        state="visible", timeout=max_timeout_seconds_per_try * 1000
                    )
                except Exception as e:
                    pass
            is_visible = await locator.is_visible()

            if is_visible:
                browser_state_summary = await browser.get_browser_state_summary()
                memory.browser_states[-1] = BrowserState(
                    url=browser_state_summary.url,
                    screenshot=browser_state_summary.screenshot,
                    title=browser_state_summary.title,
                    axtree=browser_state_summary.dom_state.llm_representation(),
                )

                if isinstance(action, ClickElementAction):
                    await click_locator(
                        action,
                        locator,
                        browser,
                        memory,
                        task,
                        max_timeout_seconds_per_try,
                    )
                elif isinstance(action, InputTextAction):
                    await input_text_locator(
                        action, locator, max_timeout_seconds_per_try
                    )
                elif isinstance(action, SelectOptionAction):
                    await select_option_locator(
                        action,
                        locator,
                        browser,
                        memory,
                        task,
                        max_timeout_seconds_per_try,
                    )
                elif isinstance(action, CheckAction):
                    await check_locator(
                        action, locator, max_timeout_seconds_per_try, browser
                    )
                elif isinstance(action, UncheckAction):
                    await uncheck_locator(
                        action, locator, max_timeout_seconds_per_try, browser
                    )
                elif isinstance(action, UploadFileAction):
                    await upload_file_locator(action, locator)
                logger.debug(
                    f"{action.__class__.__name__} successful on try {try_index + 1}"
                )
                return
            else:
                await asyncio.sleep(max_timeout_seconds_per_try)
        except Exception as e:
            last_error = e
            await asyncio.sleep(max_timeout_seconds_per_try)

    logger.debug(
        f"{action.__class__.__name__} failed after {max_tries} tries: {last_error}"
    )

    if last_error and action.assert_locator_presence:
        logger.debug(
            f"Error in {action.__class__.__name__} with assert_locator_presence: {action.__class__.__name__}: {last_error}"
        )
        raise AssertLocatorPresenceException(
            message=f"Error in {action.__class__.__name__} with assert_locator_presence: {action.__class__.__name__}",
            original_error=last_error,
            command=action.command,
        )
    return last_error


async def click_locator(
    click_element_action: ClickElementAction,
    locator: Locator,
    browser: Browser,
    memory: Memory,
    task: Task,
    max_timeout_seconds_per_try: float,
):
    async def _actual_click():

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


async def input_text_locator(
    input_text_action: InputTextAction,
    locator: Locator,
    max_timeout_seconds_per_try: float,
):

    if input_text_action.fill_or_type == "fill":
        await locator.fill(
            input_text_action.input_text,
            no_wait_after=True,
            timeout=max_timeout_seconds_per_try * 1000,
        )
    else:
        await locator.type(
            input_text_action.input_text,
            no_wait_after=True,
            timeout=max_timeout_seconds_per_try * 1000,
        )

    if input_text_action.press_enter:
        await locator.press("Enter")


async def check_locator(
    locator: Locator,
    max_timeout_seconds_per_try: float,
    browser: Browser,
    action: CheckAction,
):
    await locator.uncheck(
        no_wait_after=True, timeout=max_timeout_seconds_per_try * 1000
    )
    await asyncio.sleep(1)
    locator = await browser.get_locator_from_command(action.command)
    await locator.check(no_wait_after=True, timeout=max_timeout_seconds_per_try * 1000)


async def uncheck_locator(
    locator: Locator,
    max_timeout_seconds_per_try: float,
    browser: Browser,
    action: UncheckAction,
):
    await locator.check(no_wait_after=True, timeout=max_timeout_seconds_per_try * 1000)
    await asyncio.sleep(1)
    locator = await browser.get_locator_from_command(action.command)
    await locator.uncheck(
        no_wait_after=True, timeout=max_timeout_seconds_per_try * 1000
    )


async def upload_file_locator(upload_file_action: UploadFileAction, locator: Locator):
    await locator.set_input_files(upload_file_action.file_path)
