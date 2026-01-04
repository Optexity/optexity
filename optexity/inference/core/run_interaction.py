import asyncio
import logging
from datetime import datetime, timezone

import aiofiles

from optexity.exceptions import AssertLocatorPresenceException
from optexity.inference.agents.error_handler.error_handler import ErrorHandlerAgent
from optexity.inference.core.interaction.handle_agentic_task import handle_agentic_task
from optexity.inference.core.interaction.handle_check import (
    handle_check_element,
    handle_uncheck_element,
)
from optexity.inference.core.interaction.handle_click import handle_click_element
from optexity.inference.core.interaction.handle_input import handle_input_text
from optexity.inference.core.interaction.handle_keypress import handle_key_press
from optexity.inference.core.interaction.handle_select import handle_select_option
from optexity.inference.core.interaction.handle_upload import handle_upload_file
from optexity.inference.infra.browser import Browser
from optexity.schema.actions.interaction_action import (
    CloseOverlayPopupAction,
    CloseTabsUntil,
    DownloadUrlAsPdfAction,
    GoBackAction,
    GoToUrlAction,
    InteractionAction,
)
from optexity.schema.memory import BrowserState, Memory, OutputData
from optexity.schema.task import Task

error_handler_agent = ErrorHandlerAgent()


logger = logging.getLogger(__name__)


async def run_interaction_action(
    interaction_action: InteractionAction,
    task: Task,
    memory: Memory,
    browser: Browser,
    retries_left: int,
):
    if retries_left <= 0:
        return

    logger.debug(
        f"---------Running interaction action {interaction_action.model_dump_json(exclude_none=True)}---------"
    )

    try:
        memory.automation_state.start_2fa_time = datetime.now(timezone.utc)
        if interaction_action.click_element:
            await handle_click_element(
                interaction_action.click_element,
                task,
                memory,
                browser,
                interaction_action.max_timeout_seconds_per_try,
                interaction_action.max_tries,
            )
        elif interaction_action.input_text:
            await handle_input_text(
                interaction_action.input_text,
                task,
                memory,
                browser,
                interaction_action.max_timeout_seconds_per_try,
                interaction_action.max_tries,
            )
        elif interaction_action.select_option:
            await handle_select_option(
                interaction_action.select_option,
                task,
                memory,
                browser,
                interaction_action.max_timeout_seconds_per_try,
                interaction_action.max_tries,
            )
        elif interaction_action.check:
            await handle_check_element(
                interaction_action.check,
                task,
                memory,
                browser,
                interaction_action.max_timeout_seconds_per_try,
                interaction_action.max_tries,
            )
        elif interaction_action.uncheck:
            await handle_uncheck_element(
                interaction_action.uncheck,
                task,
                memory,
                browser,
                interaction_action.max_timeout_seconds_per_try,
                interaction_action.max_tries,
            )
        elif interaction_action.go_back:
            await handle_go_back(interaction_action.go_back, memory, browser)
        elif interaction_action.download_url_as_pdf:
            await handle_download_url_as_pdf(
                interaction_action.download_url_as_pdf, task, memory, browser
            )
        elif interaction_action.agentic_task:
            await handle_agentic_task(
                interaction_action.agentic_task, task, memory, browser
            )
        elif interaction_action.close_overlay_popup:
            await handle_agentic_task(
                interaction_action.close_overlay_popup, task, memory, browser
            )
        elif interaction_action.go_to_url:
            await handle_go_to_url(interaction_action.go_to_url, task, memory, browser)
        elif interaction_action.upload_file:
            await handle_upload_file(
                interaction_action.upload_file,
                task,
                memory,
                browser,
                interaction_action.max_timeout_seconds_per_try,
                interaction_action.max_tries,
            )
        elif interaction_action.close_current_tab:
            await browser.close_current_tab()
        elif interaction_action.switch_tab:
            await browser.switch_tab(interaction_action.switch_tab.tab_index)
        elif interaction_action.close_tabs_until:
            await handle_close_tabs_until(
                interaction_action.close_tabs_until, task, memory, browser
            )
        elif interaction_action.key_press:
            await handle_key_press(interaction_action.key_press, memory, browser)
    except AssertLocatorPresenceException as e:
        await handle_assert_locator_presence_error(
            e, interaction_action, task, memory, browser, retries_left
        )


async def handle_close_tabs_until(
    close_tabs_until_action: CloseTabsUntil,
    task: Task,
    memory: Memory,
    browser: Browser,
):

    while True:
        page = await browser.get_current_page()
        if page is None:
            return

        if close_tabs_until_action.matching_url is not None:
            if close_tabs_until_action.matching_url in page.url:
                break
        elif close_tabs_until_action.tab_index is not None:
            if len(browser.context.pages) == close_tabs_until_action.tab_index + 1:
                break

        await browser.close_current_tab()


async def handle_go_to_url(
    go_to_url_action: GoToUrlAction, task: Task, memory: Memory, browser: Browser
):
    await browser.go_to_url(go_to_url_action.url)


async def handle_go_back(
    go_back_action: GoBackAction, memory: Memory, browser: Browser
):
    page = await browser.get_current_page()
    if page is None:
        return
    await page.go_back()


async def handle_download_url_as_pdf(
    download_url_as_pdf_action: DownloadUrlAsPdfAction,
    task: Task,
    memory: Memory,
    browser: Browser,
):
    if download_url_as_pdf_action.url is not None:
        pdf_url = download_url_as_pdf_action.url
    else:
        pdf_url = await browser.get_current_page_url()

    if pdf_url is None:
        logger.error("No PDF URL found for current page")
        return
    download_path = (
        task.downloads_directory / download_url_as_pdf_action.download_filename
    )

    resp = await browser.context.request.get(pdf_url)

    if not resp.ok:
        logger.error(f"Failed to download PDF: {resp.status}")
        return

    content = await resp.body()
    async with aiofiles.open(download_path, "wb") as f:
        await f.write(content)

    memory.downloads.append(download_path)


async def handle_assert_locator_presence_error(
    error: AssertLocatorPresenceException,
    interaction_action: InteractionAction,
    task: Task,
    memory: Memory,
    browser: Browser,
    retries_left: int,
):
    logger.debug(f"Handling assert locator presence error: {error.command}")
    if retries_left > 1:
        browser_state_summary = await browser.get_browser_state_summary()
        memory.browser_states[-1] = BrowserState(
            url=browser_state_summary.url,
            screenshot=browser_state_summary.screenshot,
            title=browser_state_summary.title,
            axtree=browser_state_summary.dom_state.llm_representation(),
        )
        final_prompt, response, token_usage = error_handler_agent.classify_error(
            error.command, memory.browser_states[-1].screenshot
        )
        memory.token_usage += token_usage

        if response.error_type == "website_not_loaded":
            await asyncio.sleep(5)
            await run_interaction_action(
                interaction_action, task, memory, browser, retries_left - 1
            )
        elif response.error_type == "overlay_popup_blocking":
            close_overlay_popup_action = CloseOverlayPopupAction()
            await handle_agentic_task(close_overlay_popup_action, task, memory, browser)
            await run_interaction_action(
                interaction_action, task, memory, browser, retries_left - 1
            )
        elif response.error_type == "fatal_error":
            logger.error(
                f"Fatal error running node {memory.automation_state.step_index} after {retries_left} retries: {error.original_error}. Error: {response.detailed_reason}"
            )
            memory.variables.output_data.append(
                OutputData(unique_identifier="error", text=response.detailed_reason)
            )
            raise Exception(
                f"Fatal error running node {memory.automation_state.step_index} after {retries_left} retries: {error.original_error}. Final reason: {response.detailed_reason}"
            )
    else:
        logger.error(
            f"Error running node {memory.automation_state.step_index} after {retries_left} retries: {error.original_error}"
        )
        raise error
