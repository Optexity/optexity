import asyncio
import logging
import time
from typing import Callable

import aiofiles
import requests
from browser_use import Agent, BrowserSession, ChatGoogle, Tools

from optexity.exceptions import AssertLocatorPresenceException
from optexity.inference.agents.error_handler.error_handler import ErrorHandlerAgent
from optexity.inference.agents.index_prediction.action_prediction_locator_axtree import (
    ActionPredictionLocatorAxtree,
)
from optexity.inference.infra.browser import Browser
from optexity.schema.actions.interaction_action import (
    AgenticTask,
    ClickElementAction,
    CloseOverlayPopupAction,
    DownloadUrlAsPdfAction,
    GoBackAction,
    InputTextAction,
    InteractionAction,
    SelectOptionAction,
)
from optexity.schema.memory import Memory, OutputData

error_handler_agent = ErrorHandlerAgent()


logger = logging.getLogger(__name__)


index_prediction_agent = ActionPredictionLocatorAxtree()


async def run_interaction_action(
    interaction_action: InteractionAction,
    memory: Memory,
    browser: Browser,
    retries_left: int,
):
    if retries_left <= 0:
        return

    logger.debug(
        f"---------Running interaction action {interaction_action.model_dump_json()}---------"
    )

    try:
        if interaction_action.click_element:
            if interaction_action.start_2fa_timer:
                memory.automation_state.start_2fa_time = time.time()
            await handle_click_element(
                interaction_action.click_element,
                memory,
                browser,
                interaction_action.max_timeout_seconds_per_try,
                interaction_action.max_tries,
            )
        elif interaction_action.input_text:
            await handle_input_text(
                interaction_action.input_text,
                memory,
                browser,
                interaction_action.max_timeout_seconds_per_try,
                interaction_action.max_tries,
            )
        elif interaction_action.select_option:
            await handle_select_option(
                interaction_action.select_option,
                memory,
                browser,
                interaction_action.max_timeout_seconds_per_try,
                interaction_action.max_tries,
            )
        elif interaction_action.go_back:
            await handle_go_back(interaction_action.go_back, memory, browser)
        elif interaction_action.download_url_as_pdf:
            await handle_download_url_as_pdf(
                interaction_action.download_url_as_pdf, memory, browser
            )
        elif interaction_action.agentic_task:
            await handle_agentic_task(interaction_action.agentic_task, memory, browser)
        elif interaction_action.close_overlay_popup:
            await handle_agentic_task(
                interaction_action.close_overlay_popup, memory, browser
            )
    except AssertLocatorPresenceException as e:
        await handle_assert_locator_presence_error(
            e, interaction_action, memory, browser, retries_left
        )


async def command_based_action_with_retry(
    func: Callable,
    command: str | None,
    max_tries: int,
    max_timeout_seconds_per_try: float,
    assert_locator_presence: bool,
):
    if command is None:
        return
    last_error = None
    for try_index in range(max_tries):
        last_error = None
        try:
            await func()
            logger.debug(f"{func.__name__} successful on try {try_index + 1}")
            return
        except Exception as e:
            last_error = e
            await asyncio.sleep(max_timeout_seconds_per_try)

    logger.debug(f"{func.__name__} failed after {max_tries} tries: {last_error}")

    if last_error and assert_locator_presence:
        logger.debug(
            f"Error in {func.__name__} with assert_locator_presence: {func.__name__}: {last_error}"
        )
        raise AssertLocatorPresenceException(
            message=f"Error in {func.__name__} with assert_locator_presence: {func.__name__}",
            original_error=last_error,
            command=command,
        )


async def prompt_based_action(
    func: Callable, memory: Memory, prompt_instructions: str | None, skip_prompt: bool
):
    if skip_prompt or prompt_instructions is None:
        return
    memory.automation_state.try_index += 1
    axtree = memory.browser_states[-1].axtree

    try:
        final_prompt, response, token_usage = index_prediction_agent.predict_action(
            prompt_instructions, axtree
        )
        memory.token_usage += token_usage
        memory.browser_states[-1].final_prompt = final_prompt
        memory.browser_states[-1].llm_response = response.model_dump()
        await func(response.index)
    except Exception as e:
        logger.error(f"Error in prompt_based_action for {func.__name__}: {e}")
        return


async def handle_click_element(
    click_element_action: ClickElementAction,
    memory: Memory,
    browser: Browser,
    max_timeout_seconds_per_try: float,
    max_tries: int,
):
    if click_element_action.skip_prompt:
        max_tries = 1
        max_timeout_seconds_per_try = 20.0

    async def _click_locator():
        async def _actual_click():
            locator = await browser.get_locator_from_command(
                click_element_action.command
            )
            if click_element_action.double_click:
                await locator.dblclick(timeout=max_timeout_seconds_per_try * 1000)
            else:
                await locator.click(timeout=max_timeout_seconds_per_try * 1000)

        if click_element_action.expect_download:
            page = await browser.get_current_page()
            if page is None:
                logger.error("No page found for current page")
                return
            download_path = (
                memory.downloads_directory / click_element_action.download_filename
            )
            async with page.expect_download() as download_info:
                await _actual_click()
                download = await download_info.value
                if download:
                    await download.save_as(download_path)
                    memory.downloads.append(download_path)
                else:
                    logger.error("No download found")
        else:
            await _actual_click()

    if click_element_action.command:
        last_error = await command_based_action_with_retry(
            _click_locator,
            click_element_action.command,
            max_tries,
            max_timeout_seconds_per_try,
            click_element_action.assert_locator_presence,
        )

        if last_error is None:
            return

    await prompt_based_action(
        browser.click_index,
        memory,
        click_element_action.prompt_instructions,
        click_element_action.skip_prompt,
    )


async def handle_input_text(
    input_text_action: InputTextAction,
    memory: Memory,
    browser: Browser,
    max_timeout_seconds_per_try: float,
    max_tries: int,
):
    if input_text_action.skip_prompt:
        max_tries = 1
        max_timeout_seconds_per_try = 20.0

    async def _input_text_locator():
        locator = await browser.get_locator_from_command(input_text_action.command)
        if input_text_action.fill_or_type == "fill":
            await locator.fill(
                input_text_action.input_text, timeout=max_timeout_seconds_per_try * 1000
            )
        else:
            await locator.type(
                input_text_action.input_text,
                timeout=max_timeout_seconds_per_try * 1000,
            )

    if input_text_action.command:
        last_error = await command_based_action_with_retry(
            _input_text_locator,
            input_text_action.command,
            max_tries,
            max_timeout_seconds_per_try,
            input_text_action.assert_locator_presence,
        )

        if last_error is None:
            return

    await prompt_based_action(
        browser.input_text_index,
        memory,
        input_text_action.prompt_instructions,
        input_text_action.skip_prompt,
    )


async def handle_select_option(
    select_option_action: SelectOptionAction,
    memory: Memory,
    browser: Browser,
    max_timeout_seconds_per_try: float,
    max_tries: int,
):

    async def _select_option_locator():
        async def _actual_select_option():
            locator = await browser.get_locator_from_command(
                select_option_action.command
            )
            await locator.select_option(select_option_action.select_values)

        if select_option_action.expect_download:
            page = await browser.get_current_page()
            if page is None:
                logger.error("No page found for current page")
                return
            download_path = (
                memory.downloads_directory / select_option_action.download_filename
            )
            async with page.expect_download() as download_info:
                await _actual_select_option()
                download = await download_info.value
                if download:
                    await download.save_as(download_path)
                    memory.downloads.append(download_path)
                else:
                    logger.error("No download found")
        else:
            await _actual_select_option()

    if select_option_action.command:
        last_error = await command_based_action_with_retry(
            _select_option_locator,
            select_option_action.command,
            max_tries,
            max_timeout_seconds_per_try,
            select_option_action.assert_locator_presence,
        )

        if last_error is None:
            return


async def handle_go_back(
    go_back_action: GoBackAction, memory: Memory, browser: Browser
):
    page = await browser.get_current_page()
    if page is None:
        return
    await page.go_back()


async def handle_download_url_as_pdf(
    download_url_as_pdf_action: DownloadUrlAsPdfAction, memory: Memory, browser: Browser
):
    pdf_url = await browser.get_current_page_url()

    if pdf_url is None:
        logger.error("No PDF URL found for current page")
        return

    r = requests.get(pdf_url)
    if r.status_code != 200:
        logger.error(f"Failed to download PDF from {pdf_url}: {r.status_code}")
        return

    download_path = (
        memory.downloads_directory / download_url_as_pdf_action.download_filename
    )

    if isinstance(r.content, bytes):
        async with aiofiles.open(download_path, "wb") as f:
            await f.write(r.content)
    elif isinstance(r.content, str):
        async with aiofiles.open(download_path, "w") as f:
            await f.write(r.content)
    else:
        logger.error(f"Unsupported content type: {type(r.content)}")
        return

    memory.downloads.append(download_path)


async def handle_agentic_task(
    agentic_task_action: AgenticTask | CloseOverlayPopupAction,
    memory: Memory,
    browser: Browser,
):

    if agentic_task_action.backend == "browser_use":

        tools = Tools(
            exclude_actions=[
                "search",
                "navigate",
                "go_back",
                "upload_file",
                "scroll",
                "find_text",
                "send_keys",
                "evaluate",
                "switch",
                "close",
                "extract",
                "dropdown_options",
                "select_dropdown",
                "write_file",
                "read_file",
                "replace_file",
            ]
        )
        llm = ChatGoogle(model="gemini-flash-latest")
        browser_session = BrowserSession(
            cdp_url=browser.cdp_url, keep_alive=agentic_task_action.keep_alive
        )

        step_directory = (
            memory.logs_directory / f"step_{str(memory.automation_state.step_index)}"
        )
        step_directory.mkdir(parents=True, exist_ok=True)

        agent = Agent(
            task=agentic_task_action.task,
            llm=llm,
            browser_session=browser_session,
            use_vision=agentic_task_action.use_vision,
            tools=tools,
            calculate_cost=True,
            save_conversation_path=step_directory,
        )
        logger.debug(f"Starting browser session for agentic task {browser.cdp_url} ")
        await agent.browser_session.start()
        logger.debug(f"Finally running agentic task on browser_use {browser.cdp_url} ")
        await agent.run(max_steps=agentic_task_action.max_steps)
        logger.debug(f"Agentic task completed on browser_use {browser.cdp_url} ")
        agent.stop()
        if agent.browser_session:
            await agent.browser_session.stop()
            await agent.browser_session.reset()

    elif agentic_task_action.backend == "browserbase":
        raise NotImplementedError("Browserbase is not supported yet")


async def handle_assert_locator_presence_error(
    error: AssertLocatorPresenceException,
    interaction_action: InteractionAction,
    memory: Memory,
    browser: Browser,
    retries_left: int,
):
    logger.debug(f"Handling assert locator presence error: {error.command}")
    if retries_left > 1:
        final_prompt, response, token_usage = error_handler_agent.classify_error(
            error.command, memory.browser_states[-1].screenshot
        )
        memory.token_usage += token_usage

        if response.error_type == "website_not_loaded":
            asyncio.sleep(5)
            await run_interaction_action(
                interaction_action, memory, browser, retries_left - 1
            )
        elif response.error_type == "overlay_popup_blocking":
            close_overlay_popup_action = CloseOverlayPopupAction()
            await handle_agentic_task(close_overlay_popup_action, memory, browser)
            await run_interaction_action(
                interaction_action, memory, browser, retries_left - 1
            )
        elif response.error_type == "fatal_error":
            logger.error(
                f"Fatal error running node {memory.automation_state.step_index} after {retries_left} retries: {error.original_error}. Error: {response.detailed_reason}"
            )
            memory.variables.output_data.append(
                OutputData(text=response.detailed_reason)
            )
            raise error
        return
    else:
        logger.error(
            f"Error running node {memory.automation_state.step_index} after {retries_left} retries: {error.original_error}"
        )
        raise error
