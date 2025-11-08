import asyncio
import logging
import time
from typing import Callable

from optexity.inference.infra.browser import Browser
from optexity.schema.actions.interaction_action import (
    ClickElementAction,
    GoBackAction,
    InputTextAction,
    InteractionAction,
    SelectOptionAction,
)
from optexity.schema.memory import Memory

logger = logging.getLogger(__name__)


def get_index_from_prompt(prompt: str, axtree: str) -> int | None:
    raise NotImplementedError("Not implemented")


async def run_interaction_action(
    interaction_action: InteractionAction, memory: Memory, browser: Browser
):
    logger.debug(
        f"---------Running interaction action {interaction_action.model_dump_json()}---------"
    )

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
            asyncio.sleep(max_timeout_seconds_per_try)

    logger.debug(f"{func.__name__} failed after {max_tries} tries: {last_error}")

    if last_error and assert_locator_presence:
        logger.debug(
            f"Error in {func.__name__} with assert_locator_presence: {func.__name__}: {last_error}"
        )
        raise last_error


async def prompt_based_action(
    func: Callable, memory: Memory, prompt_instructions: str | None, skip_prompt: bool
):
    if skip_prompt or prompt_instructions is None:
        return
    memory.automation_state.try_index += 1
    axtree = memory.browser_states[-1].axtree
    try:
        index = get_index_from_prompt(prompt_instructions, axtree)
        await func(index)
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
    async def _click_locator():
        locator = await browser.get_locator_from_command(click_element_action.command)
        if click_element_action.double_click:
            await locator.dblclick(timeout=max_timeout_seconds_per_try * 1000)
        else:
            await locator.click(timeout=max_timeout_seconds_per_try * 1000)

    await command_based_action_with_retry(
        _click_locator,
        click_element_action.command,
        max_tries,
        max_timeout_seconds_per_try,
        click_element_action.assert_locator_presence,
    )

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

    await command_based_action_with_retry(
        _input_text_locator,
        input_text_action.command,
        max_tries,
        max_timeout_seconds_per_try,
        input_text_action.assert_locator_presence,
    )

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
        locator = await browser.get_locator_from_command(select_option_action.command)
        await locator.select_option(select_option_action.select_values)

    await command_based_action_with_retry(
        _select_option_locator,
        select_option_action.command,
        max_tries,
        max_timeout_seconds_per_try,
        select_option_action.assert_locator_presence,
    )


async def handle_go_back(
    go_back_action: GoBackAction, memory: Memory, browser: Browser
):
    page = await browser.get_current_page()
    if page is None:
        return
    await page.go_back()
