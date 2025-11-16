import logging

from optexity.inference.core.interaction.utils import (
    command_based_action_with_retry,
    get_index_from_prompt,
)
from optexity.inference.infra.browser import Browser
from optexity.schema.actions.interaction_action import InputTextAction
from optexity.schema.memory import Memory

logger = logging.getLogger(__name__)


async def handle_input_text(
    input_text_action: InputTextAction,
    memory: Memory,
    browser: Browser,
    max_timeout_seconds_per_try: float,
    max_tries: int,
):

    if input_text_action.command:
        last_error = await command_based_action_with_retry(
            lambda: input_text_locator(
                input_text_action, browser, max_timeout_seconds_per_try
            ),
            input_text_action.command,
            max_tries,
            max_timeout_seconds_per_try,
            input_text_action.assert_locator_presence,
        )

        if last_error is None:
            return

    if not input_text_action.skip_prompt:
        await input_text_index(input_text_action, browser)


async def input_text_locator(
    input_text_action: InputTextAction,
    browser: Browser,
    max_timeout_seconds_per_try: float,
):

    locator = await browser.get_locator_from_command(input_text_action.command)
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


async def input_text_index(
    input_text_action: InputTextAction, browser: Browser, memory: Memory
):
    try:
        index = await get_index_from_prompt(
            memory, input_text_action.prompt_instructions, browser
        )
        if index is None:
            return

        action_model = browser.backend_agent.ActionModel(
            **{
                "input": {
                    "index": int(index),
                    "text": input_text_action.input_text,
                    "clear": True,
                }
            }
        )
        await browser.backend_agent.multi_act([action_model])
    except Exception as e:
        logger.error(f"Error in input_text_index: {e}")
        return
