import logging
import re

from optexity.inference.core.interaction.handle_command import (
    command_based_action_with_retry,
)
from optexity.inference.core.interaction.utils import get_index_from_prompt
from optexity.inference.infra.browser import Browser
from optexity.schema.actions.interaction_action import InputTextAction
from optexity.schema.memory import Memory
from optexity.schema.task import Task

logger = logging.getLogger(__name__)


async def handle_input_text(
    input_text_action: InputTextAction,
    task: Task,
    memory: Memory,
    browser: Browser,
    max_timeout_seconds_per_try: float,
    max_tries: int,
):

    # {some english chars [0]}
    INT_INDEX_PATTERN = re.compile(r"^\{([A-Za-z_][A-Za-z0-9_]*)\[(\d+)\]\}$")

    if INT_INDEX_PATTERN.match(input_text_action.input_text) is not None:
        logger.debug(
            "Skipping input text because input variable was not present for this step"
        )
        return

    if input_text_action.command and not input_text_action.skip_command:
        last_error = await command_based_action_with_retry(
            input_text_action,
            browser,
            memory,
            task,
            max_tries,
            max_timeout_seconds_per_try,
        )

        if last_error is None:
            return

    if not input_text_action.skip_prompt:
        logger.debug(
            f"Executing prompt-based action: {input_text_action.__class__.__name__}"
        )
        await input_text_index(input_text_action, browser, memory)


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
