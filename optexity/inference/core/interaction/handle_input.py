import asyncio
import logging
import re

import pyautogui
import pyperclip

from optexity.exceptions import ElementNotFoundInAxtreeException
from optexity.inference.core.interaction.handle_command import (
    command_based_action_with_retry,
)
from optexity.inference.core.interaction.utils import (
    get_coordinates_from_prompt,
    get_index_from_prompt,
)
from optexity.inference.infra.browser import Browser
from optexity.schema.actions.interaction_action import InputTextAction
from optexity.schema.memory import Memory
from optexity.schema.task import Task

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05
logger = logging.getLogger(__name__)


async def handle_input_text(
    input_text_action: InputTextAction,
    task: Task,
    memory: Memory,
    browser: Browser,
    max_timeout_seconds_per_try: float,
    max_tries: int,
):

    if input_text_action.input_text is None:
        return

    # {some english chars [0]}
    INT_INDEX_PATTERN = re.compile(r"^\{([A-Za-z_][A-Za-z0-9_]*)\[(\d+)\]\}$")

    if INT_INDEX_PATTERN.match(input_text_action.input_text) is not None:
        logger.debug(
            "Skipping input text because input variable was not present for this step"
        )
        return

    if browser.channel == "rdp" or browser.backend == "computer-vision":
        await input_text_coordinates(input_text_action, browser, memory, task)
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
        await input_text_index(input_text_action, browser, memory, task)


async def input_text_index(
    input_text_action: InputTextAction, browser: Browser, memory: Memory, task: Task
):
    try:
        index = await get_index_from_prompt(
            memory,
            input_text_action.prompt_instructions,
            browser,
            task,
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
    except ElementNotFoundInAxtreeException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in input_text_index: {e}")
        return


def detect_platform() -> str:
    import sys

    return "macos" if sys.platform == "darwin" else "linux"


def modifier_key() -> str:
    return "command" if detect_platform() == "macos" else "ctrl"


async def input_text_coordinates(
    input_text_action: InputTextAction,
    browser: Browser,
    memory: Memory,
    task: Task,
):

    if input_text_action.input_text is None:
        return

    def _paste(text: str):
        _mod = modifier_key()
        pyperclip.copy(text)
        pyautogui.hotkey(_mod, "v")

    try:
        data = await get_coordinates_from_prompt(
            memory, input_text_action.prompt_instructions, browser, task
        )

        if data is None:
            logger.error("No coordinates found")
            return

        x = data[0]
        y = data[1]

        logger.debug(f"Typing text at coordinates: {x}, {y}")

        pyautogui.click(x, y)
        await asyncio.sleep(0.2)
        _paste(input_text_action.input_text)

        if input_text_action.press_enter:
            await asyncio.sleep(0.2)
            pyautogui.press("enter")

    except ElementNotFoundInAxtreeException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in input_text_coordinates: {e}")
        return
