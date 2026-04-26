import asyncio
import logging
import re

import pyautogui
import pyperclip

from optexity.exceptions import (
    ElementNotFoundInAxtreeException,
    KeywordNotFoundOnScreenException,
)
from optexity.inference.core.interaction.handle_command import (
    command_based_action_with_retry,
)
from optexity.inference.core.interaction.screenshot_comparison import (
    validate_recording_action,
)
from optexity.inference.core.interaction.utils import (
    get_coordinates_from_prompt,
    get_index_from_prompt,
    resolve_bounding_box_variables,
    resolve_keyword_with_llm_fallback,
    update_screenshot_with_highlight,
)

# from optexity.inference.core.vision.time import (
#     wait_for_screen_to_change,
#     wait_for_stable_screen,
# )
from optexity.inference.core.vision.utils import mark_screenshot
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
        await input_text_coordinates(
            input_text_action,
            browser,
            memory,
            task,
            max_tries,
            max_timeout_seconds_per_try,
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
        try:
            await update_screenshot_with_highlight(browser, memory, index)
        except Exception as e:
            logger.error(
                f"Error in updating screenshot with highlight in input_text_index: {e}"
            )

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


async def input_text_coordinates(
    input_text_action: InputTextAction,
    browser: Browser,
    memory: Memory,
    task: Task,
    max_tries: int = 3,
    max_timeout_seconds_per_try: float = 5.0,
):
    if input_text_action.input_text is None:
        return

    async def _input():
        if input_text_action.input_text is None:
            return
        if input_text_action.fill_or_type == "type":
            pyautogui.typewrite(input_text_action.input_text, interval=0.05)
        else:
            pyperclip.copy(input_text_action.input_text)
            await asyncio.sleep(0.2)
            pyautogui.hotkey(browser.modifier_key, "v")

    try:
        x, y = None, None

        if input_text_action.click_before_input:
            bbox = (
                resolve_bounding_box_variables(
                    input_text_action.bounding_box_variables, memory
                )
                if input_text_action.bounding_box_variables
                else None
            )

            if input_text_action.recording_screenshot and input_text_action.coordinates:
                x, y = await validate_recording_action(
                    input_text_action,
                    browser,
                    memory,
                    task,
                    max_tries,
                    max_timeout_seconds_per_try,
                )
            elif input_text_action.coordinates:
                x = int(input_text_action.coordinates[0])
                y = int(input_text_action.coordinates[1])

                if x == -1 and y == -1:
                    result = await resolve_keyword_with_llm_fallback(
                        keyword=input_text_action.keyword
                        or input_text_action.prompt_instructions,
                        recording_x=-1,
                        recording_y=-1,
                        prompt_instructions=input_text_action.prompt_instructions,
                        memory=memory,
                        task=task,
                        bounding_box=bbox,
                    )
                    if result is None:
                        raise KeywordNotFoundOnScreenException(
                            message=f"Could not locate element on screen for: '{input_text_action.prompt_instructions}'",
                            keyword=input_text_action.keyword
                            or input_text_action.prompt_instructions,
                        )
                    x, y = result
                elif input_text_action.keyword:
                    result = await resolve_keyword_with_llm_fallback(
                        keyword=input_text_action.keyword,
                        recording_x=x,
                        recording_y=y,
                        prompt_instructions=input_text_action.prompt_instructions,
                        memory=memory,
                        task=task,
                        bounding_box=bbox,
                    )
                    if result is None:
                        raise KeywordNotFoundOnScreenException(
                            message=f"Keyword '{input_text_action.keyword}' not found on screen.",
                            keyword=input_text_action.keyword,
                        )
                    x, y = result
                    logger.info(
                        f"Keyword '{input_text_action.keyword}' matched at ({x}, {y})"
                    )
            else:
                data = await get_coordinates_from_prompt(
                    memory, input_text_action.prompt_instructions, browser, task
                )
                if data is None:
                    logger.error("No coordinates found")
                    return
                x, y = data[0], data[1]
                memory.browser_states[-1].llm_response = f"Coordinates: {x}, {y}"
                if input_text_action.keyword:
                    result = await resolve_keyword_with_llm_fallback(
                        keyword=input_text_action.keyword,
                        recording_x=x,
                        recording_y=y,
                        prompt_instructions=input_text_action.prompt_instructions,
                        memory=memory,
                        task=task,
                        bounding_box=bbox,
                    )
                    if result is None:
                        raise KeywordNotFoundOnScreenException(
                            message=f"Keyword '{input_text_action.keyword}' not found on screen.",
                            keyword=input_text_action.keyword,
                        )
                    x, y = result
                    logger.info(
                        f"Keyword '{input_text_action.keyword}' matched at ({x}, {y})"
                    )

            logger.debug(f"Typing text at coordinates: {x}, {y}")
            pyautogui.click(x, y)
            await asyncio.sleep(0.2)

        await _input()
        # changed, score = await wait_for_screen_to_change(_paste, browser)

        # if not changed:
        #     logger.warning("Screen did not change after typing text")

        if input_text_action.press_enter:
            await asyncio.sleep(0.8)
            pyautogui.press("enter")

        if x is not None and y is not None:
            screenshot_base64 = memory.browser_states[-1].screenshot
            if screenshot_base64:
                screenshot_base64 = await mark_screenshot(screenshot_base64, x, y)
                memory.browser_states[-1].screenshot = (
                    screenshot_base64  # pyright: ignore[reportAttributeAccessIssue]
                )

    except (ElementNotFoundInAxtreeException, KeywordNotFoundOnScreenException) as e:
        raise e
    except Exception as e:
        if input_text_action.keyword:
            raise KeywordNotFoundOnScreenException(
                message=f"Failed to verify keyword '{input_text_action.keyword}' on screen due to error: {e}",
                keyword=input_text_action.keyword,
            )
        logger.error(f"Error in input_text_coordinates: {e}")
        return
