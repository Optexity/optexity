import logging

import pyautogui

from optexity.exceptions import (
    ElementNotFoundInAxtreeException,
    KeywordNotFoundOnScreenException,
)
from optexity.inference.core.interaction.handle_command import (
    command_based_action_with_retry,
)
from optexity.inference.core.interaction.screenshot_comparison import (
    _fetch_recording_screenshot,
    compare_screenshots_with_llm,
    crop_screenshot_at_coordinates,
)
from optexity.inference.core.interaction.utils import (
    get_coordinates_from_ocr_result,
    get_coordinates_from_prompt,
    get_index_from_prompt,
    handle_download,
    resolve_keyword_coordinates,
)
from optexity.inference.core.vision.utils import mark_screenshot
from optexity.inference.infra.browser import Browser
from optexity.schema.actions.interaction_action import ClickElementAction
from optexity.schema.memory import Memory
from optexity.schema.task import Task

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05
logger = logging.getLogger(__name__)


async def handle_click_element(
    click_element_action: ClickElementAction,
    task: Task,
    memory: Memory,
    browser: Browser,
    max_timeout_seconds_per_try: float,
    max_tries: int,
):

    if browser.channel == "rdp" or browser.backend == "computer-vision":
        await click_element_coordinates(
            click_element_action, browser, memory, task, max_tries
        )
        return

    if click_element_action.command and not click_element_action.skip_command:
        last_error = await command_based_action_with_retry(
            click_element_action,
            browser,
            memory,
            task,
            max_tries,
            max_timeout_seconds_per_try,
        )

        if last_error is None:
            return

    if not click_element_action.skip_prompt:
        logger.debug(
            f"Executing prompt-based action: {click_element_action.__class__.__name__}"
        )
        await click_element_index(click_element_action, browser, memory, task)


async def click_element_index(
    click_element_action: ClickElementAction,
    browser: Browser,
    memory: Memory,
    task: Task,
):

    try:
        index = await get_index_from_prompt(
            memory, click_element_action.prompt_instructions, browser, task
        )
        if index is None:
            return

        async def _actual_click_element():
            print(
                f"Clicking element with index: {index} and button: {click_element_action.button}"
            )
            action_model = browser.backend_agent.ActionModel(
                **{"click": {"index": index, "button": click_element_action.button}}
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
    except ElementNotFoundInAxtreeException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in click_element_index: {e}")
        return


async def click_element_coordinates(
    click_element_action: ClickElementAction,
    browser: Browser,
    memory: Memory,
    task: Task,
    max_tries: int = 3,
):
    try:
        x, y = None, None

        if (
            click_element_action.recording_screenshot
            and click_element_action.coordinates
        ):
            recording_x = int(click_element_action.coordinates[0])
            recording_y = int(click_element_action.coordinates[1])

            recording_b64 = await _fetch_recording_screenshot(
                click_element_action.recording_screenshot
            )
            recording_crop = crop_screenshot_at_coordinates(
                recording_b64, recording_x, recording_y
            )

            data = await get_coordinates_from_prompt(
                memory, click_element_action.prompt_instructions, browser, task
            )
            if data is None:
                raise KeywordNotFoundOnScreenException(
                    message="Could not locate element on current screen",
                    keyword=click_element_action.keyword or "element",
                )
            current_x, current_y = int(data[0]), int(data[1])
            current_screenshot_b64 = memory.browser_states[-1].screenshot

            matched = False
            for attempt in range(max_tries):
                current_crop = crop_screenshot_at_coordinates(
                    current_screenshot_b64, current_x, current_y
                )
                matches = await compare_screenshots_with_llm(
                    recording_crop,
                    current_crop,
                    click_element_action.keyword,
                    task,
                    memory,
                )
                if matches:
                    matched = True
                    x, y = current_x, current_y
                    break

                logger.info(
                    f"Recording validation attempt {attempt + 1}/{max_tries} failed, retrying..."
                )
                if attempt < max_tries - 1:
                    data = await get_coordinates_from_prompt(
                        memory,
                        click_element_action.prompt_instructions,
                        browser,
                        task,
                    )
                    if data is not None:
                        current_x, current_y = int(data[0]), int(data[1])
                        current_screenshot_b64 = memory.browser_states[-1].screenshot

            if not matched:
                raise KeywordNotFoundOnScreenException(
                    message=(
                        f"Recording screenshot validation failed after {max_tries} "
                        f"attempt(s). Element not found on current screen."
                    ),
                    keyword=click_element_action.keyword or "element",
                )

        elif click_element_action.coordinates:
            x = int(click_element_action.coordinates[0])
            y = int(click_element_action.coordinates[1])
        else:
            data = await get_coordinates_from_prompt(
                memory, click_element_action.prompt_instructions, browser, task
            )
            if data is None:
                logger.error("No coordinates found")
                return
            x = data[0]
            y = data[1]
            memory.browser_states[-1].llm_response = f"Coordinates: {x}, {y}"

        if click_element_action.keyword:
            result = await resolve_keyword_coordinates(
                click_element_action.keyword, x, y, memory
            )
            x_new, y_new = get_coordinates_from_ocr_result(result)
            logger.info(
                f"Matched keyword '{click_element_action.keyword}' with '{result.text}'. "
                f"New coordinates: ({x_new}, {y_new}), old: ({x}, {y})"
            )
            x = x_new
            y = y_new

        logger.debug(f"Clicking element at coordinates: {x}, {y}")

        if click_element_action.double_click:
            pyautogui.doubleClick(x, y)
        else:
            pyautogui.click(x, y)

        screenshot_base64 = memory.browser_states[-1].screenshot
        if screenshot_base64:
            screenshot_base64 = await mark_screenshot(screenshot_base64, x, y)
            memory.browser_states[-1].screenshot = (
                screenshot_base64  # pyright: ignore[reportAttributeAccessIssue]
            )

    except (ElementNotFoundInAxtreeException, KeywordNotFoundOnScreenException) as e:
        raise e
    except Exception as e:
        if click_element_action.keyword:
            raise KeywordNotFoundOnScreenException(
                message=f"Failed to verify keyword '{click_element_action.keyword}' on screen due to error: {e}",
                keyword=click_element_action.keyword,
            )
        logger.error(f"Error in click_element_coordinates: {e}")
        return
