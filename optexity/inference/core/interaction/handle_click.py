import logging

import pyautogui

from optexity.exceptions import (
    ElementNotFoundInAxtreeException,
)
from optexity.inference.core.interaction.handle_command import (
    command_based_action_with_retry,
)
from optexity.inference.core.interaction.utils import (
    get_coordinates_from_prompt,
    get_index_from_prompt,
    handle_download,
)
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
        await click_element_coordinates(click_element_action, browser, memory, task)
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
):

    try:

        if click_element_action.coordinates:
            x = click_element_action.coordinates[0]
            y = click_element_action.coordinates[1]
            logger.debug(f"Clicking element at coordinates: {x}, {y}")
            pyautogui.click(x, y)
            return

        data = await get_coordinates_from_prompt(
            memory, click_element_action.prompt_instructions, browser, task
        )

        if data is None:
            logger.error("No coordinates found")
            return

        x = data[0]
        y = data[1]

        logger.debug(f"Clicking element at coordinates: {x}, {y}")

        pyautogui.click(x, y)
    except ElementNotFoundInAxtreeException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in click_element_index: {e}")
        return
