import logging

from optexity.exceptions import ElementNotFoundInAxtreeException
from optexity.inference.core.interaction.handle_command import (
    command_based_action_with_retry,
)
from optexity.inference.core.interaction.utils import get_index_from_prompt
from optexity.inference.infra.browser import Browser
from optexity.schema.actions.interaction_action import HoverAction
from optexity.schema.memory import Memory
from optexity.schema.task import Task

logger = logging.getLogger(__name__)


async def handle_hover_element(
    hover_element_action: HoverAction,
    task: Task,
    memory: Memory,
    browser: Browser,
    max_timeout_seconds_per_try: float,
    max_tries: int,
):

    if hover_element_action.command and not hover_element_action.skip_command:
        last_error = await command_based_action_with_retry(
            hover_element_action,
            browser,
            memory,
            task,
            max_tries,
            max_timeout_seconds_per_try,
        )

        if last_error is None:
            return

    if not hover_element_action.skip_prompt:
        logger.debug(
            f"Executing prompt-based action: {hover_element_action.__class__.__name__}"
        )
        await hover_element_index(hover_element_action, browser, memory, task)


async def hover_element_index(
    hover_element_action: HoverAction,
    browser: Browser,
    memory: Memory,
    task: Task,
):

    try:
        index = await get_index_from_prompt(
            memory, hover_element_action.prompt_instructions, browser, task
        )
        if index is None:
            return

        print(f"Hovering element with index: {index}")

        async def _actual_hover_element():
            try:
                action_model = browser.backend_agent.ActionModel(
                    **{"hover": {"index": index}}
                )
                await browser.backend_agent.multi_act([action_model])
            except Exception as e:
                logger.error(f"Error in hover_element_index: {e} trying right click")
                node = await browser.backend_agent.browser_session.get_element_by_index(
                    index
                )
                if node is None:
                    return

                backend_page = (
                    await browser.backend_agent.browser_session.get_current_page()
                )
                element = await backend_page.get_element(node.backend_node_id)
                await element.click(button="right")

        await _actual_hover_element()
    except ElementNotFoundInAxtreeException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in hover_element_index: {e}")
        return
