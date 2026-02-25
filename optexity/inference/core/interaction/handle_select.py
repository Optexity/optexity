import logging

from browser_use.dom.serializer.serializer import DOMTreeSerializer

from optexity.exceptions import ElementNotFoundInAxtreeException
from optexity.inference.core.interaction.handle_command import (
    command_based_action_with_retry,
)
from optexity.inference.core.interaction.handle_select_utils import (
    SelectOptionValue,
    smart_select,
)
from optexity.inference.core.interaction.utils import (
    get_index_from_prompt,
    handle_download,
)
from optexity.inference.infra.browser import Browser
from optexity.schema.actions.interaction_action import SelectOptionAction
from optexity.schema.memory import Memory
from optexity.schema.task import Task

logger = logging.getLogger(__name__)


async def handle_select_option(
    select_option_action: SelectOptionAction,
    task: Task,
    memory: Memory,
    browser: Browser,
    max_timeout_seconds_per_try: float,
    max_tries: int,
):

    if select_option_action.command and not select_option_action.skip_command:
        last_error = await command_based_action_with_retry(
            select_option_action,
            browser,
            memory,
            task,
            max_tries,
            max_timeout_seconds_per_try,
        )

        if last_error is None:
            return

    if not select_option_action.skip_prompt:
        logger.debug(
            f"Executing prompt-based action: {select_option_action.__class__.__name__}"
        )
        await select_option_index(select_option_action, browser, memory, task)


async def select_option_index(
    select_option_action: SelectOptionAction,
    browser: Browser,
    memory: Memory,
    task: Task,
):
    ## TODO either perfect text match or agenic select value prediction
    try:

        index = await get_index_from_prompt(
            memory, select_option_action.prompt_instructions, browser, task
        )
        if index is None:
            return

        node = await browser.backend_agent.browser_session.get_element_by_index(index)
        if node is None:
            return

        select_option_values = DOMTreeSerializer(node)._extract_select_options(node)
        if select_option_values is None:
            return

        all_options = select_option_values["all_options"]

        all_options = [
            SelectOptionValue(value=o["value"], label=o["text"]) for o in all_options
        ]

        matched_values = await smart_select(
            all_options, select_option_action.select_values, memory
        )

        async def _actual_select_option():
            action_model = browser.backend_agent.ActionModel(
                **{
                    "select_dropdown": {
                        "index": int(index),
                        "text": matched_values[0],
                    }
                }
            )
            await browser.backend_agent.multi_act([action_model])

        if select_option_action.expect_download:
            await handle_download(
                _actual_select_option,
                memory,
                browser,
                task,
                select_option_action.download_filename,
            )
        else:
            await _actual_select_option()
    except ElementNotFoundInAxtreeException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in select_option_index: {e}")
        return
