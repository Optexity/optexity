import logging
import re

from optexity.exceptions import ElementNotFoundInAxtreeException
from optexity.inference.agents.input_text_prediction.input_text_prediction import (
    InputTextPredictionAgent,
)
from optexity.inference.core.interaction.handle_command import (
    command_based_action_with_retry,
)
from optexity.inference.core.interaction.utils import (
    get_index_from_prompt,
    update_screenshot_with_highlight,
)
from optexity.inference.infra.browser import Browser
from optexity.inference.models import get_llm_model_with_fallback
from optexity.schema.actions.interaction_action import InputTextAction
from optexity.schema.memory import BrowserState, Memory
from optexity.schema.task import Task

logger = logging.getLogger(__name__)

_input_text_prediction_cache: dict[tuple, InputTextPredictionAgent] = {}


def _get_input_text_prediction_agent(task: Task) -> InputTextPredictionAgent:
    cache_key = (task.llm_provider, task.llm_model_name)
    if cache_key not in _input_text_prediction_cache:
        model = get_llm_model_with_fallback(
            task.llm_provider, task.llm_model_name, True
        )
        _input_text_prediction_cache[cache_key] = InputTextPredictionAgent(model)
    return _input_text_prediction_cache[cache_key]


async def llm_input_text_prediction(
    prompt_instructions: str, browser: Browser, memory: Memory, task: Task
) -> str:
    browser_state_summary = await browser.get_browser_state_summary()
    memory.browser_states[-1] = BrowserState(
        url=browser_state_summary.url,
        screenshot=browser_state_summary.screenshot,
        title=browser_state_summary.title,
        axtree=browser_state_summary.dom_state.llm_representation(
            remove_empty_nodes=task.automation.remove_empty_nodes_in_axtree
        ),
    )

    try:
        if memory.browser_states[-1].axtree is None:
            logger.error("Axtree is None, cannot predict action")
            return None
        final_prompt, response, token_usage = _get_input_text_prediction_agent(
            task
        ).predict_input_text(
            prompt_instructions,
            memory.browser_states[-1].axtree,
            memory.browser_states[-1].screenshot,
        )
        memory.token_usage += token_usage
        memory.browser_states[-1].final_prompt = final_prompt
        memory.browser_states[-1].llm_response = response.model_dump()
    except Exception as e:
        logger.error(f"Error in llm_input_text_prediction: {e}")
        return None

    return response.input_text


async def handle_input_text(
    input_text_action: InputTextAction,
    task: Task,
    memory: Memory,
    browser: Browser,
    max_timeout_seconds_per_try: float,
    max_tries: int,
):

    if (
        input_text_action.input_text is None
        and not input_text_action.skip_prompt
        and input_text_action.prompt_instructions is not None
    ):
        input_text_action.input_text = await llm_input_text_prediction(
            input_text_action.prompt_instructions,
            browser,
            memory,
            task,
        )

    if input_text_action.input_text is None:
        logger.debug(
            f"Input text is None for action: {input_text_action.__class__.__name__}"
        )
        return

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
