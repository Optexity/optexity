import logging

from browser_use.dom.serializer.serializer import DOMTreeSerializer

from optexity.exceptions import ElementNotFoundInAxtreeException
from optexity.inference.agents.select_option_prediction.select_option_prediction import (
    SelectOptionPredictionAgent,
)
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
    update_screenshot_with_highlight,
)
from optexity.inference.infra.browser import Browser
from optexity.inference.models import get_llm_model_with_fallback
from optexity.schema.actions.interaction_action import SelectOptionAction
from optexity.schema.memory import BrowserState, Memory
from optexity.schema.task import Task

logger = logging.getLogger(__name__)

_select_option_prediction_cache: dict[tuple, SelectOptionPredictionAgent] = {}


def _get_select_option_prediction_agent(task: Task) -> SelectOptionPredictionAgent:
    cache_key = (task.llm_provider, task.llm_model_name)
    if cache_key not in _select_option_prediction_cache:
        model = get_llm_model_with_fallback(
            task.llm_provider, task.llm_model_name, True
        )
        _select_option_prediction_cache[cache_key] = SelectOptionPredictionAgent(model)
    return _select_option_prediction_cache[cache_key]


async def llm_select_option_prediction(
    prompt_instructions: str, browser: Browser, memory: Memory, task: Task
) -> list[str]:
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
        final_prompt, response, token_usage = _get_select_option_prediction_agent(
            task
        ).predict_select_option(
            prompt_instructions,
            memory.browser_states[-1].axtree,
            memory.browser_states[-1].screenshot,
        )
        memory.token_usage += token_usage
        memory.browser_states[-1].final_prompt = final_prompt
        memory.browser_states[-1].llm_response = response.model_dump()
    except Exception as e:
        logger.error(f"Error in llm_select_option_prediction: {e}")
        return None

    return response.select_values


async def handle_select_option(
    select_option_action: SelectOptionAction,
    task: Task,
    memory: Memory,
    browser: Browser,
    max_timeout_seconds_per_try: float,
    max_tries: int,
):

    if (
        select_option_action.select_values is None
        and not select_option_action.skip_prompt
        and select_option_action.prompt_instructions is not None
    ):
        select_option_action.select_values = await llm_select_option_prediction(
            select_option_action.prompt_instructions,
            browser,
            memory,
            task,
        )

    if select_option_action.select_values is None:
        logger.debug(
            f"Select values is None for action: {select_option_action.__class__.__name__}, skipping action"
        )
        return

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


def _build_css_selector(node) -> str | None:
    """Build a CSS selector from the node's attributes to locate it in the live DOM."""
    tag = node.node_name.lower() if node.node_name else "select"
    attrs = node.attributes or {}

    for attr in ("id", "name", "data-testid", "aria-label"):
        val = attrs.get(attr)
        if val:
            return f'{tag}[{attr}="{val}"]'

    return None


async def _playwright_select_option(
    browser: Browser, node, matched_values: list[str]
) -> bool:
    """Select an option via Playwright, searching across all frames (pierces shadow DOM and iframes)."""
    css_selector = _build_css_selector(node)
    if css_selector is None:
        return False

    page = await browser.get_current_page()

    for frame in page.frames:
        try:
            locator = frame.locator(css_selector)
            if await locator.count() > 0:
                await locator.first.select_option(value=matched_values[0])
                return True
        except Exception:
            continue

    return False


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
        try:
            await update_screenshot_with_highlight(browser, memory, index)
        except Exception as e:
            logger.error(
                f"Error in updating screenshot with highlight in select_option_index: {e}"
            )

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
            all_options, select_option_action.select_values, memory, task
        )

        logger.debug(
            f"Matched values for {select_option_action.command}: {matched_values}"
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
            results = await browser.backend_agent.multi_act([action_model])
            if results and results[0].error:
                logger.debug(
                    f"Falling back to playwright select_option: {results[0].error}"
                )
                playwright_success = await _playwright_select_option(
                    browser, node, matched_values
                )
                logger.debug(
                    f"Playwright select_option succeeded: {playwright_success}"
                )

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
