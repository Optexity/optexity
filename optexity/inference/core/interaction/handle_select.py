import logging

from browser_use.dom.serializer.serializer import DOMTreeSerializer

from optexity.exceptions import (
    AxtreeIndexActionFailedException,
    ElementNotFoundInAxtreeException,
    ExpectedDownloadFailedException,
)
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
    LocatorExtraction,
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


async def _playwright_select_option(
    browser: Browser, node, matched_values: list[str]
) -> tuple[bool, list[dict]]:
    """Select an option via Playwright. Tries the shared, ranked, frame-aware,
    uniqueness-verified candidate list (``LocatorExtraction.candidates_from_tree_node``
    — the same heuristic every other tier uses) in order, best-first, instead of
    the single unranked, unescaped CSS guess this used to hand-roll (which had
    the same quote-escaping bug fixed elsewhere in ``LocatorExtraction._css_attr``,
    reintroduced independently here because this code didn't know that fix
    existed). Falls back to a brute-force scan of every frame with the top
    candidate's bare selector only if none of the ranked candidates resolve —
    a last resort, not the primary mechanism.

    Returns ``(success, candidates)`` so the caller can log whichever candidate
    actually worked, instead of logging the failed primary index guess.
    """
    page = await browser.get_current_page()
    method_str = f".select_option({matched_values[0]!r})"
    candidates = await LocatorExtraction.candidates_from_tree_node(
        node, method_str, page
    )
    for candidate in candidates:
        try:
            located = eval(candidate["locator"].removesuffix(method_str))
            if await located.count() == 1:
                await located.select_option(value=matched_values[0])
                return True, candidates
        except Exception:
            continue

    # Last-resort brute force: the frame-chain candidates above all failed to
    # resolve (e.g. a frame boundary selector went stale) — scan every frame
    # directly with whatever bare selector scored highest. Reuses
    # `_bare_frame_selector` off-label (it's named/tuned for iframe boundary
    # elements, but its id/name/data-testid/title priority is exactly the
    # "single best raw selector" this brute-force scan also needs — a `src`
    # match just never fires for a non-iframe node like a `<select>`).
    fallback_selector = LocatorExtraction._bare_frame_selector(node) or None
    if fallback_selector:
        for frame in page.frames:
            try:
                locator = frame.locator(fallback_selector)
                if await locator.count() > 0:
                    await locator.first.select_option(value=matched_values[0])
                    return True, candidates
            except Exception:
                continue

    return False, candidates


async def select_option_index(
    select_option_action: SelectOptionAction,
    browser: Browser,
    memory: Memory,
    task: Task,
):
    ## TODO either perfect text match or agenic select value prediction
    try:

        # Second value (the EnhancedDOMTreeNode) is unused here — this handler
        # already fetches `node` separately below via `get_element_by_index`
        # (to read the <select>'s options), which serves the same "pre-action"
        # purpose the other handlers use `get_index_from_prompt`'s node for.
        index, _ = await get_index_from_prompt(
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
            raise AxtreeIndexActionFailedException(
                message=f"Failed to resolve element at axtree index {index} for select_option",
                index=index,
                original_error="get_element_by_index returned None",
            )

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
                playwright_success, candidates = await _playwright_select_option(
                    browser, node, matched_values
                )
                logger.debug(
                    f"Playwright select_option succeeded: {playwright_success}"
                )
                if not playwright_success:
                    raise RuntimeError(
                        f"select_dropdown failed and playwright fallback miss: {results[0].error}"
                    )
                # Record the candidate that actually worked via the fallback,
                # not the primary index guess that just failed — logging that
                # unconditionally (the old behavior) attributed a locator to a
                # path that never ran.
                LocatorExtraction.record_locator_candidates(memory, candidates)
            else:
                await LocatorExtraction.log_interacted_locator(
                    browser,
                    index,
                    node,
                    f".select_option({matched_values[0]!r})",
                    memory,
                )

        try:
            if select_option_action.expect_download:
                await handle_download(
                    _actual_select_option,
                    memory,
                    browser,
                    task,
                    select_option_action.download_filename,
                    select_option_action.download_metadata,
                )
            else:
                await _actual_select_option()
        except ExpectedDownloadFailedException:
            # expect_download was True but no file was produced; fail the task
            # with the fixed message instead of masking it as a select failure.
            raise
        except Exception as e:
            raise AxtreeIndexActionFailedException(
                message=f"Failed to select option at axtree index {index}",
                index=index,
                original_error=e,
            )
    except (
        ElementNotFoundInAxtreeException,
        AxtreeIndexActionFailedException,
        ExpectedDownloadFailedException,
    ):
        raise
    except Exception as e:
        logger.error(f"Error in select_option_index: {e}")
        return
