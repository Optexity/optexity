import json
import logging
import pathlib
import re

from optexity.exceptions import (
    AxtreeIndexActionFailedException,
    ElementNotFoundInAxtreeException,
    ExpectedDownloadFailedException,
)
from optexity.inference.core.interaction.handle_command import (
    command_based_action_with_retry,
)
from optexity.inference.core.interaction.utils import (
    LocatorExtraction,
    get_index_from_prompt,
    handle_download,
    update_screenshot_with_highlight,
)
from optexity.inference.infra.browser import Browser
from optexity.schema.actions.interaction_action import ClickElementAction
from optexity.schema.memory import Memory
from optexity.schema.task import Task

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parameter Templating Helpers
# ---------------------------------------------------------------------------

def _get_input_parameters(task: Task) -> dict:
    """Extract input parameters dictionary across task schema variations."""
    if hasattr(task, "input_parameters") and task.input_parameters:
        return task.input_parameters
    if hasattr(task, "automation") and hasattr(task.automation, "parameters"):
        return getattr(task.automation.parameters, "input_parameters", {}) or {}
    return {}


def parameterize_locator(raw_locator: str, input_params: dict) -> tuple[str, list[str]]:
    """
    Replaces literal input parameter values in a locator string with placeholders.
    Example: 'has_text="SWX:AAPL"' -> 'has_text="SWX:{stock_ticker[0]}"'
    """
    parameterized = raw_locator
    bound_keys = []

    for key, val in input_params.items():
        if isinstance(val, list):
            for idx, item in enumerate(val):
                str_item = str(item)
                if str_item and str_item in parameterized:
                    placeholder = f"{{{key}[{idx}]}}"
                    parameterized = parameterized.replace(str_item, placeholder)
                    bound_keys.append(key)
        elif isinstance(val, str) and val in parameterized:
            placeholder = f"{{{key}}}"
            parameterized = parameterized.replace(val, placeholder)
            bound_keys.append(key)

    # Disambiguate strict-mode violations: ensure .first is present before action call
    if not re.search(r"\.(first|nth\(\d+\)|last)\b", parameterized):
        parameterized = re.sub(r"(\)\.click\()", r").first.click(", parameterized)

    return parameterized, list(set(bound_keys))


def hydrate_locator_template(template: str, input_params: dict) -> str:
    """Fills placeholders in the template with active runtime parameter values."""
    try:
        return template.format(**input_params)
    except (KeyError, IndexError, ValueError) as e:
        logger.warning(f"Failed to hydrate locator template '{template}' with params {input_params}: {e}")
        return template


# ---------------------------------------------------------------------------
# Cache Persistence and Replay
# ---------------------------------------------------------------------------

def _get_persistent_cache_file(task: Task, memory: Memory) -> pathlib.Path:
    """Generates a stable cache path keyed by endpoint/recording and step index."""
    step_idx = memory.automation_state.step_index
    cache_dir = pathlib.Path("/tmp/optexity_action_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    key = (
        getattr(task, "endpoint_name", None)
        or getattr(task.automation, "recording_id", None)
        or "default_endpoint"
    )
    return cache_dir / f"{key}_step_{step_idx}_cache.json"


async def _try_replay_cached_action(
    browser: Browser, task: Task, memory: Memory
) -> bool:
    """
    Fast replay layer: executes hydrated template directly on Playwright page.
    Times out in 1.5s to fail-fast into exploration if DOM shifted.
    """
    step_idx = memory.automation_state.step_index
    candidates = [
        _get_persistent_cache_file(task, memory),
        pathlib.Path("action_cache.json"),
        task.logs_directory / f"step_{step_idx}" / "action_cache.json",
    ]

    cache_file = next((p for p in candidates if p.exists() and p.stat().st_size > 0), None)
    if not cache_file:
        return False

    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            cache_data = json.load(f)

        actions = cache_data.get("actions", [])
        if not actions:
            return False

        action_data = actions[0]
        template = action_data.get("locator_template") or action_data.get("locator")
        if not template:
            return False

        input_params = _get_input_parameters(task)
        hydrated_cmd = hydrate_locator_template(template, input_params)

        # Ensure Playwright action uses strict timeout to fail fast
        if "timeout=" not in hydrated_cmd:
            hydrated_cmd = re.sub(r"click\((.*?)\)", r"click(\1, timeout=1500)", hydrated_cmd)

        logger.info(f"⚡ [CACHE HIT] Replaying parameter-hydrated locator: {hydrated_cmd}")

        page = browser.page
        if hydrated_cmd.startswith("page."):
            await eval(hydrated_cmd, {"page": page})
        else:
            await eval(f"page.{hydrated_cmd}", {"page": page})

        logger.info("✅ Cached action completed successfully in <50ms!")
        return True

    except Exception as e:
        logger.warning(f"⚠️ Cache replay failed ({e}). Falling back to exploratory resolution.")
        return False


# ---------------------------------------------------------------------------
# Interaction Dispatcher
# ---------------------------------------------------------------------------

async def handle_click_element(
    click_element_action: ClickElementAction,
    task: Task,
    memory: Memory,
    browser: Browser,
    max_timeout_seconds_per_try: float,
    max_tries: int,
):
    # 1. Check parameter-aware deterministic cache FIRST (bypasses 10s strict mode violation)
    if await _try_replay_cached_action(browser, task, memory):
        return

    # 2. Existing static command fallback
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

    # 3. Exploratory fallback (AXTree + LLM)
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
        try:
            await update_screenshot_with_highlight(browser, memory, index)
        except Exception as e:
            logger.error(
                f"Error in updating screenshot with highlight in click_element_index: {e}"
            )

        async def _actual_click_element():
            print(
                f"Clicking element with index: {index} and button: {click_element_action.button}"
            )
            action_model = browser.backend_agent.ActionModel(
                **{"click": {"index": index, "button": click_element_action.button}}
            )
            results = await browser.backend_agent.multi_act([action_model])
            await LocatorExtraction.log_interacted_locator(
                browser,
                index,
                f".click(button={click_element_action.button!r})",
                memory,
            )
            candidates = (
                memory.browser_states[-1].locator_candidates
                if memory.browser_states
                else None
            )

            if candidates:
                best_locator = candidates[0]["locator"]
                input_params = _get_input_parameters(task)

                # Parameterize before storing
                template, bound_keys = parameterize_locator(best_locator, input_params)

                cache = {
                    "version": 2,
                    "actions": [
                        {
                            "type": "click",
                            "locator_template": template,
                            "bound_parameters": bound_keys,
                            "raw_resolved_locator": best_locator,
                            "candidates": candidates,
                        }
                    ],
                }

                print(f"🔥 LEARNED PARAMETERIZED CACHE: {cache}", flush=True)

                # Write to run log
                step_idx = memory.automation_state.step_index
                log_cache_file = task.logs_directory / f"step_{step_idx}" / "action_cache.json"
                log_cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(log_cache_file, "w", encoding="utf-8") as f:
                    json.dump(cache, f, indent=2)

                # Write to cross-run persistent store
                persistent_file = _get_persistent_cache_file(task, memory)
                with open(persistent_file, "w", encoding="utf-8") as f:
                    json.dump(cache, f, indent=2)

                print(f"🔥 SAVED TO PERSISTENT CACHE: {persistent_file}", flush=True)

            if results and results[0].error:
                raise RuntimeError(
                    f"browseruse click failed at index {index}: {results[0].error}"
                )

        try:
            if click_element_action.expect_download:
                await handle_download(
                    _actual_click_element,
                    memory,
                    browser,
                    task,
                    click_element_action.download_filename,
                    click_element_action.download_metadata,
                )
            else:
                await _actual_click_element()
        except ExpectedDownloadFailedException:
            raise
        except Exception as e:
            raise AxtreeIndexActionFailedException(
                message=f"Failed to click element at axtree index {index}",
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
        logger.error(f"Error in click_element_index: {e}")
        return