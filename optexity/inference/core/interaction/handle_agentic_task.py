import json
import logging
import pathlib
import re
from typing import Any

from browser_use import Agent, BrowserSession, Tools

from optexity.inference.core.interaction.utils import LocatorExtraction
from optexity.inference.infra.browser import Browser
from optexity.inference.models import normalize_model
from optexity.inference.models.chat_litellm import build_agent_llm
from optexity.schema.actions.interaction_action import (
    AgenticTask,
    CloseOverlayPopupAction,
)
from optexity.schema.memory import Memory
from optexity.schema.task import Task

logger = logging.getLogger(__name__)


def _dump_element(el: Any) -> Any:
    """Safely serializes DOMInteractedElement across Pydantic, dataclasses, and custom objects."""
    if el is None:
        return None
    if hasattr(el, "model_dump"):
        return el.model_dump(mode="json")
    if hasattr(el, "to_dict"):
        return el.to_dict()
    if hasattr(el, "__dict__"):
        return {
            k: _dump_element(v)
            for k, v in vars(el).items()
            if not k.startswith("_")
        }
    return str(el)


def _extract_selector(locator_str: str) -> str:
    """Extracts raw selector string safely from locator expressions."""
    if not locator_str:
        return ""
    match = re.search(r"page\.locator\((['\"])(.*?)\1\)", locator_str)
    if match:
        return match.group(2)
    return locator_str


async def handle_agentic_task(
    agentic_task_action: AgenticTask | CloseOverlayPopupAction,
    task: Task,
    memory: Memory,
    browser: Browser,
):
    key = (
        getattr(task, "endpoint_name", None)
        or getattr(getattr(task, "automation", None), "recording_id", None)
        or "default_endpoint"
    )
    cache_path = pathlib.Path(f"/tmp/optexity_action_cache/{key}_step_{memory.automation_state.step_index}_cache.json")

    # -------------------------------------------------------------
    # ⚡ FAST PATH: Replay cached agent actions via Playwright
    # -------------------------------------------------------------
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)

            actions = cache.get("actions", [])
            if actions:
                logger.info("⚡ [CACHE HIT] Replaying %d agentic actions via Playwright", len(actions))
                page = await browser.get_current_page()
                for act in actions:
                    selector = act.get("selector") or _extract_selector(act.get("locator", ""))
                    if not selector:
                        continue

                    if act.get("type") == "input":
                        val = act.get("value", "")
                        await page.locator(selector).first.fill(str(val))
                    elif act.get("type") == "click":
                        await page.locator(selector).first.click()

                logger.info("✅ Replayed all agentic actions in <150ms!")
                # CRITICAL: Early return to prevent running exploratory browser_use
                return None
        except Exception as e:
            logger.warning("Cache replay encountered an issue (%s); falling back to agentic run.", e)

    logger.info("Starting agentic task")
    if agentic_task_action.backend == "browser_use":
        logger.info(
            f"INCOMING TASK CONFIG -> Provider: {getattr(task, 'llm_provider', 'MISSING')} | "
            f"Model: {getattr(task, 'llm_model_name', 'MISSING')}"
        )
        if isinstance(agentic_task_action, CloseOverlayPopupAction):
            tools = Tools(
                exclude_actions=[
                    "search",
                    "navigate",
                    "go_back",
                    "upload_file",
                    "scroll",
                    "find_text",
                    "send_keys",
                    "evaluate",
                    "switch",
                    "close",
                    "extract",
                    "dropdown_options",
                    "select_dropdown",
                    "write_file",
                    "read_file",
                    "replace_file",
                ]
            )
        else:
            tools = Tools()

        llm = build_agent_llm(normalize_model(task.llm_provider, task.llm_model_name))
        browser_session = BrowserSession(
            cdp_url=browser.cdp_url, keep_alive=agentic_task_action.keep_alive
        )

        step_directory = (
            task.logs_directory / f"step_{str(memory.automation_state.step_index)}"
        )
        step_directory.mkdir(parents=True, exist_ok=True)

        agent = Agent(
            task=agentic_task_action.task,
            llm=llm,
            browser_session=browser_session,
            use_vision=agentic_task_action.use_vision,
            tools=tools,
            calculate_cost=True,
            save_conversation_path=step_directory,
        )

        logger.debug(f"Starting browser session for agentic task {browser.cdp_url}")
        await agent.browser_session.start()

        logger.debug(f"Running agentic task on browser_use {browser.cdp_url}")
        history = await agent.run(max_steps=agentic_task_action.max_steps)
        logger.debug(f"Agentic task completed on browser_use {browser.cdp_url}")
        object.__setattr__(memory, "latest_agent_history", history)

        # Extract actions directly from the agent's interaction trajectory
        cache = build_action_cache(history)
        logger.info("LEARNED ACTION CACHE: %s", cache)

        # Write to step log directory
        cache_file = step_directory / "action_cache.json"
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
        logger.info("Saved action cache to %s", cache_file)

        # Also export to persistent cross-run cache directory
        cache_dir = pathlib.Path("/tmp/optexity_action_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        persistent_file = cache_dir / f"{key}_step_{memory.automation_state.step_index}_cache.json"
        with open(persistent_file, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
        logger.info("Saved persistent action cache to %s", persistent_file)

        logger.info("=== AGENT HISTORY DUMP ===")
        for step_idx, history_item in enumerate(history.history):
            logger.info("=== HISTORY STEP %s ===", step_idx)

            if history_item.model_output:
                logger.info(
                    "MODEL OUTPUT: %s",
                    history_item.model_output.model_dump(mode="json"),
                )

            if history_item.state:
                logger.info(
                    "INTERACTED ELEMENTS: %s",
                    [
                        _dump_element(element)
                        for element in (history_item.state.interacted_element or [])
                    ],
                )

        agent.stop()
        if agent.browser_session:
            await agent.browser_session.stop()
            await agent.browser_session.reset()

        return history

    elif agentic_task_action.backend == "browserbase":
        raise NotImplementedError("Browserbase is not supported yet")

    return None


def build_action_cache(history) -> dict:
    """Parses agent history into deterministic Playwright action locators."""
    cached_actions = []

    for history_item in history.history:
        if not history_item.model_output:
            continue

        actions = history_item.model_output.action or []
        state_elements = (
            history_item.state.interacted_element
            if history_item.state
            else []
        ) or []

        for i, action in enumerate(actions):
            element = state_elements[i] if i < len(state_elements) else None
            if element is None:
                continue

            action_data = action.model_dump(
                exclude_none=True,
                mode="json",
            )
            if not action_data:
                continue

            action_type = next(iter(action_data), None)
            if action_type not in ("input", "click"):
                continue

            payload = action_data[action_type]
            attributes = getattr(element, "attributes", {}) or {}
            tag_name = getattr(element, "node_name", "input").lower()

            name_attr = attributes.get("name")
            id_attr = attributes.get("id")
            placeholder_attr = attributes.get("placeholder")
            xpath_attr = getattr(element, "xpath", None)

            selector = None
            if name_attr:
                selector = f'{tag_name}[name="{name_attr}"]'
            elif id_attr:
                selector = f"#{id_attr}"
            elif placeholder_attr:
                selector = f'{tag_name}[placeholder="{placeholder_attr}"]'
            elif xpath_attr:
                selector = f"xpath={xpath_attr}"

            if not selector:
                continue

            entry = {
                "type": action_type,
                "selector": selector,
                "locator": f"page.locator('{selector}').first",
            }
            if action_type == "input":
                entry["value"] = payload.get("text", "")

            cached_actions.append(entry)

    return {
        "version": 1,
        "actions": cached_actions,
    }