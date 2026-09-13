import logging
from typing import Any

from browser_use import Agent, BrowserSession, Tools

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


async def handle_agentic_task(
    agentic_task_action: AgenticTask | CloseOverlayPopupAction,
    task: Task,
    memory: Memory,
    browser: Browser,
):
    """Returns ``(history, interacted_nodes)``. ``interacted_nodes`` maps
    ``(step_number, action_index) -> EnhancedDOMTreeNode`` for every action the
    agent took that resolved to a live selector-map entry, captured *live* via
    ``register_new_step_callback`` — i.e. from the same pre-action
    ``browser_state_summary`` that decided each action, before browser_use
    collapses it into the lossy ``DOMInteractedElement`` (which drops
    ``parent_node`, making frame-chain reconstruction impossible from
    ``history`` alone). Callers that don't need locator evidence (the
    ``CloseOverlayPopupAction`` call sites) can simply discard the second
    value.
    """
    interacted_nodes: dict[tuple[int, int], Any] = {}

    def _capture_interacted_nodes(browser_state_summary, model_output, step_number):
        selector_map = browser_state_summary.dom_state.selector_map
        for action_index, action in enumerate(model_output.action):
            index = action.get_index()
            if index is not None and index in selector_map:
                interacted_nodes[(step_number, action_index)] = selector_map[index]

    if agentic_task_action.backend == "browser_use":

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
            register_new_step_callback=_capture_interacted_nodes,
        )
        logger.debug(f"Starting browser session for agentic task {browser.cdp_url} ")
        await agent.browser_session.start()
        logger.debug(f"Finally running agentic task on browser_use {browser.cdp_url} ")
        history = await agent.run(max_steps=agentic_task_action.max_steps)
        logger.debug(f"Agentic task completed on browser_use {browser.cdp_url} ")

        agent.stop()
        if agent.browser_session:
            await agent.browser_session.stop()
            await agent.browser_session.reset()

        return history, interacted_nodes

    elif agentic_task_action.backend == "browserbase":
        raise NotImplementedError("Browserbase is not supported yet")

    return None, interacted_nodes
