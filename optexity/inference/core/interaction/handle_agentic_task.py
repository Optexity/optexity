import logging

from browser_use import Agent, BrowserSession, Tools
from browser_use.agent.history_compiler import compile_history_to_action_cache

from optexity.inference.infra.browser import Browser
from optexity.inference.models import normalize_model
from optexity.inference.models.chat_litellm import build_agent_llm
from optexity.schema.actions.interaction_action import (
    AgenticTask,
    CloseOverlayPopupAction,
)
from optexity.schema.memory import Memory
from optexity.schema.task import Task
from optexity.schema.token_usage import TokenUsage

logger = logging.getLogger(__name__)

AGENT_HISTORY_FILENAME = "raw_history.json"
AGENT_ACTION_CACHE_FILENAME = "browser_use_action_cache.json"


async def handle_agentic_task(
    agentic_task_action: AgenticTask | CloseOverlayPopupAction,
    task: Task,
    memory: Memory,
    browser: Browser,
):

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
            task.logs_directory / f"step_{memory.automation_state.step_index!s}"
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
        logger.debug(f"Starting browser session for agentic task {browser.cdp_url} ")
        await agent.browser_session.start()
        try:
            logger.debug(
                f"Finally running agentic task on browser_use {browser.cdp_url} "
            )
            history = await agent.run(max_steps=agentic_task_action.max_steps)
            if history.usage is not None:
                memory.token_usage += TokenUsage(
                    input_tokens=history.usage.total_prompt_tokens,
                    output_tokens=history.usage.total_completion_tokens,
                    total_tokens=history.usage.total_tokens,
                    calculated_total_tokens=history.usage.total_tokens,
                    input_cost=history.usage.total_prompt_cost,
                    output_cost=history.usage.total_completion_cost,
                    total_cost=history.usage.total_cost,
                )
            history_path = step_directory / AGENT_HISTORY_FILENAME
            agent.save_history(history_path)
            logger.info(
                "Saved browser-use agent history with %d steps to %s",
                len(history),
                history_path,
            )
            cache_path = step_directory / AGENT_ACTION_CACHE_FILENAME
            try:
                compile_history_to_action_cache(
                    history_path,
                    cache_path,
                    task_instruction=agentic_task_action.task,
                )
            except Exception:
                logger.exception(
                    "Failed to compile browser-use agent history cache from %s",
                    history_path,
                )
            logger.debug(f"Agentic task completed on browser_use {browser.cdp_url} ")
            return history
        finally:
            agent.stop()
            if agent.browser_session:
                await agent.browser_session.stop()
                await agent.browser_session.reset()

    elif agentic_task_action.backend == "browserbase":
        raise NotImplementedError("Browserbase is not supported yet")

    return None
