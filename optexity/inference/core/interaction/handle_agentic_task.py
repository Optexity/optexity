import logging

from browser_use import Agent, BrowserSession, Tools

from optexity.inference.infra.browser import Browser
from optexity.inference.models import normalize_model
from optexity.inference.models.chat_openrouter import build_agent_llm
from optexity.inference.models.llm_model import to_token_usage
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
    feature: str = "agentic_task",
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
        llm = build_agent_llm(
            normalize_model(task.llm_provider, task.llm_model_name), feature
        )
        browser_session = BrowserSession(
            cdp_url=browser.cdp_url, keep_alive=agentic_task_action.keep_alive
        )

        step_directory = (
            task.logs_directory / f"step_{str(memory.automation_state.step_index)}"
        )
        step_directory.mkdir(parents=True, exist_ok=True)

        try:
            agent = Agent(
                task=agentic_task_action.task,
                llm=llm,
                browser_session=browser_session,
                use_vision=agentic_task_action.use_vision,
                tools=tools,
                calculate_cost=True,
                llm_timeout=llm.call_timeout,
                save_conversation_path=step_directory,
            )
            logger.debug(
                f"Starting browser session for agentic task {browser.cdp_url} "
            )
            await agent.browser_session.start()
            logger.debug(
                f"Finally running agentic task on browser_use {browser.cdp_url} "
            )
            history = await agent.run(max_steps=agentic_task_action.max_steps)
            logger.debug(f"Agentic task completed on browser_use {browser.cdp_url} ")

            agent.stop()
            if agent.browser_session:
                await agent.browser_session.stop()
                await agent.browser_session.reset()

            return history
        finally:
            # browser-use doesn't add its LLM calls to the task's usage, so do it here.
            memory.token_usage += to_token_usage(llm.usage)

    elif agentic_task_action.backend == "browserbase":
        raise NotImplementedError("Browserbase is not supported yet")

    return None
