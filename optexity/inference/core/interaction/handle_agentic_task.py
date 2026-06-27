import logging

from browser_use import Agent, BrowserSession, ChatGoogle, Tools

from optexity.inference.core.interaction.handle_computer_use import (
    run_computer_use_agent,
)
from optexity.inference.infra.browser import Browser
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
    # browser_channel="rdp" always drives via Computer Use (pyautogui against
    # the display), whether we RDP into a machine or just opened a URL in a
    # local browser. The agentic_task.backend field is ignored in this case;
    # provider (Claude/Gemini) comes from agentic_task.model.
    if browser.channel == "rdp":
        await run_computer_use_agent(agentic_task_action, task, memory, browser)
        return

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
        llm = ChatGoogle(model="gemini-flash-latest")
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
        logger.debug(f"Starting browser session for agentic task {browser.cdp_url} ")
        await agent.browser_session.start()
        logger.debug(f"Finally running agentic task on browser_use {browser.cdp_url} ")
        await agent.run(max_steps=agentic_task_action.max_steps)
        logger.debug(f"Agentic task completed on browser_use {browser.cdp_url} ")

        agent.stop()
        if agent.browser_session:
            await agent.browser_session.stop()
            await agent.browser_session.reset()

    elif agentic_task_action.backend == "browserbase":
        raise NotImplementedError("Browserbase is not supported yet")
