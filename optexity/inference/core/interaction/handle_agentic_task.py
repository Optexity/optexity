import inspect
import json
import logging
import os

from browser_use import Agent, BrowserSession, ChatGoogle, Tools

from optexity.inference.infra.browser import Browser
from optexity.learning.step_cache import build_optexity_automation
from optexity.schema.actions.interaction_action import (
    AgenticTask,
    CloseOverlayPopupAction,
)
from optexity.schema.memory import Memory
from optexity.schema.task import Task

logger = logging.getLogger(__name__)


def _agent_supports_step_cache() -> bool:
    try:
        return "step_cache_path" in inspect.signature(Agent.__init__).parameters
    except (TypeError, ValueError):
        return False


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
        llm = ChatGoogle(model="gemini-flash-latest")
        browser_session = BrowserSession(
            cdp_url=browser.cdp_url, keep_alive=agentic_task_action.keep_alive
        )

        step_directory = (
            task.logs_directory / f"step_{str(memory.automation_state.step_index)}"
        )
        step_directory.mkdir(parents=True, exist_ok=True)
        step_cache_path = step_directory / "step_cache.json"
        optexity_automation_path = step_directory / "cached_automation.json"

        agent_kwargs: dict = {
            "task": agentic_task_action.task,
            "llm": llm,
            "browser_session": browser_session,
            "use_vision": agentic_task_action.use_vision,
            "tools": tools,
            "calculate_cost": True,
            "save_conversation_path": step_directory,
        }
        os.environ["BROWSER_USE_STEP_CACHE_PATH"] = str(step_cache_path)
        if _agent_supports_step_cache():
            agent_kwargs["step_cache_path"] = step_cache_path
            agent_kwargs["step_cache_url"] = task.automation.url
        else:
            logger.warning(
                "browser-use fork without step_cache_path support detected; "
                "set BROWSER_USE_STEP_CACHE_PATH and install your browser-use fork "
                "with: pip install -e /path/to/browser-use"
            )

        agent = Agent(**agent_kwargs)
        logger.debug(f"Starting browser session for agentic task {browser.cdp_url} ")
        await agent.browser_session.start()
        logger.debug(f"Finally running agentic task on browser_use {browser.cdp_url} ")
        await agent.run(max_steps=agentic_task_action.max_steps)
        logger.debug(f"Agentic task completed on browser_use {browser.cdp_url} ")

        if step_cache_path.exists():
            with open(step_cache_path, encoding="utf-8") as cache_file:
                cache_data = json.load(cache_file)
            automation = build_optexity_automation(
                cache_data, url=task.automation.url
            )
            with open(optexity_automation_path, "w", encoding="utf-8") as out_file:
                json.dump(automation, out_file, indent=2)
            logger.info(
                "Saved cached Optexity automation to %s (%d node(s))",
                optexity_automation_path,
                len(automation.get("nodes", [])),
            )
        else:
            logger.warning(
                "No step_cache.json at %s — install browser-use fork: "
                "pip install -e /path/to/browser-use",
                step_cache_path,
            )

        agent.stop()
        if agent.browser_session:
            await agent.browser_session.stop()
            await agent.browser_session.reset()

    elif agentic_task_action.backend == "browserbase":
        raise NotImplementedError("Browserbase is not supported yet")
