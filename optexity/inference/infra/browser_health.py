"""Browser session health checks and dedicated-browser restart signaling."""

import asyncio
import logging
import os
from pathlib import Path

from browser_use.browser.views import BrowserStateSummary

from optexity.inference.infra.browser import Browser
from optexity.schema.memory import BrowserState, Memory
from optexity.schema.task import Task

logger = logging.getLogger(__name__)

BROWSER_STATE_SUMMARY_TIMEOUT_SECONDS = 35.0

DRIVER_CLOSED_MARKERS = (
    "connection closed",
    "target closed",
    "browser closed",
    "no close frame",
    "has been closed",
    "target crashed",
    "browser context",
    "context closed",
)


def is_driver_closed_error(e: BaseException) -> bool:
    msg = str(e).lower()
    return any(m in msg for m in DRIVER_CLOSED_MARKERS)


def is_browser_session_poisoned_error(e: BaseException) -> bool:
    if is_driver_closed_error(e):
        return True
    return isinstance(e, (asyncio.TimeoutError, TimeoutError))


def get_child_process_id_from_env() -> int | None:
    val = os.environ.get("CHILD_PROCESS_ID")
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        return None


def get_browser_restart_flag_path(child_process_id: int) -> Path:
    return Path(f"/tmp/optexity_browser_restart_{child_process_id}")


def request_browser_restart(child_process_id: int, reason: str) -> None:
    path = get_browser_restart_flag_path(child_process_id)
    path.write_text(reason[:2000])
    logger.warning(
        "Requested dedicated browser restart (child_process_id=%s): %s",
        child_process_id,
        reason[:500],
    )


def consume_browser_restart_request(child_process_id: int) -> str | None:
    path = get_browser_restart_flag_path(child_process_id)
    if not path.is_file():
        return None
    try:
        reason = path.read_text()
    finally:
        path.unlink(missing_ok=True)
    return reason


def update_memory_browser_state_from_summary(
    browser_state_summary: BrowserStateSummary,
    memory: Memory,
    task: Task,
) -> None:
    memory.browser_states[-1] = BrowserState(
        url=browser_state_summary.url,
        screenshot=browser_state_summary.screenshot,
        title=browser_state_summary.title,
        axtree=browser_state_summary.dom_state.llm_representation(
            remove_empty_nodes=task.automation.remove_empty_nodes_in_axtree
        ),
    )


async def fetch_browser_state_for_classifier(
    browser: Browser,
    memory: Memory,
    task: Task,
    *,
    include_full_page: bool = False,
) -> BrowserStateSummary | None:
    """Fetch full axtree + screenshot; return None and signal restart if session is poisoned."""
    child_process_id = get_child_process_id_from_env()
    try:
        browser_state_summary = await asyncio.wait_for(
            browser.get_browser_state_summary(include_full_page=include_full_page),
            timeout=BROWSER_STATE_SUMMARY_TIMEOUT_SECONDS,
        )
        update_memory_browser_state_from_summary(browser_state_summary, memory, task)
        return browser_state_summary
    except Exception as e:
        logger.warning(
            "Failed to fetch browser state for classifier (include_full_page=%s): %s",
            include_full_page,
            e,
            exc_info=True,
        )
        if child_process_id is not None and is_browser_session_poisoned_error(e):
            request_browser_restart(child_process_id, str(e))
        return None
