"""Glue for the agentic-cache pipeline: filter -> synthesize -> persist -> verify, and the
cache-hit check that lets handle_agentic_task.py skip constructing browser_use.Agent
entirely. Called from handle_agentic_task.py."""

import logging

from browser_use.agent.views import AgentHistoryList

from optexity.inference.core.interaction.agentic_cache.filter import filter_agentic_trace
from optexity.inference.core.interaction.agentic_cache.replay import replay_cached_nodes
from optexity.inference.core.interaction.agentic_cache.store import (
    CachedBlockStore,
    compute_automation_key,
)
from optexity.inference.core.interaction.agentic_cache.synthesize import build_candidate_nodes
from optexity.inference.infra.browser import Browser
from optexity.schema.actions.interaction_action import AgenticTask
from optexity.schema.memory import Memory
from optexity.schema.task import Task

logger = logging.getLogger(__name__)

_store: CachedBlockStore | None = None


def _get_store() -> CachedBlockStore:
    global _store
    if _store is None:
        _store = CachedBlockStore()
    return _store


async def try_serve_from_cache(
    agentic_task_action: AgenticTask,
    task: Task,
    memory: Memory,
    browser: Browser,
) -> bool:
    """Replay the latest verified cached block for this automation_key, if one exists.
    Returns True when it was served successfully (caller should skip constructing
    Agent(...) entirely). Returns False on a cache miss OR a failed replay -- in the
    failure case, re-navigates to the automation's start URL first so a subsequent real
    agent run starts from a clean page rather than wherever the partial replay left off."""
    if task.automation is None:
        return False

    automation_key = compute_automation_key(task.automation.url, agentic_task_action.task)
    block = _get_store().latest_verified(automation_key)
    if block is None:
        return False

    success, error = await replay_cached_nodes(block.nodes, task, memory, browser)
    if success:
        logger.info(
            f"agentic_cache: cache hit -- replayed {len(block.nodes)} node(s), zero LLM "
            f"calls (automation_key={automation_key}, row id={block.id})"
        )
        return True

    logger.warning(
        f"agentic_cache: cached block replay failed ({error}); re-navigating and "
        f"falling back to agentic run (automation_key={automation_key}, row id={block.id})"
    )
    try:
        await browser.go_to_url(task.automation.url)
    except Exception as e:
        logger.warning(f"agentic_cache: re-navigation after failed replay also failed: {e}")
    return False


async def capture_and_verify(
    history: AgentHistoryList,
    task: Task,
    memory: Memory,
    browser: Browser,
    url: str,
    task_text: str,
    source_run_id: str | None = None,
) -> None:
    """Filter + synthesize a completed browser-use run, persist the result as a new
    (unverified) cached block, then eagerly verify it -- fresh navigation back to `url`,
    replay for real, and only then mark it verified. Best-effort by design -- callers
    should wrap this in their own try/except so a bug here can never break a live agentic
    run."""
    usage = getattr(history, "usage", None)
    if usage is not None:
        logger.info(
            f"agentic_cache: source run LLM usage -- {usage.total_tokens} tokens "
            f"(prompt={usage.total_prompt_tokens}, completion={usage.total_completion_tokens}, "
            f"cached={usage.total_prompt_cached_tokens}) cost=${usage.total_cost:.5f} "
            f"across {usage.entry_count} call(s)"
        )

    filtered = filter_agentic_trace(history)
    nodes, skipped = build_candidate_nodes(filtered, task_text=task_text)

    if not nodes:
        logger.info(
            f"agentic_cache: no deterministic nodes synthesized for task={task_text!r} "
            f"({len(skipped)} action(s) skipped)"
        )
        return

    automation_key = compute_automation_key(url, task_text)
    store = _get_store()
    row_id = store.save(
        automation_key=automation_key,
        nodes=nodes,
        source_run_id=source_run_id,
        verified=False,
    )
    logger.info(
        f"agentic_cache: captured {len(nodes)} node(s), skipped {len(skipped)} action(s) "
        f"for automation_key={automation_key} (row id={row_id}, source_run_id={source_run_id})"
    )

    await browser.go_to_url(url)  # fresh start -- never continue from wherever the agent left off
    success, error = await replay_cached_nodes(nodes, task, memory, browser)
    if success:
        store.mark_verified(row_id)
        logger.info(f"agentic_cache: verified row id={row_id}")
    else:
        logger.warning(
            f"agentic_cache: verification failed for row id={row_id}: {error} "
            f"(left unverified; will retry on the next capture)"
        )
