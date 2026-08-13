"""Shared replay primitive used by both the cache-hit path (skip the agent) and
verification (confirm a freshly-captured block before trusting it)."""

import logging

from optexity.inference.infra.browser import Browser
from optexity.schema.automation import ActionNode
from optexity.schema.memory import Memory
from optexity.schema.task import Task

logger = logging.getLogger(__name__)


async def replay_cached_nodes(
    nodes: list[ActionNode],
    task: Task,
    memory: Memory,
    browser: Browser,
) -> tuple[bool, str | None]:
    """Execute a list of cached ActionNodes for real via the normal deterministic
    dispatcher, and report honestly whether it worked.

    retries_left=1 is deliberate: run_interaction_action's own error handler
    (handle_assert_locator_presence_error in run_interaction.py) only invokes its LLM-based
    error classifier when retries_left > 1 -- at retries_left <= 1 it raises the original
    exception directly. Combined with assert_locator_presence=True on synthesized nodes
    (see synthesize.py), a locator failure here is a clean, catchable exception rather than
    a silent no-op -- the exact gap found in manual testing on the herokuapp-login replay.
    """
    # Deferred import: run_interaction.py imports handle_agentic_task at module load time,
    # which (via pipeline.py) would otherwise import this module -> a circular import.
    from optexity.inference.core.run_interaction import run_interaction_action

    for node in nodes:
        if node.interaction_action is None:
            continue
        try:
            await run_interaction_action(node.interaction_action, task, memory, browser, 1)
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    return True, None
