"""Optional entry points for the cross-run procedural-memory feature.

The normal Optexity runtime must remain importable with the published
``optexity-browser-use`` dependency, which does not yet contain the take-home
history compiler.  Keep the compiler-dependent session module behind the
feature flag and import it only when learning memory is actually enabled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from optexity.schema.automation import ActionNode
from optexity.schema.memory import Memory
from optexity.schema.task import Task
from optexity.utils.settings import settings

if TYPE_CHECKING:
    from optexity.inference.core.learning_memory.session import LearningMemorySession

ACTION_CACHE_FILENAME = "browser_use_action_cache.json"
LEARNING_SESSION_STATE_KEY = "learning_memory_session"


def create_learning_session(task: Task) -> LearningMemorySession | None:
    """Create the optional learning session without affecting normal imports."""

    if not settings.LEARNING_MEMORY_ENABLED:
        return None
    try:
        from optexity.inference.core.learning_memory.session import (
            create_learning_session as create_session,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "browser_use.agent.history_compiler":
            raise RuntimeError(
                "Learning memory requires the paired Browser Use checkout that "
                "provides browser_use.agent.history_compiler"
            ) from exc
        raise
    return create_session(task)


def get_learning_session(memory: Memory) -> LearningMemorySession | None:
    """Return a previously created session without importing optional code."""

    return memory.state.get(LEARNING_SESSION_STATE_KEY)


def is_cacheable_agentic_node(node: ActionNode) -> bool:
    """Return whether a node is an eligible Browser Use discovery boundary."""

    interaction = node.interaction_action
    return bool(
        interaction is not None
        and interaction.agentic_task is not None
        and interaction.agentic_task.backend == "browser_use"
    )


def __getattr__(name: str) -> Any:
    """Preserve the public exception/session imports for explicit consumers."""

    if name in {"LearningMemorySession", "LearningReplayError"}:
        from optexity.inference.core.learning_memory import session

        return getattr(session, name)
    raise AttributeError(name)


__all__ = (
    "ACTION_CACHE_FILENAME",
    "LEARNING_SESSION_STATE_KEY",
    "create_learning_session",
    "get_learning_session",
    "is_cacheable_agentic_node",
)
