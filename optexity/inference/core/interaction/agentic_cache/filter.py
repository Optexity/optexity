"""Stage 1 of the agentic-trace-to-deterministic-nodes pipeline: rule-based noise
filtering. No LLM involved -- see the plan in
C:\\Users\\prate\\.claude\\plans\\i-m-working-on-a-jiggly-acorn.md for the reasoning.
"""

import logging

from browser_use.agent.views import AgentHistoryList

from optexity.inference.core.interaction.agentic_cache.models import FilteredAction

logger = logging.getLogger(__name__)

# Only these browser-use action names get synthesized into deterministic nodes right now.
# Everything else is dropped rather than mis-synthesized; extend this table (and
# synthesize.ACTION_BUILDERS) as new action types are observed in real traces.
INDEX_ACTIONS = {"click", "input"}

# Read-only / verification-only actions: no optexity interaction_action equivalent.
# Dropped by default; promoting to assertion_action/extraction_action is a future extension.
READ_ONLY_ACTIONS = {"evaluate", "extract"}

# Terminal bookkeeping, not a page action.
DROP_ACTIONS = {"done"}


def _log_dropped(name: str, reason: str, **extra) -> None:
    logger.debug(f"agentic_cache: dropped action {name!r} (reason={reason}) {extra or ''}")


def filter_agentic_trace(history: AgentHistoryList) -> list[FilteredAction]:
    """Flatten a completed browser-use run into the surviving atomic actions worth
    synthesizing into deterministic nodes, in original execution order."""
    kept: list[FilteredAction] = []

    for step in history.history:
        if step.model_output is None:
            continue

        actions = step.model_output.action
        interacted = step.state.interacted_element or [None] * len(actions)
        results = step.result

        for action, element, result in zip(actions, interacted, results):
            # action.model_dump() is used rather than attribute access because when more
            # than one action type is registered, browser-use wraps each action in an
            # ActionModelUnion(RootModel[...]) -- the real per-action field only exists on
            # its `.root`, not on the wrapper itself. model_dump() is the one API that
            # works the same way regardless of whether the model is wrapped.
            dumped = action.model_dump(exclude_none=True)
            if not dumped:
                continue
            name = next(iter(dumped))
            params = dumped[name]  # plain dict, e.g. {"index": 17, "text": "myname", "clear": True}

            if result.error is not None:
                _log_dropped(name, "action_error", error=result.error)
                continue
            if name in DROP_ACTIONS:
                continue
            if name in READ_ONLY_ACTIONS:
                _log_dropped(name, "read_only_no_interaction_equivalent")
                continue
            if name in INDEX_ACTIONS and element is None:
                _log_dropped(name, "no_resolved_element")
                continue
            if name not in INDEX_ACTIONS:
                _log_dropped(name, "action_type_not_yet_synthesizable")
                continue

            kept.append(
                FilteredAction(name=name, params=params, element=element, result=result)
            )

    return _collapse_by_element_identity(kept)


def _identity_key(action: FilteredAction) -> tuple[str, object] | None:
    el = action.element
    if el is None:
        return None
    if el.stable_hash is not None:
        return ("hash", el.stable_hash)
    if el.x_path:
        return ("xpath", el.x_path)
    return None


def _collapse_by_element_identity(actions: list[FilteredAction]) -> list[FilteredAction]:
    """The same element acted on more than once (e.g. a corrected input) collapses to only
    its LAST occurrence, kept at that occurrence's original position so replay order still
    matches page flow. Actions with no reliable identity are never collapsed."""
    last_index: dict[tuple[str, object], int] = {}
    for i, action in enumerate(actions):
        key = _identity_key(action)
        if key is not None:
            last_index[key] = i

    keep_indices = {i for i, a in enumerate(actions) if _identity_key(a) is None}
    keep_indices |= set(last_index.values())

    dropped = len(actions) - len(keep_indices)
    if dropped:
        logger.debug(f"agentic_cache: collapsed {dropped} superseded repeat action(s)")

    return [a for i, a in enumerate(actions) if i in keep_indices]
