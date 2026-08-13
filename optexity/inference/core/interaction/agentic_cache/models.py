from dataclasses import dataclass
from typing import Any

from browser_use.dom.views import DOMInteractedElement

from optexity.schema.automation import ActionNode


@dataclass
class FilteredAction:
    """A single surviving browser-use action after noise-filtering, still carrying the
    live-run context (interacted element, result) needed for synthesis."""

    name: str  # e.g. "click", "input" -- the registered browser-use action/tool name
    params: dict[str, Any]  # the action's own param dict, e.g. {"index": 17, "text": "myname", "clear": True}
    element: DOMInteractedElement | None
    result: Any  # browser_use.agent.views.ActionResult for this action


@dataclass
class CachedBlock:
    """One row of the cached_blocks table."""

    id: int
    automation_key: str
    nodes: list[ActionNode]
    verified: bool
    version: int
    created_at: str
    last_verified_at: str | None
    source_run_id: str | None
