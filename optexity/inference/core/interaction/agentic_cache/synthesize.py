"""Stage 2 of the agentic-trace-to-deterministic-nodes pipeline: rule-based synthesis.

No LLM involved. Element -> Playwright ``command`` string reuses the existing
``LocatorExtraction`` heuristic (optexity/inference/core/interaction/utils.py), which already
does this job for the LLM-index-prediction fallback path. Action-name -> ActionNode is a
static lookup table plus field copying.
"""

import logging
import re
from types import SimpleNamespace

from browser_use.dom.views import DOMInteractedElement

from optexity.inference.core.interaction.agentic_cache.models import FilteredAction
from optexity.inference.core.interaction.utils import LocatorExtraction
from optexity.schema.actions.interaction_action import (
    ClickElementAction,
    InputTextAction,
    InteractionAction,
)
from optexity.schema.automation import ActionNode

logger = logging.getLogger(__name__)

_DOWNLOAD_RE = re.compile(r"\bdownload", re.IGNORECASE)


def _wrap(element: DOMInteractedElement) -> SimpleNamespace:
    """Adapt browser-use's DOMInteractedElement to the shape LocatorExtraction expects
    (same information, different field names -- see utils.py's own locator_from_playwright
    for the same pattern against a live Playwright element)."""
    ax_name = element.ax_name or ""
    return SimpleNamespace(
        tag_name=element.node_name,
        attributes=element.attributes or {},
        ax_node=SimpleNamespace(role=None, name=ax_name or None),
        xpath=element.x_path,
        get_meaningful_text_for_llm=lambda: ax_name,
    )


def resolve_command(
    element: DOMInteractedElement,
) -> tuple[str | None, list[tuple[int, str, str]]]:
    """The best Playwright ``command`` string for an interacted element, plus the full
    ranked candidate list (for logging/audit). Returns (None, []) when no viable candidate
    exists at all."""
    candidates = LocatorExtraction.scored_candidates(_wrap(element))
    if not candidates:
        return None, []
    return candidates[0][2], candidates


def _looks_like_download(action: FilteredAction) -> bool:
    """Heuristic only -- browser-use doesn't expose "this click triggered a download"
    directly. The (deferred) self-verification replay is what should actually corroborate
    this; this just keeps a genuinely download-triggering click from silently synthesizing
    into a plain click_element node that optexity's own download-tracking would then miss
    (the exact gap observed on the herokuapp /download automation). Checks the per-action
    result text; the task text itself is checked separately by the caller, since a step's
    own result rarely narrates "download" even when the click plainly triggers one (e.g.
    the LLM's memory just says "Clicked index N" / "Identified the file link")."""
    if action.name != "click":
        return False
    blob = " ".join(
        filter(None, [action.result.long_term_memory, action.result.extracted_content])
    )
    return bool(_DOWNLOAD_RE.search(blob))


def build_click_node(
    action: FilteredAction, force_download: bool = False
) -> tuple[ActionNode | None, list]:
    command, candidates = resolve_command(action.element)
    if command is None:
        return None, candidates
    node = ActionNode(
        type="action_node",
        interaction_action=InteractionAction(
            click_element=ClickElementAction(
                command=command,
                expect_download=force_download or _looks_like_download(action),
                # Forces a clean, catchable AssertLocatorPresenceException on failure
                # instead of silently degrading to the LLM-index-prediction fallback path
                # (see replay.py's docstring for why that matters for verification).
                assert_locator_presence=True,
                skip_prompt=True,
            )
        ),
    )
    return node, candidates


def build_input_node(action: FilteredAction) -> tuple[ActionNode | None, list]:
    command, candidates = resolve_command(action.element)
    if command is None:
        return None, candidates
    node = ActionNode(
        type="action_node",
        interaction_action=InteractionAction(
            input_text=InputTextAction(
                command=command,
                input_text=action.params.get("text", ""),
                fill_or_type="fill",
                assert_locator_presence=True,
                skip_prompt=True,
            )
        ),
    )
    return node, candidates


ACTION_BUILDERS = {
    "click": build_click_node,
    "input": build_input_node,
}


def build_candidate_nodes(
    actions: list[FilteredAction],
    task_text: str = "",
) -> tuple[list[ActionNode], list[FilteredAction]]:
    """Synthesize deterministic ActionNodes for every action the rule table can handle.
    Returns (nodes, skipped) -- skipped actions had no builder or no stable locator
    candidate at all, and are left for manual/LLM synthesis later.

    ``task_text`` is used only to flag ``expect_download`` on the last click action when
    the original agentic_task's own instructions mention "download" (see
    ``_looks_like_download``'s docstring for why the per-action result text alone often
    isn't enough)."""
    task_mentions_download = bool(_DOWNLOAD_RE.search(task_text))
    last_click_index = None
    for i, action in enumerate(actions):
        if action.name == "click":
            last_click_index = i

    nodes: list[ActionNode] = []
    skipped: list[FilteredAction] = []

    for i, action in enumerate(actions):
        builder = ACTION_BUILDERS.get(action.name)
        if builder is build_click_node:
            force_download = task_mentions_download and i == last_click_index
            node, candidates = builder(action, force_download=force_download)
        else:
            node, candidates = builder(action) if builder else (None, [])
        if node is None:
            skipped.append(action)
            logger.debug(
                f"agentic_cache: could not synthesize {action.name!r} action "
                f"deterministically (no stable locator candidates)"
            )
        else:
            logger.debug(
                f"agentic_cache: synthesized {action.name!r} -> {node.interaction_action} "
                f"(chose 1 of {len(candidates)} candidates)"
            )
            nodes.append(node)

    return nodes, skipped
