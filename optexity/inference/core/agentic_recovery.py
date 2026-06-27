"""Agentic recovery for RDP automations.

When an interaction step on an RDP / computer-vision automation fails screen
validation (``KeywordNotFoundOnScreenException``), ``run_automation`` runs a
short Computer Use agent to unblock the screen before retrying the step once.

This module owns the construction of the natural-language task prompt for that
agent: it describes the failed step plus its immediate neighbours so the agent
understands what the automation was doing and where it is headed.
"""

import logging
from copy import deepcopy
from typing import cast

from optexity.inference.core.variable_resolver import (
    resolve_api_variables_in_node,
    resolve_dynamic_indices_in_node,
)
from optexity.schema.automation import ActionNode, ForLoopNode, IfElseNode
from optexity.schema.memory import Memory
from optexity.schema.task import Task

logger = logging.getLogger(__name__)

# Max agentic recovery attempts allowed per task. Bounds LLM cost and damage
# if the recovery itself starts misfiring.
MAX_AGENTIC_RECOVERIES_PER_TASK = 3

# Steps the agent is allowed before we give up and retry the failed action once.
AGENTIC_RECOVERY_MAX_STEPS = 6

# How many surrounding nodes to describe on each side of the failed step when
# building recovery context.
AGENTIC_RECOVERY_CONTEXT_WINDOW = 2


def _describe_node(node) -> str:
    """Short description of a node for recovery context.

    Reads only target descriptions (``prompt_instructions`` / ``keyword``),
    never typed values or selected option contents. Callers resolve a node's
    variables before describing it, so any ``{var[...]}`` / ``{secure[...]}``
    placeholders inside ``prompt_instructions`` will already be substituted.
    """
    if isinstance(node, IfElseNode):
        return "a conditional (if/else) branch"
    if isinstance(node, ForLoopNode):
        return "a loop over a list of values"
    if not isinstance(node, ActionNode):
        return "an unknown step"

    ia = node.interaction_action
    if ia is not None:
        if ia.click_element is not None:
            desc = f'click "{ia.click_element.prompt_instructions}"'
            if ia.click_element.keyword:
                desc += f' (on-screen text "{ia.click_element.keyword}")'
            return desc
        if ia.input_text is not None:
            desc = f'type into "{ia.input_text.prompt_instructions}"'
            if getattr(ia.input_text, "keyword", None):
                desc += f' (on-screen text "{ia.input_text.keyword}")'
            return desc
        target = getattr(
            getattr(ia, "select_option", None), "prompt_instructions", None
        )
        if ia.select_option is not None:
            return f'select an option in "{target}"' if target else "select an option"
        if getattr(ia, "hover", None) is not None:
            t = getattr(ia.hover, "prompt_instructions", None)
            return f'hover over "{t}"' if t else "hover over an element"
        if ia.scroll is not None:
            return "scroll the screen"
        if ia.key_press is not None:
            return "press a keyboard key"
        if getattr(ia, "go_to_url", None) is not None:
            return "navigate to a URL"
        return "an interaction step"
    if node.extraction_action is not None:
        return "extract data from the screen"
    if node.assertion_action is not None:
        return "assert a condition on the screen"
    if node.captcha_action is not None:
        return "solve a captcha"
    if node.python_script_action is not None:
        return "run a script"
    if node.powershell_action is not None:
        return "run a PowerShell command"
    if node.sleep_action is not None:
        return "wait"
    return "a step"


async def _resolved_copy(node, task: Task, memory: Memory):
    """Deepcopy ``node`` and resolve its variables the same way
    ``run_action_node`` does, so its description shows real values rather than
    ``{placeholder}`` text. Non-action nodes need no resolution. Best-effort:
    returns the original node if resolution fails."""
    if not isinstance(node, ActionNode):
        return node
    try:
        clone = deepcopy(node)
        resolve_dynamic_indices_in_node(clone, memory.variables.generated_variables)
        # replace_variables is annotated narrower than its runtime contract (it
        # str()-coerces any value), so cast to satisfy the type checker without
        # touching the shared schema signature.
        await clone.replace_variables(cast(dict, task.input_parameters))
        await clone.replace_variables(
            cast(dict, task.secure_parameters), task.workspace_id, task.api_key
        )
        await clone.replace_variables(memory.variables.generated_variables)
        resolve_api_variables_in_node(clone, memory.variables.generated_variables)
        return clone
    except Exception as e:
        logger.debug(f"recovery: could not resolve neighbor node variables: {e}")
        return node


async def _neighbor_descriptions(
    siblings,
    node_index,
    task: Task,
    memory: Memory,
    window: int = AGENTIC_RECOVERY_CONTEXT_WINDOW,
) -> tuple[list[str], list[str]]:
    """Describe up to ``window`` nodes before and after the failed node within
    its sibling list, resolving each node's variables first. Returns
    (previous_descriptions, next_descriptions)."""
    if not siblings or node_index is None:
        return [], []
    prev_nodes = siblings[max(0, node_index - window) : node_index]
    next_nodes = siblings[node_index + 1 : node_index + 1 + window]
    prev = [_describe_node(await _resolved_copy(n, task, memory)) for n in prev_nodes]
    nxt = [_describe_node(await _resolved_copy(n, task, memory)) for n in next_nodes]
    return prev, nxt


def _compose_prompt(intent: str, prev_descs: list[str], next_descs: list[str]) -> str:
    """Assemble the recovery task prompt with failed-step intent and the
    surrounding steps for goal awareness."""
    lines = [f"An automated step just failed. The step was trying to: {intent}."]
    if prev_descs:
        lines.append(
            "Just before this, the automation did: " + "; then ".join(prev_descs) + "."
        )
    if next_descs:
        lines.append(
            "Once this step succeeds, the automation will go on to: "
            + "; then ".join(next_descs)
            + "."
        )
    lines.append(
        "\nLook at the current screen and work out why the failed step could not "
        "be completed. Common causes: a popup / alert / cookie or consent banner / "
        "modal dialog covering the target, the page or app still loading, or an "
        "unexpected screen. Take the shortest sequence of actions that returns the "
        "app to a state where the failed step can succeed on retry.\n\n"
        "Hard rules:\n"
        "1. Do NOT perform the failed step yourself, and do NOT run any of the "
        "later steps. The automation will retry the failed step automatically right "
        "after you finish — your only job is to clear whatever is preventing it.\n"
        "2. NEVER click 'Delete', 'Remove', or 'Discard permanently'. Deletion is "
        "not allowed under any circumstance.\n"
        "3. For Save dialogs ('Save', \"Don't Save\", 'Cancel', 'Discard', 'No', "
        "'Yes'), read the dialog and choose the option that best returns the app to "
        "a usable state. Don't pick blindly.\n"
        "4. Stay on the current screen: do not open new windows, navigate to other "
        "screens, or kill processes.\n"
        "5. If the screen looks like it is still loading, you may wait briefly.\n"
        "6. If nothing appears to be blocking the failed step, do NOT click "
        "anything — stop and reply with a one-line summary of what you see."
    )
    return "\n".join(lines)


async def build_recovery_prompt(
    failed_node: ActionNode,
    siblings: list | None,
    node_index: int | None,
    task: Task,
    memory: Memory,
) -> str:
    """Build the Computer Use recovery task prompt for ``failed_node``.

    ``failed_node`` has already had its variables resolved in place by
    ``run_action_node`` before dispatch, so it is described as-is; the
    neighbouring nodes are resolved on a copy before being described.
    """
    prev_descs, next_descs = await _neighbor_descriptions(
        siblings, node_index, task, memory
    )
    return _compose_prompt(_describe_node(failed_node), prev_descs, next_descs)
