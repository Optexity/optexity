import logging
from functools import lru_cache
from importlib import resources

from optexity.exceptions import ElementNotFoundInAxtreeException
from optexity.inference.core.interaction.handle_agentic_task import handle_agentic_task
from optexity.inference.infra.browser import Browser
from optexity.schema.actions.interaction_action import AgenticTask, InteractionAction
from optexity.schema.automation import ActionNode, ForLoopNode, IfElseNode
from optexity.schema.memory import Memory
from optexity.schema.task import Task

logger = logging.getLogger(__name__)

# Guardrails for the fallback agent: keep it short and scoped to a single step.
FALLBACK_MAX_STEPS = 12
# How many steps before/after the current one to include for workflow context.
WINDOW_RADIUS = 2


@lru_cache(maxsize=1)
def _load_fallback_prompt_template() -> str:
    return (
        resources.files("optexity.prompts")
        .joinpath("agentic_fallback.md")
        .read_text(encoding="utf-8")
    )


def _summarize_action_node(node: ActionNode) -> str | None:
    """Return a short human-readable summary of an action node for context."""
    ia = node.interaction_action
    if ia is not None:
        for name in [
            "click_element",
            "input_text",
            "select_option",
            "check",
            "uncheck",
            "hover",
            "upload_file",
            "key_press",
            "scroll",
            "go_to_url",
            "download_url_as_pdf",
            "go_back",
        ]:
            sub = getattr(ia, name, None)
            if sub is not None:
                desc = (
                    getattr(sub, "prompt_instructions", "")
                    or getattr(sub, "command", "")
                    or getattr(sub, "url", "")
                    or ""
                )
                label = name.replace("_", " ")
                return f"{label}: {desc}".strip().rstrip(":").strip()
        return "interaction action"
    if node.extraction_action is not None:
        return "extract data"
    if node.assertion_action is not None:
        return "assertion check"
    if node.captcha_action is not None:
        return "solve captcha"
    if node.human_in_loop_action is not None:
        return "human-in-loop step"
    if node.python_script_action is not None:
        return "python script"
    if node.sleep_action is not None:
        return "wait"
    return None


def _describe_goal(interaction_action: InteractionAction, fallback_command: str) -> str:
    """Build a complete, self-contained goal for the fallback agent.

    error.command only carries the locator description (prompt_instructions). For
    input/select steps the *value* to enter lives in a separate field, so we must
    splice it in or the agent won't know what to type/select.
    """
    ia = interaction_action

    if ia.click_element is not None:
        base = ia.click_element.prompt_instructions or fallback_command
        return f"Click: {base}"
    if ia.input_text is not None:
        base = ia.input_text.prompt_instructions or fallback_command
        value = ia.input_text.input_text
        if value:
            return f'Type the value "{value}" into: {base}'
        return f"Type into: {base}"
    if ia.select_option is not None:
        base = ia.select_option.prompt_instructions or fallback_command
        values = ia.select_option.select_values
        if values:
            return f"Select option(s) {values} in: {base}"
        return f"Select an option in: {base}"
    if ia.check is not None:
        return f"Check (tick) the checkbox: {ia.check.prompt_instructions or fallback_command}"
    if ia.uncheck is not None:
        return f"Uncheck the checkbox: {ia.uncheck.prompt_instructions or fallback_command}"
    if ia.hover is not None:
        return f"Hover over: {ia.hover.prompt_instructions or fallback_command}"
    if ia.upload_file is not None:
        return f"Upload a file to: {ia.upload_file.prompt_instructions or fallback_command}"

    return fallback_command


def _flatten_action_nodes(nodes, out: list) -> None:
    """Statically flatten the automation tree into a linear list of ActionNodes.

    Both branches of if/else and the body of for-loops are included so the agent
    sees the surrounding intent regardless of runtime branching.
    """
    for node in nodes:
        if isinstance(node, ActionNode):
            out.append(node)
        elif isinstance(node, ForLoopNode):
            _flatten_action_nodes(node.nodes, out)
        elif isinstance(node, IfElseNode):
            _flatten_action_nodes(node.if_nodes, out)
            _flatten_action_nodes(node.else_nodes, out)


def _describe_node_for_window(node: ActionNode) -> str:
    """Value-bearing description of a node for the workflow window.

    Reuses the goal builder (which splices in input/select values) so the agent
    can verify a previous step actually took effect, falling back to a short
    summary for non-interaction nodes.
    """
    ia = node.interaction_action
    if ia is not None:
        desc = _describe_goal(ia, "")
        if desc and desc.strip():
            return desc
    return _summarize_action_node(node) or "step"


def _build_workflow_window(task: Task, interaction_action: InteractionAction) -> str:
    """Build a small window (prev + current + next steps) around the failing step.

    Previous steps are rendered with their full value-bearing goals so the agent
    can check whether each already-run prerequisite actually landed on the page.
    The current step is marked; next steps are kept as light context only.

    The current step is located by object identity of its interaction_action.
    This resolves for top-level nodes; loop-expanded nodes are deep-copied at
    runtime and won't match, in which case we degrade gracefully.
    """
    try:
        flat: list[ActionNode] = []
        _flatten_action_nodes(task.automation.nodes, flat)

        current_idx = None
        for i, node in enumerate(flat):
            if node.interaction_action is interaction_action:
                current_idx = i
                break

        if current_idx is None:
            return "(surrounding workflow steps unavailable)"

        start = max(0, current_idx - WINDOW_RADIUS)
        end = min(len(flat), current_idx + WINDOW_RADIUS + 1)
        lines = []
        for i in range(start, end):
            if i < current_idx:
                desc = _describe_node_for_window(flat[i])
                lines.append(f"  [already ran] step {i}: {desc}")
            elif i == current_idx:
                desc = _describe_node_for_window(flat[i])
                lines.append(f"  >> CURRENT (failed locator) >> step {i}: {desc}")
            else:
                summary = _summarize_action_node(flat[i]) or "step"
                lines.append(f"  [do NOT do — context only] step {i}: {summary}")
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"Failed to build workflow window for agentic fallback: {e}")
        return "(surrounding workflow steps unavailable)"


async def run_axtree_fallback_agent(
    interaction_action: InteractionAction,
    error: ElementNotFoundInAxtreeException,
    task: Task,
    memory: Memory,
    browser: Browser,
):
    """Hand a single failed (axtree -1) step to a general browser_use agent.

    The agent is given the step goal, a window of surrounding workflow steps, and
    the failure logs, then asked to accomplish only this step (dismissing any
    popup/interstitial that gets in the way).
    """
    goal = _describe_goal(interaction_action, error.command or "(no goal provided)")
    workflow_window = _build_workflow_window(task, interaction_action)

    error_logs = str(error.message)
    if getattr(error, "original_error", None) is not None:
        error_logs += f"\nUnderlying error: {error.original_error}"

    try:
        current_url = await browser.get_current_page_url() or "(unknown)"
    except Exception:
        current_url = "(unknown)"

    prompt = (
        _load_fallback_prompt_template()
        .replace("<<GOAL>>", str(goal))
        .replace("<<WORKFLOW_WINDOW>>", workflow_window)
        .replace("<<ERROR_LOGS>>", error_logs)
        .replace("<<CURRENT_URL>>", str(current_url))
    )

    fallback_action = AgenticTask(
        task=prompt,
        max_steps=FALLBACK_MAX_STEPS,
        backend="browser_use",
        use_vision=True,
        keep_alive=True,
    )

    logger.debug(
        f"Running agentic fallback for goal '{goal}' on {current_url} "
        f"(max_steps={FALLBACK_MAX_STEPS})"
    )
    return await handle_agentic_task(fallback_action, task, memory, browser)
