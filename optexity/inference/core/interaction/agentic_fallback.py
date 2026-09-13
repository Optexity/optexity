import logging
from functools import lru_cache
from importlib import resources

import aiofiles

from optexity.exceptions import ElementNotFoundInAxtreeException
from optexity.inference.core.interaction.handle_agentic_task import handle_agentic_task
from optexity.inference.core.interaction.utils import LocatorExtraction
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
# How much of the run log (optexity.log, the same file we ship to S3) to feed the
# agent. We tail it so a long run doesn't blow up the prompt; bump if needed.
FALLBACK_LOG_TAIL_CHARS = 20000


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
    if node.dynamic_form_mapping_action is not None:
        return "dynamic form mapping"
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


async def _read_run_log_tail(task: Task) -> str:
    """Read the tail of the task's runtime log (optexity.log).

    This is the same log we persist to S3; the recent lines capture what the
    deterministic run was doing right up to the -1 failure, which is the most
    useful debugging context for the fallback agent.
    """
    try:
        async with aiofiles.open(
            task.log_file_path, "r", encoding="utf-8", errors="replace"
        ) as f:
            content = await f.read()
    except FileNotFoundError:
        return "(run log not available)"
    except Exception as e:
        logger.error(f"Failed to read run log for agentic fallback: {e}")
        return "(run log not available)"

    if not content:
        return "(run log empty)"
    if len(content) > FALLBACK_LOG_TAIL_CHARS:
        return (
            f"...(truncated; showing last {FALLBACK_LOG_TAIL_CHARS} chars)...\n"
            + content[-FALLBACK_LOG_TAIL_CHARS:]
        )
    return content


def _render_input_parameters(task: Task) -> str:
    """Render the automation's (non-secret) input parameters for the agent.

    Shown in ``{key[index]} = "value"`` form so the agent can map a step's
    placeholder to its real value and fill an empty/missing field. Only
    ``input_parameters`` are exposed — ``secure_parameters`` (which resolve to
    real secrets) are deliberately never sent to the fallback agent.
    """
    params = task.input_parameters or {}
    lines: list[str] = []
    for key, values in params.items():
        if not isinstance(values, list):
            continue
        for i, value in enumerate(values):
            lines.append(f'  - {{{key}[{i}]}} = "{value}"')
    return "\n".join(lines) if lines else "(no input parameters provided)"


def _expected_browser_use_action_names(interaction_action: InteractionAction) -> set:
    """Which browser_use ``ActionModel`` key(s) would count as "doing" the
    original failed *interaction_action*, so a successful fallback run's
    actions can be filtered down to ones that actually match — a fallback
    agent can wander (dismiss a popup, poke around) before doing the one
    thing it was asked to do, and only an action of the right type is
    eligible at all, before the error/proximity gates below narrow further.
    ``check``/``uncheck`` both surface as a plain "click" from browser_use's
    perspective (see the ``## TODO`` in ``handle_check.py`` — check/uncheck
    aren't distinct browser_use actions today).
    """
    if (
        interaction_action.click_element
        or interaction_action.check
        or interaction_action.uncheck
    ):
        return {"click"}
    if interaction_action.input_text:
        return {"input"}
    if interaction_action.select_option:
        return {"select_dropdown"}
    if interaction_action.hover:
        return {"hover"}
    if interaction_action.upload_file:
        return {"upload_file"}
    return set()


def _pick_agentic_interacted_node(history, interacted_nodes: dict, interaction_action):
    """Attribute the fallback run to the one action that actually made
    progress on the original goal, not just any action of a matching type.

    Layered gates, in order:
    1. The run must have succeeded overall (``history.is_successful()``) — a
       failed/timed-out run's actions are exploratory by construction and
       shouldn't feed locator evidence at all.
    2. Action type must match what the original failed step was trying to do.
    3. The action's own execution result must be error-free
       (``ActionResult.error is None``) — a same-type action that itself
       raised is definitely not "the" interaction, retried or not.
    4. Among survivors, prefer the one *nearest* the run's terminal
       success — walking backward from the end — over "last in the whole
       trajectory": a hit-and-try run can do the right thing first and then
       take an unrelated later action of the same type (e.g. re-verifying,
       or a subsequent step) that would otherwise wrongly win under a naive
       "last occurrence" rule.

    Returns ``None`` if nothing survives all four gates — callers must not
    force-fit a mismatched/failed action just because the run overall
    succeeded.
    """
    if history is None:
        return None
    try:
        if history.is_successful() is not True:
            return None
    except Exception:
        return None

    expected = _expected_browser_use_action_names(interaction_action)
    if not expected:
        return None

    for step_idx, item in enumerate(reversed(history.history)):
        step_number = (
            item.metadata.step_number
            if item.metadata is not None
            else len(history.history) - 1 - step_idx
        )
        if item.model_output is None:
            continue
        actions = list(enumerate(item.model_output.action))
        results = item.result or []
        for action_index, action in reversed(actions):
            try:
                action_name = next(iter(action.model_dump(exclude_unset=True).keys()))
            except StopIteration:
                continue
            if action_name not in expected:
                continue
            if action_index >= len(results) or results[action_index].error is not None:
                continue
            node = interacted_nodes.get((step_number, action_index))
            if node is not None:
                # Opportunistic context for a human reading the debugger, not a
                # gate: this is the model's own self-critique carried on this
                # step's current_state, which technically evaluates whatever
                # preceded this action rather than this action itself (the
                # step that would evaluate *this* action hasn't necessarily
                # run yet). Never treated as authoritative — the hard gates
                # above (is_successful()/ActionResult.error) already decided
                # this action counts.
                eval_text = None
                if item.model_output.current_state is not None:
                    eval_text = item.model_output.current_state.evaluation_previous_goal
                return node, eval_text
    return None


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
    if memory.browser_states:
        memory.browser_states[-1].resolution_tier = "agentic"

    goal = _describe_goal(interaction_action, error.command or "(no goal provided)")
    workflow_window = _build_workflow_window(task, interaction_action)

    error_logs = str(error.message)
    if getattr(error, "original_error", None) is not None:
        error_logs += f"\nUnderlying error: {error.original_error}"

    run_log = await _read_run_log_tail(task)
    error_logs += f"\n\n--- Recent run log (optexity.log) ---\n{run_log}"

    try:
        current_url = await browser.get_current_page_url() or "(unknown)"
    except Exception:
        current_url = "(unknown)"

    prompt = (
        _load_fallback_prompt_template()
        .replace("<<GOAL>>", str(goal))
        .replace("<<WORKFLOW_WINDOW>>", workflow_window)
        .replace("<<INPUT_PARAMETERS>>", _render_input_parameters(task))
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
    history, interacted_nodes = await handle_agentic_task(
        fallback_action, task, memory, browser
    )

    # Best-effort: attribute the run to the action that actually made
    # progress and record its ranked/verified locator candidates. Guarded so
    # any failure here can never affect the fallback's own success/failure
    # result (`history` is still returned unconditionally below) — this is
    # pure logging/telemetry, same posture as the other tiers.
    try:
        picked = _pick_agentic_interacted_node(
            history, interacted_nodes, interaction_action
        )
        if picked is not None and memory.browser_states:
            node, eval_text = picked
            page = await browser.get_current_page()
            if page is not None:
                # No single trailing ".click()"/".fill(...)" call the way the
                # command/axtree tiers have — the fallback agent could have
                # done any browser_use action type, and `interaction_action`
                # only tells us the *target* type, not exactly which
                # parameters browser_use used. Leaving `method` blank still
                # records which element/locator resolved the step; only the
                # copy-paste convenience of a trailing call is lost.
                candidates = await LocatorExtraction.candidates_from_tree_node(
                    node, "", page
                )
                if candidates:
                    memory.browser_states[-1].locator_candidates = candidates
                    logger.info(
                        f"Agentic fallback locator: {candidates[0]['locator']} "
                        f"(verified={candidates[0]['verified']}, "
                        f"model self-eval: {eval_text!r})"
                    )
    except Exception as e:
        logger.debug(
            f"Failed to record agentic fallback locator: {type(e).__name__}: {e}"
        )

    return history
