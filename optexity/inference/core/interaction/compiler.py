import json
import logging
import os
import re
from typing import Any, Dict
import litellm

from optexity.schema.automation import (
    ActionNode,
    Automation,
    InteractionAction,
)
from optexity.schema.actions.interaction_action import (
    ClickElementAction,
    InputTextAction,
)
from optexity.utils.settings import settings

logger = logging.getLogger(__name__)

COMPILER_PROMPT = """You are an expert Playwright automation engineer.
Analyze this agent interaction trace and extract deterministic actions.

Supported Actions:
1. "input":
   - "command": Playwright locator command (e.g., locator('input[name="04fullname"]').first)
   - "text": Text value entered
2. "click":
   - "command": Playwright locator command (e.g., locator('button[type="submit"]').first)

Rules:
- Prefer stable attributes: name > id > placeholder > aria-label > xpath.
- Always append `.first` to locators.
- Return ONLY a valid JSON list of objects.

Output format:
[
  {"type": "input", "command": "locator('input[name=\"04fullname\"]').first", "text": "myname"},
  {"type": "click", "command": "locator('button[type=\"submit\"]').first"}
]
"""


def _parameterize_text(text: str, input_params: Dict[str, Any]) -> str:
    """Replaces concrete input strings with their parameter template keys."""
    if not text or not input_params:
        return text

    parameterized = text
    for key, val in input_params.items():
        if isinstance(val, list):
            for idx, item in enumerate(val):
                str_item = str(item)
                if str_item and str_item in parameterized:
                    parameterized = parameterized.replace(str_item, f"{{{key}[{idx}]}}")
        elif isinstance(val, str) and val in parameterized:
            parameterized = parameterized.replace(val, f"{{{key}}}")
    return parameterized


async def compile_trajectory_to_automation(
    original_automation: Automation,
    history: Any,
    task_input_parameters: Dict[str, Any],
) -> Automation:
    trace_summary = []
    for step in getattr(history, "history", []):
        if not step.model_output or not step.model_output.action:
            continue
        actions = step.model_output.action
        elements = (step.state.interacted_element if step.state else []) or []
        for i, action in enumerate(actions):
            el = elements[i] if i < len(elements) else None
            action_dict = action.model_dump(exclude_none=True) if hasattr(action, "model_dump") else {}
            el_dict = {
                "attributes": getattr(el, "attributes", {}) if el else {},
                "xpath": getattr(el, "xpath", None) or getattr(el, "x_path", None) if el else None,
            }
            trace_summary.append({"action": action_dict, "element": el_dict})

    if not trace_summary:
        logger.warning("No interaction steps found in history to compile.")
        return original_automation

    api_key = getattr(settings, "ANTHROPIC_API_KEY", None) or os.environ.get("LLM_MODEL_API_KEY")

    response = await litellm.acompletion(
        model="anthropic/claude-sonnet-5",
        messages=[
            {"role": "system", "content": COMPILER_PROMPT},
            {"role": "user", "content": json.dumps(trace_summary, indent=2)},
        ],
        api_key=api_key,
    )

    raw_content = response.choices[0].message.content.strip()
    match = re.search(r"\[.*\]", raw_content, re.DOTALL)
    if not match:
        raise ValueError(f"Compiler did not return a valid JSON list: {raw_content}")

    extracted_actions = json.loads(match.group(0))

    # 1. Build concrete Playwright ActionNodes
    synthesized_nodes = []
    for item in extracted_actions:
        action_type = item.get("type", "input")
        command = item.get("command", "")

        # Clean locator if model prepends page.
        if command.startswith("page."):
            command = command[len("page."):]

        if action_type == "input":
            hydrated_text = _parameterize_text(str(item.get("text", "")), task_input_parameters)
            node = ActionNode(
                type="action_node",
                interaction_action=InteractionAction(
                    input_text=InputTextAction(
                        command=command,
                        input_text=hydrated_text,
                        fill_or_type="fill",
                        click_before_input=True,
                        press_enter=False,
                    )
                ),
            )
            synthesized_nodes.append(node)

        elif action_type == "click":
            node = ActionNode(
                type="action_node",
                interaction_action=InteractionAction(
                    click_element=ClickElementAction(
                        command=command,
                    )
                ),
            )
            synthesized_nodes.append(node)

    # 2. Splice synthesized nodes into the original DAG in place of the agentic task node
    compiled_automation = original_automation.model_copy(deep=True)
    new_nodes = []

    for node in compiled_automation.nodes:
        if (
            node.type == "action_node"
            and node.interaction_action
            and node.interaction_action.agentic_task
        ):
            # Replace the exploratory agent node with the deterministic compiled nodes
            new_nodes.extend(synthesized_nodes)
        else:
            new_nodes.append(node)

    compiled_automation.nodes = new_nodes
    return compiled_automation