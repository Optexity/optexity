from pathlib import Path
from typing import Any

from optexity.guardrails.context import get_guardrail_runtime

INTERACTION_FIELDS = {
    "click_element": "click",
    "input_text": "input",
    "select_option": "select",
    "check": "check",
    "uncheck": "uncheck",
    "hover": "hover",
    "download_url_as_pdf": "download",
    "scroll": "scroll",
    "upload_file": "upload",
    "go_to_url": "navigate",
    "go_back": "go_back",
    "switch_tab": "switch_tab",
    "close_current_tab": "close_tab",
    "close_all_but_last_tab": "close_tab",
    "close_tabs_until": "close_tab",
    "agentic_task": "agentic_task",
    "close_overlay_popup": "agentic_task",
    "key_press": "key_press",
}


def interaction_action_name(interaction_action: Any) -> tuple[str, Any]:
    for field, action_name in INTERACTION_FIELDS.items():
        value = getattr(interaction_action, field, None)
        if value is not None:
            return action_name, value
    return "unknown", interaction_action


async def authorize_interaction(interaction_action: Any, browser: Any) -> None:
    runtime = get_guardrail_runtime()
    if runtime is None or not runtime.policy.enabled:
        return
    action_name, action = interaction_action_name(interaction_action)
    current_url = await browser.get_current_page_url()
    target_url = getattr(action, "url", None)
    data = None
    if action_name == "input":
        data = getattr(action, "input_text", None)
        target_url = current_url
    elif action_name == "upload":
        target_url = current_url
        file_path = getattr(action, "file_path", None)
        if file_path:
            runtime.authorize_upload_path(file_path)
        data = file_path or getattr(action, "file_url", None)
    elif action_name == "click" and getattr(action, "expect_download", False):
        runtime.consume("downloads")
    elif action_name == "select" and getattr(action, "expect_download", False):
        runtime.consume("downloads")
    runtime.authorize_action(
        action_name,
        source="workflow",
        current_url=current_url,
        target_url=target_url,
        data=data,
    )


async def authorize_action_node(action_node: Any, browser: Any) -> None:
    runtime = get_guardrail_runtime()
    if runtime is None or not runtime.policy.enabled:
        return
    mapping = {
        "python_script_action": "python_script",
        "powershell_action": "powershell",
        "assertion_action": "assertion",
        "sleep_action": "sleep",
        # CAPTCHA and existing human-in-loop behavior are deliberately outside
        # the AI-action guardrail project and remain unchanged.
    }
    for field, action_name in mapping.items():
        if getattr(action_node, field, None) is not None:
            runtime.authorize_action(
                action_name,
                source="workflow",
                current_url=await browser.get_current_page_url(),
            )
            return
    extraction = getattr(action_node, "extraction_action", None)
    if extraction is not None:
        if getattr(extraction, "python_script", None) is not None:
            name = "python_script"
        elif getattr(extraction, "api_call", None) is not None:
            # The API handler authorizes the concrete URL, payload, redirect
            # behavior, and call budget immediately before network I/O.
            return
        else:
            name = "extract"
        runtime.authorize_action(
            name,
            source="workflow",
            current_url=await browser.get_current_page_url(),
        )
        return
    misc = getattr(action_node, "misc_action", None)
    if misc is not None:
        if getattr(misc, "llm_query", None) is not None:
            name = "llm_query"
        elif getattr(misc, "set_variable", None) is not None:
            name = "set_variable"
        else:
            name = "extract"
        runtime.authorize_action(
            name,
            source="workflow",
            current_url=await browser.get_current_page_url(),
        )


async def authorize_private_node(private_node: Any, browser: Any) -> None:
    """Authorize an opaque plugin handler before any plugin code executes."""
    runtime = get_guardrail_runtime()
    if runtime is None or not runtime.policy.enabled:
        return
    runtime.authorize_action(
        "private_node",
        source=f"plugin:{private_node.handler}",
        current_url=await browser.get_current_page_url(),
    )


def authorize_download_destination(path: str | Path) -> None:
    runtime = get_guardrail_runtime()
    if runtime is None or not runtime.policy.enabled:
        return
    # Download outputs must stay inside the task directory, which is registered
    # as an allowed root by GuardrailRuntime.from_task.
    resolved = Path(path).resolve()
    allowed = any(
        resolved == root or root in resolved.parents
        for root in runtime.allowed_upload_roots
    )
    runtime._enforce(
        denied=not allowed,
        code="download_path_allow" if allowed else "download_path_not_allowed",
        message=f"Download destination is outside the task directory: {resolved}",
        event_type="filesystem",
        action="download",
        source="workflow",
        details={"path": str(resolved)},
    )


def authorize_external_request(
    url: str,
    *,
    action: str,
    data: Any = None,
    source: str = "workflow",
) -> None:
    runtime = get_guardrail_runtime()
    if runtime is None or not runtime.policy.enabled:
        return
    runtime.authorize_action(
        action,
        source=source,
        target_url=url,
        data=data,
    )
