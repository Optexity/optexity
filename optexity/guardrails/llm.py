from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from optexity.guardrails.context import get_guardrail_runtime


def prepare_llm_text(text: str, *, source: str = "llm_prompt") -> str:
    runtime = get_guardrail_runtime()
    if runtime is None or not runtime.policy.enabled:
        return text
    return runtime.inspect_untrusted_text(text, source=source)


def prepare_llm_call(
    prompt: str,
    system_instruction: str | None,
    screenshot: str | None = None,
    pdf_url: Any = None,
) -> tuple[str, str | None, str | None, Any]:
    runtime = get_guardrail_runtime()
    if runtime is None or not runtime.policy.enabled:
        return prompt, system_instruction, screenshot, pdf_url
    runtime.consume("llm_calls")
    prompt = runtime.inspect_untrusted_text(prompt, source="llm_prompt")
    if system_instruction is not None:
        # System instructions are trusted; only redact secrets accidentally
        # interpolated into them. Prompt-injection scanning is intentionally not run.
        system_instruction = runtime.redact_text(system_instruction)
    if not runtime.policy.data_protection.allow_screenshots_to_llm:
        if screenshot is not None:
            runtime.audit.emit(
                decision="audit",
                code="llm_screenshot_removed",
                event_type="data_protection",
                source="llm_gateway",
            )
        screenshot = None
    if not runtime.policy.data_protection.allow_files_to_llm:
        if pdf_url is not None:
            runtime.audit.emit(
                decision="audit",
                code="llm_file_removed",
                event_type="data_protection",
                source="llm_gateway",
            )
        pdf_url = None
    elif pdf_url is not None:
        parsed = urlparse(str(pdf_url))
        if parsed.scheme in {"http", "https"}:
            runtime.authorize_action(
                "extract",
                source="llm_file_gateway",
                target_url=str(pdf_url),
            )
        else:
            runtime.authorize_file_path(
                Path(pdf_url), action="extract", source="llm_file_gateway"
            )
    return prompt, system_instruction, screenshot, pdf_url


def sanitize_serialized_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    runtime = get_guardrail_runtime()
    if runtime is None or not runtime.policy.enabled:
        return messages

    def clean(value: Any, *, trusted: bool = False) -> Any:
        if isinstance(value, str):
            if value.startswith("data:image/"):
                if not runtime.policy.data_protection.allow_screenshots_to_llm:
                    return "[IMAGE_BLOCKED_BY_GUARDRAIL]"
                return value
            if value.startswith("data:application/"):
                if not runtime.policy.data_protection.allow_files_to_llm:
                    return "[FILE_BLOCKED_BY_GUARDRAIL]"
                return value
            if trusted:
                return runtime.redact_text(value)
            return runtime.inspect_untrusted_text(value, source="agent_message")
        if isinstance(value, list):
            return [clean(item, trusted=trusted) for item in value]
        if isinstance(value, dict):
            return {key: clean(item, trusted=trusted) for key, item in value.items()}
        return value

    runtime.consume("llm_calls")
    sanitized = []
    for message in messages:
        # System/developer messages are application-owned. They still receive
        # secret redaction, but prompt-injection scanning applies only to
        # untrusted/user/browser content.
        trusted = message.get("role") in {"system", "developer"}
        sanitized.append(clean(message, trusted=trusted))
    return sanitized
