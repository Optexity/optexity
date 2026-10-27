"""One structured log entry per LLM attempt.

Always written to the `optexity.llm_calls` logger; with LLM_CLOUD_LOGGING=true
also sent to Google Cloud Logging (log name `llm-calls`), where log-based metrics
and the LLM dashboard read it. Prompt and response text only with
LLM_LOG_CONTENT=true.
"""

import json
import logging

from optexity.utils.llm_settings import llm_settings

from .config import PROJECT

_logger = logging.getLogger("optexity.llm_calls")
_cloud_handler: logging.Handler | None = None


def _attach_cloud_handler() -> None:
    global _cloud_handler
    if _cloud_handler is not None or not llm_settings.LLM_CLOUD_LOGGING:
        return
    from google.cloud.logging import Client
    from google.cloud.logging.handlers import CloudLoggingHandler
    from google.cloud.logging_v2.resource import Resource

    # Background transport: entries are batched and sent off the calling thread.
    handler = CloudLoggingHandler(
        Client(project=PROJECT),
        name="llm-calls",
        resource=Resource(type="global", labels={}),
    )
    _logger.addHandler(handler)
    _cloud_handler = handler


def emit(entry: dict) -> None:
    try:
        _attach_cloud_handler()
    except Exception as e:  # logging must never break an LLM call
        _logger.warning(
            "Cloud Logging unavailable (%s); LLM logs go to stdout only.", e
        )
        llm_settings.LLM_CLOUD_LOGGING = False
    ok = entry["status"] == "ok"
    level = logging.INFO if ok else logging.ERROR if entry["final"] else logging.WARNING
    attempt = f" attempt {entry['attempt']}" if entry["attempt"] > 1 else ""
    outcome = f"${entry['cost_usd']:.4f}" if ok else entry["error_class"]
    summary = (
        f"{'✓' if ok else '✗'} {entry['service']}/{entry['feature']} "
        f"{entry['model_used']}{attempt} {outcome} {entry['latency_ms'] / 1000:.1f}s"
    )
    # In Cloud Logging the fields travel as json_fields; stdout carries them as JSON.
    message = summary if _cloud_handler else f"{summary} {json.dumps(entry)}"
    _logger.log(level, message, extra={"json_fields": entry})


# Together under Cloud Logging's 256 KB entry limit.
_PROMPT_CHARS = 120_000
_RESPONSE_CHARS = 60_000


def _clip(text: str, limit: int) -> str:
    """Keeps the start (instructions) and the end (latest turn), cuts the middle."""
    if len(text) <= limit:
        return text
    head = limit // 4
    cut = len(text) - limit
    return f"{text[:head]}\n[... {cut:,} characters cut ...]\n{text[head - limit:]}"


def _content(content) -> str:
    if isinstance(content, str):
        return content
    out = []
    for part in content or []:
        if part.get("type") == "text":
            out.append(part.get("text", ""))
            continue
        # Images and files are never logged, only their type and size.
        source = part.get("image_url") or part.get("file") or {}
        url = source.get("url") or source.get("file_data") or ""
        mime = url[5:].split(";", 1)[0] if url.startswith("data:") else part.get("type")
        out.append(f"[{mime} {len(url) * 3 // 4096}KB]")
    return "\n".join(out)


def content_fields(messages: list[dict], reply_text: str | None) -> dict:
    prompt = "\n\n".join(
        f"[{m.get('role')}]\n{_content(m.get('content'))}" for m in messages
    )
    return {
        "prompt": _clip(prompt, _PROMPT_CHARS),
        "response": None if reply_text is None else _clip(reply_text, _RESPONSE_CHARS),
    }


def flush_llm_logs() -> None:
    """Send buffered entries now (call before a short-lived process exits)."""
    if _cloud_handler is not None:
        _cloud_handler.flush()
