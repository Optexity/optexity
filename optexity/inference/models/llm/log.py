"""One structured log entry per LLM attempt.

Always written to the `optexity.llm_calls` logger; with LLM_CLOUD_LOGGING=true
also sent to Google Cloud Logging (log name `llm-calls`), where log-based metrics
and the LLM dashboard read it. Never contains prompt or response text.
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
        Client(project=PROJECT), name="llm-calls", resource=Resource(type="global", labels={})
    )
    # The fields travel as json_fields; a fixed message stops the JSON line being stored twice.
    handler.setFormatter(logging.Formatter("llm_call", validate=False))
    _logger.addHandler(handler)
    _cloud_handler = handler


def emit(entry: dict) -> None:
    try:
        _attach_cloud_handler()
    except Exception as e:  # logging must never break an LLM call
        _logger.warning("Cloud Logging unavailable (%s); LLM logs go to stdout only.", e)
        llm_settings.LLM_CLOUD_LOGGING = False
    _logger.info(json.dumps(entry), extra={"json_fields": entry})


def flush_llm_logs() -> None:
    """Send buffered entries now (call before a short-lived process exits)."""
    if _cloud_handler is not None:
        _cloud_handler.flush()
