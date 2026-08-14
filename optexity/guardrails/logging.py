import logging

from optexity.guardrails.context import get_guardrail_runtime


class GuardrailRedactionFilter(logging.Filter):
    """Redact task secrets before a record reaches persistent handlers."""

    def filter(self, record: logging.LogRecord) -> bool:
        runtime = get_guardrail_runtime()
        if runtime is None or not runtime.policy.enabled:
            return True
        message = record.getMessage()
        record.msg = runtime.redact_text(message)
        record.args = ()
        return True
