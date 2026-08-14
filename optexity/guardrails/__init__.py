"""Deterministic safety controls for AI-driven browser automation.

The LLM may propose an action, but this package is the authority that decides
whether the action is allowed.  It deliberately contains no model calls.
"""

from optexity.guardrails.exceptions import GuardrailViolation
from optexity.guardrails.models import GuardrailPolicy
from optexity.guardrails.runtime import GuardrailRuntime

__all__ = ["GuardrailPolicy", "GuardrailRuntime", "GuardrailViolation"]
