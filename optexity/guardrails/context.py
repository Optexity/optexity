from contextvars import ContextVar, Token
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optexity.guardrails.runtime import GuardrailRuntime

_runtime: ContextVar["GuardrailRuntime | None"] = ContextVar(
    "optexity_guardrail_runtime", default=None
)


def get_guardrail_runtime() -> "GuardrailRuntime | None":
    return _runtime.get()


def set_guardrail_runtime(runtime: "GuardrailRuntime") -> Token:
    return _runtime.set(runtime)


def reset_guardrail_runtime(token: Token) -> None:
    _runtime.reset(token)
