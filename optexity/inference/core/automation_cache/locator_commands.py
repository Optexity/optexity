from __future__ import annotations

import ast

from optexity.inference.core.automation_cache.models import (
    ActionCacheConversionError,
    UnconvertedStep,
)

_SINGLE_STRING_ARGUMENT_METHODS = frozenset(
    {"get_by_label", "get_by_placeholder", "get_by_test_id", "locator"}
)


def validate_playwright_locator_command(
    command: str,
    *,
    source_step_number: int,
    browser_use_action: str,
) -> None:
    """Reject executable cache strings outside the compiler's locator grammar."""

    problem = UnconvertedStep(
        source_step_number=source_step_number,
        browser_use_action=browser_use_action,
        explanation="The cached locator command is outside the safe Playwright locator grammar.",
    )
    try:
        expression = ast.parse(f"page.{command}", mode="eval")
    except SyntaxError as exc:
        raise ActionCacheConversionError("Invalid cached locator", [problem]) from exc

    call = expression.body
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
        raise ActionCacheConversionError("Invalid cached locator", [problem])
    if not isinstance(call.func.value, ast.Name) or call.func.value.id != "page":
        raise ActionCacheConversionError("Invalid cached locator", [problem])
    method = call.func.attr
    if method in _SINGLE_STRING_ARGUMENT_METHODS:
        valid_signature = (
            len(call.args) == 1
            and not call.keywords
            and _is_string_literal(call.args[0])
        )
    elif method == "get_by_role":
        valid_signature = (
            len(call.args) == 1
            and _is_string_literal(call.args[0])
            and len(call.keywords) == 1
            and call.keywords[0].arg == "name"
            and _is_string_literal(call.keywords[0].value)
        )
    else:
        valid_signature = False
    if not valid_signature:
        raise ActionCacheConversionError("Invalid cached locator", [problem])


def _is_string_literal(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)
