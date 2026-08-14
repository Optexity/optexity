"""Small, non-executable expression languages used by workflow policies."""

import ast
import operator
from typing import Any

from optexity.guardrails.exceptions import GuardrailViolation

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_COMPARE_OPERATORS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
    ast.In: lambda left, right: left in right,
    ast.NotIn: lambda left, right: left not in right,
}


def safe_evaluate(expression: str, variables: dict[str, Any]) -> Any:
    """Evaluate a Python-style data expression without calls or code access."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise GuardrailViolation("invalid_expression", str(error)) from error

    def evaluate(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in variables or node.id.startswith("_"):
                raise GuardrailViolation(
                    "unknown_expression_name", f"Unknown expression name: {node.id}"
                )
            return variables[node.id]
        if isinstance(node, ast.List):
            return [evaluate(item) for item in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(evaluate(item) for item in node.elts)
        if isinstance(node, ast.Set):
            return {evaluate(item) for item in node.elts}
        if isinstance(node, ast.Dict):
            return {
                evaluate(key): evaluate(value)
                for key, value in zip(node.keys, node.values)
            }
        if isinstance(node, ast.Subscript):
            container = evaluate(node.value)
            index = evaluate(node.slice)
            return container[index]
        if isinstance(node, ast.Attribute):
            value = evaluate(node.value)
            if node.attr.startswith("_") or not isinstance(value, dict):
                raise GuardrailViolation(
                    "unsafe_expression_attribute",
                    f"Only public keys on dictionaries are accessible: {node.attr}",
                )
            return value[node.attr]
        if isinstance(node, ast.UnaryOp):
            operand = evaluate(node.operand)
            if isinstance(node.op, ast.Not):
                return not operand
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return +operand
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                result = evaluate(node.values[0])
                for item in node.values[1:]:
                    if not result:
                        return result
                    result = evaluate(item)
                return result
            if isinstance(node.op, ast.Or):
                result = evaluate(node.values[0])
                for item in node.values[1:]:
                    if result:
                        return result
                    result = evaluate(item)
                return result
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            return _BINARY_OPERATORS[type(node.op)](
                evaluate(node.left), evaluate(node.right)
            )
        if isinstance(node, ast.Compare):
            left = evaluate(node.left)
            for operation, comparator in zip(node.ops, node.comparators):
                function = _COMPARE_OPERATORS.get(type(operation))
                if function is None:
                    break
                right = evaluate(comparator)
                if not function(left, right):
                    return False
                left = right
            else:
                return True
        raise GuardrailViolation(
            "unsafe_expression",
            f"Expression construct is not allowed: {type(node).__name__}",
        )

    return evaluate(tree)


_LOCATOR_METHODS = {
    "and_",
    "filter",
    "frame_locator",
    "get_by_alt_text",
    "get_by_label",
    "get_by_placeholder",
    "get_by_role",
    "get_by_test_id",
    "get_by_text",
    "get_by_title",
    "locator",
    "nth",
    "or_",
}
_LOCATOR_PROPERTIES = {"content_frame", "first", "last"}


def safe_locator_from_command(page: Any, command: str) -> Any:
    """Build a Playwright locator from a whitelisted method-chain grammar."""

    def literal(node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.List):
            return [literal(item) for item in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(literal(item) for item in node.elts)
        if isinstance(node, ast.Dict):
            return {
                literal(key): literal(value)
                for key, value in zip(node.keys, node.values)
            }
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -literal(node.operand)
        raise GuardrailViolation(
            "unsafe_locator_argument",
            f"Locator argument must be literal data: {type(node).__name__}",
        )

    def evaluate(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Name) and node.id == "page":
            return page
        if isinstance(node, ast.Attribute) and node.attr in _LOCATOR_PROPERTIES:
            return getattr(evaluate(node.value), node.attr)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            if method_name not in _LOCATOR_METHODS:
                raise GuardrailViolation(
                    "locator_method_not_allowed",
                    f"Locator method is not allowed: {method_name}",
                )
            receiver = evaluate(node.func.value)
            method = getattr(receiver, method_name)
            args = [literal(argument) for argument in node.args]
            kwargs = {}
            for keyword in node.keywords:
                if keyword.arg is None or keyword.arg.startswith("_"):
                    raise GuardrailViolation(
                        "unsafe_locator_argument",
                        "Expanded locator arguments are forbidden",
                    )
                kwargs[keyword.arg] = literal(keyword.value)
            return method(*args, **kwargs)
        raise GuardrailViolation(
            "unsafe_locator_command",
            f"Locator construct is not allowed: {type(node).__name__}",
        )

    # Stored commands omit the page root. Attaching the fixed root before AST
    # parsing still leaves every accessible method and argument under our
    # evaluator's allowlist; no Python eval or attribute fallback is involved.
    source = f"page.{command}"
    try:
        return evaluate(ast.parse(source, mode="eval"))
    except GuardrailViolation:
        raise
    except Exception as error:
        raise GuardrailViolation("invalid_locator_command", str(error)) from error
