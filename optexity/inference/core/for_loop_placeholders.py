"""Placeholder expansion helpers for for_loop_node iterations."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _bind_index(node, index: int, index_variable_name: str):
    """Bind bare ``{<index_variable_name>}`` → ``<N>``.

    Always applied last, so it cannot corrupt the source-specific patterns
    (``{var[...]}``, ``{locator[...]}``, ``index_of(...)``) bound before it.
    """
    node.replace(f"{{{index_variable_name}}}", f"{index}")
    return node


def expand_for_loop_placeholders(
    node,
    variable_names: list[str],
    index: int,
    index_variable_name: str,
):
    """Bind loop placeholders for one iteration onto a deep-copied node.

    Replacement order matters:
    1. ``{var[<index_variable_name>]}`` → ``{var[<N>]}``
    2. ``{index_of(primary)}`` → ``<N>``
    3. bare ``{<index_variable_name>}`` → ``<N>``
    """
    for variable_name in variable_names:
        try:
            node.replace(
                f"{{{variable_name}[{index_variable_name}]}}",
                f"{{{variable_name}[{index}]}}",
            )
        except Exception as e:
            logger.error(
                f"Error replacing variable {variable_name} in for loop node: {e}"
            )
            continue

    node.replace(f"{{index_of({variable_names[0]})}}", f"{index}")
    return _bind_index(node, index, index_variable_name)


def expand_locator_for_loop_placeholders(
    node,
    locator_command: str,
    index: int,
    index_variable_name: str,
):
    """Bind locator-loop placeholders for one iteration onto a deep-copied node.

    Mirrors variable-loop shape (``{var[index]}`` / bare ``{index}``):
    1. ``{locator[<index_variable_name>]}`` → ``<locator>.nth(<N>)``
    2. bare ``{<index_variable_name>}`` → ``<N>``

    There is deliberately no ``{index_of(locator)}``: unlike
    ``{index_of(<variable>)}`` it carries no per-loop name, so in nested locator
    loops the outer loop binds the inner loop's occurrences. Use the bare
    ``{<index_variable_name>}`` form, which is scoped per level.
    """
    node.replace(
        f"{{locator[{index_variable_name}]}}",
        f"{locator_command}.nth({index})",
    )
    return _bind_index(node, index, index_variable_name)


def expand_iteration_placeholders(
    node,
    index: int,
    index_variable_name: str,
    variable_names: list[str] | None = None,
    locator_command: str | None = None,
):
    """Dispatch one iteration's bindings to the right expander.

    Exactly one of ``variable_names`` / ``locator_command`` is expected, mirroring
    the for_loop_node schema's variable_name / locator XOR.
    """
    if locator_command is not None:
        return expand_locator_for_loop_placeholders(
            node, locator_command, index, index_variable_name
        )
    assert variable_names is not None, "expected variable_names or locator_command"
    return expand_for_loop_placeholders(
        node, variable_names, index, index_variable_name
    )
