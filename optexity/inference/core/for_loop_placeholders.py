"""Placeholder expansion helpers for for_loop_node iterations."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


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
    3. bare ``{<index_variable_name>}`` → ``<N>`` (last, so it cannot
       corrupt ``index_of(...)`` or ``{var[...]}`` patterns)
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
    node.replace(f"{{{index_variable_name}}}", f"{index}")
    return node


def expand_locator_for_loop_placeholders(
    node,
    locator_command: str,
    index: int,
    index_variable_name: str,
):
    """Bind locator-loop placeholders for one iteration onto a deep-copied node.

    Mirrors variable-loop shape (``{var[index]}`` / bare ``{index}``):
    1. ``{locator[<index_variable_name>]}`` → ``<locator>.nth(<N>)``
    2. ``{index_of(locator)}`` → ``<N>``
    3. bare ``{<index_variable_name>}`` → ``<N>`` (last, so it cannot
       corrupt ``{locator[...]}`` or ``index_of(locator)``)
    """
    node.replace(
        f"{{locator[{index_variable_name}]}}",
        f"{locator_command}.nth({index})",
    )
    node.replace("{index_of(locator)}", f"{index}")
    node.replace(f"{{{index_variable_name}}}", f"{index}")
    return node
