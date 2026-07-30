"""Standalone checks for for_loop_node placeholder expansion and XOR validation.

Run: python -m optexity.schema.for_loop_placeholder_checks
"""

from __future__ import annotations

from copy import deepcopy

from pydantic import ValidationError

from optexity.inference.core.for_loop_placeholders import (
    expand_for_loop_placeholders,
    expand_locator_for_loop_placeholders,
)
from optexity.schema.automation import ActionNode, ForLoopNode


def _click_command(command: str, prompt: str | None = None) -> ActionNode:
    payload: dict = {
        "type": "action_node",
        "interaction_action": {"click_element": {"command": command}},
    }
    if prompt is not None:
        payload["interaction_action"]["click_element"]["prompt_instructions"] = prompt
    return ActionNode.model_validate(payload)


def check_existing_variable_loop_validates() -> None:
    node = ForLoopNode.model_validate(
        {"type": "for_loop_node", "variable_name": "items", "nodes": []}
    )
    assert node.variable_name == "items"
    assert node.locator is None
    dumped = node.model_dump()
    assert "locator" not in dumped
    assert dumped["variable_name"] == "items"


def check_locator_loop_omits_variable_name_in_dump() -> None:
    node = ForLoopNode.model_validate(
        {
            "type": "for_loop_node",
            "locator": 'get_by_role("row")',
            "nodes": [],
        }
    )
    dumped = node.model_dump()
    assert "variable_name" not in dumped
    assert dumped["locator"] == 'get_by_role("row")'


def check_whitespace_locator_normalized_with_variable() -> None:
    node = ForLoopNode.model_validate(
        {
            "type": "for_loop_node",
            "variable_name": "items",
            "locator": "  ",
            "nodes": [],
        }
    )
    assert node.variable_name == "items"
    assert node.locator is None
    assert "locator" not in node.model_dump()


def check_xor_validation() -> None:
    for payload in (
        {"type": "for_loop_node", "nodes": []},
        {
            "type": "for_loop_node",
            "variable_name": "items",
            "locator": 'get_by_role("row")',
            "nodes": [],
        },
        {"type": "for_loop_node", "variable_name": "  ", "nodes": []},
        {"type": "for_loop_node", "locator": "  ", "nodes": []},
    ):
        try:
            ForLoopNode.model_validate(payload)
        except ValidationError:
            continue
        raise AssertionError(f"expected XOR failure for {payload}")


def check_locator_index_name_reserved() -> None:
    try:
        ForLoopNode.model_validate(
            {
                "type": "for_loop_node",
                "locator": 'get_by_role("row")',
                "index_variable_name": "locator",
                "nodes": [],
            }
        )
    except ValidationError:
        return
    raise AssertionError("index_variable_name='locator' should be rejected")


def check_variable_expand_unchanged() -> None:
    node = _click_command(
        'get_by_text("{items[index]}")',
        prompt="idx {index} of {index_of(items)}",
    )
    expanded = expand_for_loop_placeholders(deepcopy(node), ["items"], 0, "index")
    assert (
        expanded.interaction_action.click_element.command == 'get_by_text("{items[0]}")'
    )
    assert expanded.interaction_action.click_element.prompt_instructions == "idx 0 of 0"


def check_locator_expand_array_style() -> None:
    node = _click_command(
        '{locator[index]}.get_by_role("button", name="Edit")',
        prompt="row {index} / {index_of(locator)}",
    )
    expanded = expand_locator_for_loop_placeholders(
        deepcopy(node), 'get_by_role("row")', 2, "index"
    )
    assert (
        expanded.interaction_action.click_element.command
        == 'get_by_role("row").nth(2).get_by_role("button", name="Edit")'
    )
    assert expanded.interaction_action.click_element.prompt_instructions == "row 2 / 2"


def check_nested_locator_index_names() -> None:
    node = _click_command(
        "{locator[row]} then {locator[cell]} idxs {row}/{cell}",
    )
    after_outer = expand_locator_for_loop_placeholders(
        deepcopy(node), 'get_by_role("row")', 1, "row"
    )
    after_inner = expand_locator_for_loop_placeholders(
        after_outer, 'get_by_role("cell")', 3, "cell"
    )
    assert (
        after_inner.interaction_action.click_element.command
        == 'get_by_role("row").nth(1) then get_by_role("cell").nth(3) idxs 1/3'
    )


def check_old_format_locator_loop_migration() -> None:
    parent = ForLoopNode.model_validate(
        {
            "type": "for_loop_node",
            "variable_name": "items",
            "nodes": [
                {
                    "locator": 'get_by_role("row")',
                    "nodes": [
                        {
                            "type": "action_node",
                            "interaction_action": {
                                "click_element": {"command": "{locator[index]}"}
                            },
                        }
                    ],
                }
            ],
        }
    )
    assert parent.nodes[0].type == "for_loop_node"
    assert parent.nodes[0].locator == 'get_by_role("row")'


def main() -> None:
    checks = [
        check_existing_variable_loop_validates,
        check_locator_loop_omits_variable_name_in_dump,
        check_whitespace_locator_normalized_with_variable,
        check_xor_validation,
        check_locator_index_name_reserved,
        check_variable_expand_unchanged,
        check_locator_expand_array_style,
        check_nested_locator_index_names,
        check_old_format_locator_loop_migration,
    ]
    for check in checks:
        check()
        print(f"ok  {check.__name__}")
    print(f"passed {len(checks)} checks")


if __name__ == "__main__":
    main()
