"""Unit tests for nested for_loop_node and index_variable_name."""

import unittest
from copy import deepcopy

from pydantic import ValidationError

from optexity.schema.automation import ActionNode, ForLoopNode


def _expand(node, variable_names, index, index_variable_name):
    """Mirror of runtime placeholder expansion (kept local to avoid heavy imports)."""
    for variable_name in variable_names:
        node.replace(
            f"{{{variable_name}[{index_variable_name}]}}",
            f"{{{variable_name}[{index}]}}",
        )
    node.replace(f"{{index_of({variable_names[0]})}}", f"{index}")
    node.replace(f"{{{index_variable_name}}}", f"{index}")
    return node


_CLICK = {
    "type": "action_node",
    "interaction_action": {
        "click_element": {
            "command": 'get_by_text("{items[item_i]}")',
            "prompt_instructions": (
                "Page {page_i} ({pages[page_i]}), "
                "item {item_i} ({items[item_i]}) idx={index_of(items)}"
            ),
        }
    },
}


class TestForLoopNodeSchema(unittest.TestCase):
    def test_default_index_variable_name(self):
        node = ForLoopNode.model_validate(
            {
                "type": "for_loop_node",
                "variable_name": "order_ids",
                "nodes": [
                    {
                        "type": "action_node",
                        "interaction_action": {
                            "go_to_url": {"url": "https://example.com"}
                        },
                    }
                ],
            }
        )
        self.assertEqual(node.index_variable_name, "index")

    def test_nested_for_loop_allowed(self):
        node = ForLoopNode.model_validate(
            {
                "type": "for_loop_node",
                "variable_name": "pages",
                "index_variable_name": "page_i",
                "nodes": [
                    {
                        "type": "for_loop_node",
                        "variable_name": "items",
                        "index_variable_name": "item_i",
                        "nodes": [_CLICK],
                    }
                ],
            }
        )
        self.assertIsInstance(node.nodes[0], ForLoopNode)
        self.assertEqual(node.index_variable_name, "page_i")
        self.assertEqual(node.nodes[0].index_variable_name, "item_i")

    def test_invalid_index_variable_name_rejected(self):
        with self.assertRaises(ValidationError):
            ForLoopNode.model_validate(
                {
                    "type": "for_loop_node",
                    "variable_name": "x",
                    "index_variable_name": "bad-name",
                    "nodes": [
                        {
                            "type": "action_node",
                            "interaction_action": {
                                "go_to_url": {"url": "https://example.com"}
                            },
                        }
                    ],
                }
            )

    def test_index_variable_name_cannot_match_loop_variable(self):
        with self.assertRaises(ValidationError):
            ForLoopNode.model_validate(
                {
                    "type": "for_loop_node",
                    "variable_name": "pages",
                    "index_variable_name": "pages",
                    "nodes": [
                        {
                            "type": "action_node",
                            "interaction_action": {
                                "go_to_url": {"url": "https://example.com"}
                            },
                        }
                    ],
                }
            )

    def test_index_variable_name_cannot_be_index_of(self):
        with self.assertRaises(ValidationError):
            ForLoopNode.model_validate(
                {
                    "type": "for_loop_node",
                    "variable_name": "pages",
                    "index_variable_name": "index_of",
                    "nodes": [
                        {
                            "type": "action_node",
                            "interaction_action": {
                                "go_to_url": {"url": "https://example.com"}
                            },
                        }
                    ],
                }
            )


class TestForLoopPlaceholderExpansion(unittest.TestCase):
    def test_nested_index_names_expand_independently(self):
        action = ActionNode.model_validate(_CLICK)
        after_outer = _expand(deepcopy(action), ["pages"], 1, "page_i")
        after_inner = _expand(after_outer, ["items"], 2, "item_i")
        click = after_inner.interaction_action.click_element
        self.assertEqual(click.command, 'get_by_text("{items[2]}")')
        self.assertEqual(
            click.prompt_instructions,
            "Page 1 ({pages[1]}), item 2 ({items[2]}) idx=2",
        )

    def test_default_index_still_works(self):
        action = ActionNode.model_validate(
            {
                "type": "action_node",
                "interaction_action": {
                    "click_element": {
                        "command": 'get_by_text("{order_ids[index]}")',
                        "prompt_instructions": "n={index} v={order_ids[index]}",
                    }
                },
            }
        )
        expanded = _expand(deepcopy(action), ["order_ids"], 0, "index")
        click = expanded.interaction_action.click_element
        self.assertEqual(click.command, 'get_by_text("{order_ids[0]}")')
        self.assertEqual(click.prompt_instructions, "n=0 v={order_ids[0]}")


if __name__ == "__main__":
    unittest.main()
