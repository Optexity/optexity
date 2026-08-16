from __future__ import annotations

import unittest
from copy import deepcopy

from browser_use.agent.history_compiler import compile_history_data

from optexity.inference.core.automation_cache.models import ActionCacheConversionError
from optexity.inference.core.learning_memory.session import _require_verified_source_run


def _history_with_judge(verdict: bool | None) -> dict:
    terminal_result: dict[str, object] = {
        "is_done": True,
        "success": True,
    }
    if verdict is not None:
        terminal_result["judgement"] = {"verdict": verdict}
    return {
        "history": [
            {
                "state": {
                    "url": "https://example.test",
                    "title": "Example",
                    "interacted_element": [
                        {
                            "node_name": "INPUT",
                            "attributes": {"id": "query", "type": "text"},
                            "ax_name": "Query",
                            "x_path": "html/body/input",
                        }
                    ],
                },
                "model_output": {
                    "action": [{"input": {"index": 1, "text": "value", "clear": True}}]
                },
                "result": [{"success": True}],
            },
            {
                "state": {
                    "url": "https://example.test",
                    "title": "Example",
                    "interacted_element": [None],
                },
                "model_output": {"action": [{"done": {"text": "Complete"}}]},
                "result": [terminal_result],
            },
        ]
    }


class LearningMemorySourceGateTests(unittest.TestCase):
    def test_explicit_source_judge_pass_allows_discovery(self) -> None:
        cache = compile_history_data(_history_with_judge(True))

        _require_verified_source_run(cache)

    def test_missing_or_negative_source_judge_cannot_create_memory(self) -> None:
        for verdict in (None, False):
            with self.subTest(verdict=verdict):
                history = deepcopy(_history_with_judge(verdict))
                cache = compile_history_data(history)
                with self.assertRaises(ActionCacheConversionError):
                    _require_verified_source_run(cache)


if __name__ == "__main__":
    unittest.main()
