from __future__ import annotations

import unittest
from typing import Any

from optexity.inference.core.learning_memory.capabilities import prepare_action_node
from optexity.inference.core.learning_memory.models import (
    LearnedStep,
    LearnedStepStrategy,
    LearningPolicy,
    LocatorCandidateMemory,
    LocatorCapability,
    LocatorValidationOutcome,
)
from optexity.schema.automation import ActionNode


class _FakeLocator:
    def __init__(self, probes: list[dict[str, Any]]) -> None:
        self.probes = probes
        self.calls = 0

    async def evaluate_all(self, _script: str, _arguments: dict) -> dict[str, Any]:
        probe = self.probes[min(self.calls, len(self.probes) - 1)]
        self.calls += 1
        return probe


class _FakeBrowser:
    def __init__(self, locator: _FakeLocator) -> None:
        self.locator = locator
        self.resolutions = 0

    async def get_locator_from_command(self, _command: str) -> _FakeLocator:
        self.resolutions += 1
        return self.locator


def _input_node() -> ActionNode:
    return ActionNode.model_validate(
        {
            "type": "action_node",
            "interaction_action": {
                "input_text": {
                    "command": 'locator("input[id=\\"query\\"]")',
                    "input_text": "value",
                    "fill_or_type": "fill",
                }
            },
            "end_sleep_time": 0,
        }
    )


def _learned_input_step() -> LearnedStep:
    return LearnedStep(
        node_index=0,
        source_step_number=1,
        browser_use_action="input",
        optexity_action="input_text",
        strategy=LearnedStepStrategy.LOCATOR,
        capability=LocatorCapability.INPUT,
        candidates=[
            LocatorCandidateMemory(
                command='locator("input[id=\\"query\\"]")',
                locator_kind="css_id",
                original_rank=0,
            )
        ],
    )


def _ready_probe() -> dict[str, Any]:
    return {
        "count": 1,
        "capabilityPassed": True,
        "structurePassed": True,
        "visible": True,
        "enabled": True,
        "editable": True,
        "inputCompatible": True,
    }


class LearningMemoryReadinessTests(unittest.IsolatedAsyncioTestCase):
    async def test_successful_immediate_probe_does_not_wait(self) -> None:
        locator = _FakeLocator([_ready_probe()])
        browser = _FakeBrowser(locator)

        prepared = await prepare_action_node(
            _input_node(),
            _learned_input_step(),
            browser,  # type: ignore[arg-type]
            LearningPolicy(readiness_wait_ms=100),
        )

        self.assertEqual(browser.resolutions, 1)
        self.assertEqual(len(prepared.events), 1)
        self.assertEqual(prepared.events[0].outcome, LocatorValidationOutcome.PASSED)
        self.assertEqual(prepared.events[0].validation_attempt, "immediate")

    async def test_no_match_waits_once_then_retries(self) -> None:
        locator = _FakeLocator(
            [
                {"count": 0, "capabilityPassed": False},
                _ready_probe(),
                _ready_probe(),
            ]
        )
        browser = _FakeBrowser(locator)

        prepared = await prepare_action_node(
            _input_node(),
            _learned_input_step(),
            browser,  # type: ignore[arg-type]
            LearningPolicy(readiness_wait_ms=100),
        )

        self.assertEqual(browser.resolutions, 3)
        self.assertEqual(
            [event.validation_attempt for event in prepared.events],
            ["immediate", "after_readiness_wait"],
        )
        self.assertEqual(
            [event.outcome for event in prepared.events],
            [LocatorValidationOutcome.NO_MATCH, LocatorValidationOutcome.PASSED],
        )
        self.assertTrue(prepared.events[-1].page_ready)


if __name__ == "__main__":
    unittest.main()
