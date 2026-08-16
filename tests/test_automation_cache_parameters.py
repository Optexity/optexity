from __future__ import annotations

import unittest

from optexity.inference.core.automation_cache.converter import (
    _parameterize_planned_steps,
    _value_has_runtime_parameter,
)
from optexity.inference.core.automation_cache.models import (
    ConversionMode,
    ConvertedStep,
    PlannedStep,
    PlannedStepStatus,
)
from optexity.inference.core.automation_cache.parameters import (
    ParameterAllocator,
    ParameterKind,
    RuntimeParameterBinding,
)
from optexity.inference.core.learning_memory.session import (
    _parameter_contract,
    _stable_entry_context,
)
from optexity.schema.actions.interaction_action import InteractionAction
from optexity.schema.automation import (
    ActionNode,
    AmazonSecretsManagerParameter,
    SecureParameter,
)


def _planned_step(
    step_number: int,
    browser_action: str,
    optexity_action: str,
    interaction_action: dict,
) -> PlannedStep:
    node = ActionNode.model_validate(
        {
            "type": "action_node",
            "interaction_action": interaction_action,
            "end_sleep_time": 0,
        }
    )
    converted = ConvertedStep(
        source_step_number=step_number,
        deterministic_candidate_number=step_number,
        browser_use_action=browser_action,
        optexity_action=optexity_action,
        conversion_mode=ConversionMode.NATIVE_DETERMINISTIC,
        prompt_fallback_enabled=False,
        recorded_page_transition_after_step=False,
    )
    return PlannedStep(
        source_step_number=step_number,
        browser_use_action=browser_action,
        status=PlannedStepStatus.CONVERTED,
        converted_step=converted,
        node=node,
    )


def _interaction(step: PlannedStep) -> InteractionAction:
    assert step.node is not None
    assert step.node.interaction_action is not None
    return step.node.interaction_action


class AutomationCacheParameterTests(unittest.TestCase):
    def test_all_direct_string_values_keep_their_runtime_namespaces(self) -> None:
        steps = [
            _planned_step(
                1,
                "navigate",
                "go_to_url",
                {"go_to_url": {"url": "https://example.test/items/42"}},
            ),
            _planned_step(
                2,
                "find_text",
                "scroll_to_text",
                {"scroll_to_text": {"text": "Generated result"}},
            ),
            _planned_step(
                3,
                "send_keys",
                "key_press",
                {"key_press": {"keys": "private shortcut"}},
            ),
        ]
        secure_definition = SecureParameter(
            amazon_secrets_manager=AmazonSecretsManagerParameter(
                secret_name="test/shortcut",
                region_name="us-east-1",
            )
        )
        bindings = [
            RuntimeParameterBinding(
                reference="{result_text[0]}",
                value="Generated result",
                kind=ParameterKind.GENERATED,
            ),
            RuntimeParameterBinding(
                reference="{shortcut[0]}",
                value="private shortcut",
                kind=ParameterKind.SECURE,
            ),
        ]

        converted, parameters = _parameterize_planned_steps(
            steps,
            source_input_parameters={"target_url": ["https://example.test/items/42"]},
            runtime_parameter_bindings=bindings,
            source_secure_parameters={"shortcut": [secure_definition]},
            source_generated_parameters={"result_text": []},
        )

        go_to_url = _interaction(converted[0]).go_to_url
        scroll_to_text = _interaction(converted[1]).scroll_to_text
        key_press = _interaction(converted[2]).key_press
        assert go_to_url is not None
        assert scroll_to_text is not None
        assert key_press is not None
        self.assertEqual(go_to_url.url, "{target_url[0]}")
        self.assertEqual(scroll_to_text.text, "{result_text[0]}")
        self.assertEqual(key_press.keys, "{shortcut[0]}")
        self.assertEqual(parameters.input_parameters, {"target_url": []})
        self.assertEqual(parameters.generated_parameters, {"result_text": []})
        self.assertEqual(
            parameters.secure_parameters, {"shortcut": [secure_definition]}
        )
        self.assertNotIn("private shortcut", repr(bindings[1]))

    def test_unmatched_value_becomes_an_explicit_required_parameter(self) -> None:
        converted, parameters = _parameterize_planned_steps(
            [
                _planned_step(
                    1,
                    "navigate",
                    "go_to_url",
                    {"go_to_url": {"url": "https://unknown.test"}},
                )
            ],
            source_input_parameters=None,
            runtime_parameter_bindings=None,
            source_secure_parameters=None,
            source_generated_parameters=None,
        )

        go_to_url = _interaction(converted[0]).go_to_url
        assert go_to_url is not None
        self.assertEqual(go_to_url.url, "{step_1_navigation_url[0]}")
        self.assertEqual(parameters.input_parameters, {"step_1_navigation_url": []})

    def test_learning_preserves_static_values_but_rebinds_runtime_values(self) -> None:
        converted, parameters = _parameterize_planned_steps(
            [
                _planned_step(
                    1,
                    "navigate",
                    "go_to_url",
                    {"go_to_url": {"url": "https://example.test/items/42"}},
                ),
                _planned_step(
                    2,
                    "find_text",
                    "scroll_to_text",
                    {"scroll_to_text": {"text": "Product details"}},
                ),
            ],
            source_input_parameters=None,
            runtime_parameter_bindings=[
                RuntimeParameterBinding(
                    reference="{target_url[0]}",
                    value="https://example.test/items/42",
                    kind=ParameterKind.INPUT,
                )
            ],
            source_secure_parameters=None,
            source_generated_parameters=None,
            preserve_unmatched_literals=True,
        )

        go_to_url = _interaction(converted[0]).go_to_url
        scroll_to_text = _interaction(converted[1]).scroll_to_text
        assert go_to_url is not None
        assert scroll_to_text is not None
        self.assertEqual(go_to_url.url, "{target_url[0]}")
        self.assertEqual(scroll_to_text.text, "Product details")
        self.assertEqual(parameters.input_parameters, {"target_url": []})

    def test_ambiguous_runtime_value_is_not_frozen_into_memory(self) -> None:
        allocator = ParameterAllocator(
            runtime_bindings=[
                RuntimeParameterBinding(
                    reference="{first[0]}",
                    value="same",
                    kind=ParameterKind.INPUT,
                ),
                RuntimeParameterBinding(
                    reference="{second[0]}",
                    value="same",
                    kind=ParameterKind.INPUT,
                ),
            ],
            preserve_unmatched_literals=True,
        )

        with self.assertRaisesRegex(ValueError, "multiple runtime parameters"):
            allocator.bind("same", source_step_number=1, suffix="input_text")

    def test_sensitive_value_is_safe_only_with_one_runtime_reference(self) -> None:
        binding = RuntimeParameterBinding(
            reference="{password[0]}",
            value="resolved-secret",
            kind=ParameterKind.SECURE,
        )
        self.assertTrue(
            _value_has_runtime_parameter(
                "resolved-secret",
                source_input_parameters=None,
                runtime_parameter_bindings=[binding],
            )
        )
        self.assertFalse(
            _value_has_runtime_parameter(
                "resolved-secret",
                source_input_parameters={"duplicate": ["resolved-secret"]},
                runtime_parameter_bindings=[binding],
            )
        )

    def test_workflow_compatibility_ignores_parameter_values(self) -> None:
        aapl_contract = _parameter_contract({"stock_ticker": ["AAPL"]})
        nvda_contract = _parameter_contract({"stock_ticker": ["NVDA"]})
        self.assertEqual(aapl_contract, nvda_contract)

        aapl_entry = _stable_entry_context(
            "https://example.test/search",
            "https://example.test/search?symbol=AAPL",
            input_parameters={"stock_ticker": ["AAPL"]},
        )
        nvda_entry = _stable_entry_context(
            "https://example.test/search",
            "https://example.test/search?symbol=NVDA",
            input_parameters={"stock_ticker": ["NVDA"]},
        )
        self.assertEqual(aapl_entry, nvda_entry)


if __name__ == "__main__":
    unittest.main()
