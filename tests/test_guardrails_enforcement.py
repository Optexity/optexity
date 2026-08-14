import unittest
from types import SimpleNamespace

from optexity.guardrails.context import reset_guardrail_runtime, set_guardrail_runtime
from optexity.guardrails.enforcement import (
    authorize_action_node,
    authorize_interaction,
    authorize_private_node,
)
from optexity.guardrails.exceptions import GuardrailViolation
from optexity.guardrails.models import GuardrailPolicy
from optexity.guardrails.runtime import GuardrailRuntime


class FakeBrowser:
    def __init__(self, url: str):
        self.url = url

    async def get_current_page_url(self):
        return self.url


class EnforcementBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.runtime = GuardrailRuntime(
            task_id="integration-task",
            policy=GuardrailPolicy(),
            start_url="https://portal.example.com",
            input_parameters={"password": ["secret-value"]},
        )
        self.token = set_guardrail_runtime(self.runtime)
        self.browser = FakeBrowser("https://portal.example.com/form")

    async def asyncTearDown(self):
        reset_guardrail_runtime(self.token)

    async def test_typed_navigation_is_denied_before_handler_execution(self):
        interaction = SimpleNamespace(
            **{
                field: None
                for field in (
                    "click_element",
                    "input_text",
                    "select_option",
                    "check",
                    "uncheck",
                    "hover",
                    "download_url_as_pdf",
                    "scroll",
                    "upload_file",
                    "go_back",
                    "switch_tab",
                    "close_current_tab",
                    "close_all_but_last_tab",
                    "close_tabs_until",
                    "agentic_task",
                    "close_overlay_popup",
                    "key_press",
                )
            },
            go_to_url=SimpleNamespace(url="https://attacker.example/collect"),
        )
        with self.assertRaisesRegex(GuardrailViolation, "Target URL"):
            await authorize_interaction(interaction, self.browser)

    async def test_python_extraction_is_not_misclassified_as_safe_extraction(self):
        node = SimpleNamespace(
            python_script_action=None,
            powershell_action=None,
            assertion_action=None,
            sleep_action=None,
            extraction_action=SimpleNamespace(
                python_script=SimpleNamespace(script="dangerous()"), api_call=None
            ),
            misc_action=None,
        )
        with self.assertRaisesRegex(GuardrailViolation, "python_script"):
            await authorize_action_node(node, self.browser)

    async def test_regular_extraction_remains_allowed(self):
        node = SimpleNamespace(
            python_script_action=None,
            powershell_action=None,
            assertion_action=None,
            sleep_action=None,
            extraction_action=SimpleNamespace(python_script=None, api_call=None),
            misc_action=None,
        )
        await authorize_action_node(node, self.browser)
        self.assertEqual(self.runtime.audit.events[-1].action, "extract")

    async def test_opaque_private_node_is_blocked_before_plugin_execution(self):
        node = SimpleNamespace(handler="vendor.secret_export")
        with self.assertRaisesRegex(GuardrailViolation, "private_node"):
            await authorize_private_node(node, self.browser)


if __name__ == "__main__":
    unittest.main()
