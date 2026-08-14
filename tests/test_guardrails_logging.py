import logging
import unittest

from optexity.guardrails.context import reset_guardrail_runtime, set_guardrail_runtime
from optexity.guardrails.logging import GuardrailRedactionFilter
from optexity.guardrails.models import GuardrailPolicy
from optexity.guardrails.runtime import GuardrailRuntime


class GuardrailLoggingTests(unittest.TestCase):
    def setUp(self):
        self.runtime = GuardrailRuntime(
            task_id="logging-task",
            policy=GuardrailPolicy(),
            start_url="https://example.com",
            input_parameters={"password": ["never-log-this"]},
        )
        self.token = set_guardrail_runtime(self.runtime)

    def tearDown(self):
        reset_guardrail_runtime(self.token)

    def test_persistent_log_filter_redacts_formatted_arguments(self):
        record = logging.LogRecord(
            "test",
            logging.INFO,
            __file__,
            1,
            "credential=%s",
            ("never-log-this",),
            None,
        )
        self.assertTrue(GuardrailRedactionFilter().filter(record))
        self.assertEqual(record.getMessage(), "credential=[REDACTED_SECRET]")

    def test_nested_sensitive_fields_are_redacted_even_when_value_is_new(self):
        value = {
            "result": {
                "access_token": "new-model-produced-token",
                "display_name": "Ada",
            }
        }
        redacted = self.runtime.redact_value(value)
        self.assertEqual(
            redacted["result"]["access_token"], "[REDACTED_SENSITIVE_FIELD]"
        )
        self.assertEqual(redacted["result"]["display_name"], "Ada")


if __name__ == "__main__":
    unittest.main()
