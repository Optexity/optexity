import unittest

from optexity.guardrails.exceptions import GuardrailViolation
from optexity.guardrails.expressions import safe_evaluate, safe_locator_from_command


class SafeExpressionTests(unittest.TestCase):
    def test_workflow_boolean_and_index_expressions(self):
        scope = {
            "logged_in": [True],
            "status": ["ready"],
            "attempts": [2],
        }
        self.assertTrue(
            safe_evaluate(
                "logged_in[0] and status[0] == 'ready' and attempts[0] < 3",
                scope,
            )
        )
        self.assertFalse(safe_evaluate("not logged_in[0]", scope))

    def test_poll_response_dot_paths(self):
        response = {"status_code": 200, "body": {"jobs": [{"status": "done"}]}}
        self.assertTrue(
            safe_evaluate(
                "status_code == 200 and body.jobs[0].status == 'done'", response
            )
        )

    def test_calls_imports_and_dunder_access_are_rejected(self):
        expressions = [
            "__import__('os').system('id')",
            "value.__class__",
            "open('/etc/passwd')",
            "[x for x in values]",
        ]
        for expression in expressions:
            with (
                self.subTest(expression=expression),
                self.assertRaises(GuardrailViolation),
            ):
                safe_evaluate(expression, {"value": "x", "values": [1]})


class FakeLocator:
    def __init__(self, calls=None):
        self.calls = calls if calls is not None else []

    def _call(self, call_name, *args, **kwargs):
        self.calls.append((call_name, args, kwargs))
        return FakeLocator(self.calls)

    def get_by_role(self, *args, **kwargs):
        return self._call("get_by_role", *args, **kwargs)

    def get_by_text(self, *args, **kwargs):
        return self._call("get_by_text", *args, **kwargs)

    def locator(self, *args, **kwargs):
        return self._call("locator", *args, **kwargs)

    def nth(self, *args, **kwargs):
        return self._call("nth", *args, **kwargs)

    @property
    def first(self):
        self.calls.append(("first", (), {}))
        return FakeLocator(self.calls)


class SafeLocatorTests(unittest.TestCase):
    def test_whitelisted_locator_chain(self):
        page = FakeLocator()
        result = safe_locator_from_command(
            page, 'get_by_role("button", name="Save", exact=True).nth(0)'
        )
        self.assertIsInstance(result, FakeLocator)
        self.assertEqual(page.calls[0][0], "get_by_role")
        self.assertEqual(page.calls[1], ("nth", (0,), {}))

    def test_locator_property_chain(self):
        page = FakeLocator()
        safe_locator_from_command(page, 'get_by_text("Continue").first')
        self.assertEqual(page.calls[-1][0], "first")

    def test_arbitrary_methods_and_nonliteral_arguments_are_rejected(self):
        page = FakeLocator()
        commands = [
            "__class__.__mro__",
            'locator(__import__("os").system("id"))',
            'evaluate("document.cookie")',
        ]
        for command in commands:
            with self.subTest(command=command), self.assertRaises(GuardrailViolation):
                safe_locator_from_command(page, command)


if __name__ == "__main__":
    unittest.main()
