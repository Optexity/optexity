from types import SimpleNamespace

from optexity.utils.proxy_config import (
    PROXY_ASSIGNMENT,
    ProxyAssignment,
    pick_proxy,
    pick_random_proxy,
)

A, B, C = (
    "http://isp.example:8001",
    "http://isp.example:8002",
    "http://isp.example:8003",
)
POOL = SimpleNamespace(
    PROXY_URL=f"{A},{B},{C}",
    PROXY_USERNAME="user-a",
    PROXY_PASSWORD="secret",
    PROXY_PROVIDER="other",
    PROXY_COUNTRY=None,
)


class TestPickProxy:
    def test_returns_the_requested_entry(self):
        assert {pick_proxy(POOL, B).url for _ in range(20)} == {B}

    def test_uses_the_running_tasks_assignment_and_marks_it_used(self):
        assignment = ProxyAssignment(C)
        token = PROXY_ASSIGNMENT.set(assignment)
        try:
            assert pick_proxy(POOL).url == C
        finally:
            PROXY_ASSIGNMENT.reset(token)
        assert assignment.used

    def test_falls_back_to_any_entry_for_an_unknown_url(self):
        assert pick_proxy(POOL, "http://gone.example:9999").url in {A, B, C}

    def test_none_without_a_pool(self):
        assert pick_proxy(SimpleNamespace(PROXY_URL=None), A) is None


class TestPickRandomProxy:
    def test_stays_random_even_with_an_assignment(self):
        token = PROXY_ASSIGNMENT.set(ProxyAssignment(A))
        try:
            assert {pick_random_proxy(POOL).url for _ in range(200)} == {A, B, C}
        finally:
            PROXY_ASSIGNMENT.reset(token)
