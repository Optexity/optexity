import asyncio

import pytest
from browser_use.llm.exceptions import ModelProviderError, ModelRateLimitError
from browser_use.llm.messages import UserMessage
from pydantic import BaseModel

from optexity.inference.models import Tier, llm_model
from optexity.inference.models.chat_openrouter import build_agent_llm
from optexity.inference.models.openrouter import LLMOutOfCredits

from .openrouter.fakes import FakeOpenRouter, completion, error, make_client


class Step(BaseModel):
    action: str


@pytest.fixture
def fake(monkeypatch):
    async def no_wait(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_wait)

    def install(*replies):
        fake = FakeOpenRouter(*replies)
        monkeypatch.setattr(llm_model, "_client", make_client(fake))
        return fake

    return install


def invoke(llm, output_format=None):
    return asyncio.run(llm.ainvoke([UserMessage(content="go")], output_format))


def test_structured_step_uses_non_strict_browser_use_schema(fake):
    f = fake(completion('{"action": "click"}'))
    result = invoke(build_agent_llm(), Step)
    assert result.completion == Step(action="click")
    fmt = f.requests[0]["response_format"]["json_schema"]
    assert fmt["strict"] is False and fmt["name"] == "Step"
    assert f.requests[0]["trace"]["feature"] == "agentic_task"
    assert result.usage.prompt_tokens == 10


def test_plain_text_completion(fake):
    fake(completion("hello"))
    assert invoke(build_agent_llm()).completion == "hello"


def test_model_is_a_plain_string_for_browser_use():
    assert build_agent_llm().model == "default"
    assert build_agent_llm(Tier.PRO).model == "tier:pro"
    assert build_agent_llm("google/gemini-2.5-pro").model == "google/gemini-2.5-pro"


def test_all_models_rate_limited_is_a_rate_limit_error(fake):
    fake(*[error(429)] * 6)
    with pytest.raises(ModelRateLimitError):
        invoke(build_agent_llm(), Step)


def test_other_failures_are_retryable_provider_errors(fake):
    fake(*[error(503)] * 6)
    with pytest.raises(ModelProviderError):
        invoke(build_agent_llm(), Step)


def test_out_of_credits_escapes_so_the_agent_stops(fake):
    fake(error(402))
    with pytest.raises(LLMOutOfCredits):
        invoke(build_agent_llm(), Step)


def test_call_timeout_covers_every_retry_and_fallback(monkeypatch):
    monkeypatch.setattr(llm_model, "_config", None)
    # lite: gemini 60s, haiku 120s (Anthropic minimum), gpt 60s; 2 attempts each
    # plus up to 10s backoff per model.
    assert build_agent_llm().call_timeout == (60 * 2 + 10) + (120 * 2 + 10) + (
        60 * 2 + 10
    )
    assert build_agent_llm(Tier.PRO).call_timeout == 3 * (300 * 2 + 10)


def test_usage_accumulates_across_calls(fake):
    fake(completion('{"action": "click"}'), completion('{"action": "type"}'))
    llm = build_agent_llm()
    invoke(llm, Step)
    invoke(llm, Step)
    assert (llm.usage.prompt_tokens, llm.usage.completion_tokens) == (20, 10)
    assert llm.usage.cost == pytest.approx(0.002)
