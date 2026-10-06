import base64

import pytest
from pydantic import BaseModel

from optexity.inference.models import (
    Tier,
    get_llm_model,
    get_llm_model_with_fallback,
    llm_model,
    normalize_model,
)
from optexity.inference.models.openrouter import LLMOutOfCredits

from .openrouter.fakes import FakeOpenRouter, completion, error, make_client

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


class Answer(BaseModel):
    answer: str


@pytest.fixture
def fake(monkeypatch):
    def install(*replies, **config):
        fake = FakeOpenRouter(*replies)
        monkeypatch.setattr(llm_model, "_client", make_client(fake, **config))
        return fake

    return install


def test_default_call_uses_lite_tier_and_maps_token_usage(fake):
    f = fake(completion('{"answer": "42"}'))
    parsed, usage = get_llm_model(None).get_model_response_with_structured_output(
        prompt="q", response_schema=Answer, system_instruction="be brief"
    )
    assert parsed == Answer(answer="42")
    body = f.requests[0]
    assert body["model"] == "google/gemini-3.5-flash-lite"
    assert body["messages"][0] == {"role": "system", "content": "be brief"}
    assert body["messages"][1]["content"][-1] == {"type": "text", "text": "q"}
    assert (usage.input_tokens, usage.output_tokens, usage.thoughts_tokens) == (
        10,
        5,
        2,
    )
    assert (usage.total_tokens, usage.total_cost) == (15, 0.001)


def test_tier_and_explicit_model(fake):
    f = fake(completion('{"answer": "a"}'), completion('{"answer": "b"}'))
    get_llm_model(Tier.PRO).get_model_response_with_structured_output("q", Answer)
    get_llm_model(
        "openrouter/google/gemini-2.5-flash"
    ).get_model_response_with_structured_output("q", Answer)
    assert f.models == ["google/gemini-2.5-pro", "google/gemini-2.5-flash"]


def test_captcha_style_override_else_tier(fake):
    f = fake(completion('{"answer": "a"}'), completion('{"answer": "b"}'))
    for override in (None, "anthropic/claude-sonnet-4.6"):
        get_llm_model(override or Tier.PRO).get_model_response_with_structured_output(
            "q", Answer
        )
    assert f.models == ["google/gemini-2.5-pro", "anthropic/claude-sonnet-4.6"]


def test_screenshot_and_pdf_are_attached(fake, tmp_path):
    f = fake(completion('{"answer": "a"}'))
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")
    get_llm_model(None).get_model_response_with_structured_output(
        "q", Answer, screenshot=base64.b64encode(PNG).decode(), pdf_url=pdf
    )
    kinds = [p["type"] for p in f.requests[0]["messages"][-1]["content"]]
    assert kinds == ["image_url", "file", "text"]


def test_feature_and_session_tags_reach_openrouter(fake):
    f = fake(completion('{"answer": "a"}'), service="optexity-worker")
    get_llm_model(None).get_model_response_with_structured_output(
        "q", Answer, feature="captcha", session_id="task-1"
    )
    body = f.requests[0]
    assert body["user"] == "optexity-worker:captcha"
    assert body["session_id"] == "task-1"
    assert body["trace"]["feature"] == "captcha"


def test_fatal_errors_reach_the_caller(fake):
    fake(error(402, "Insufficient credits"))
    with pytest.raises(LLMOutOfCredits):
        get_llm_model(None).get_model_response_with_structured_output("q", Answer)


def test_provider_and_model_pairs():
    assert normalize_model(None, None) is None
    assert normalize_model("gemini", None) is None
    assert normalize_model(None, "google/x") == "google/x"
    assert normalize_model("google", "x") == "google/x"
    assert get_llm_model_with_fallback(None, None, True) is get_llm_model(None, True)
