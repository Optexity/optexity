"""LLM layer: model resolution, retries/fallback (fake providers), schemas, cost."""

import pytest

from optexity.inference.models.llm import LLMError, Tier, complete, config, prices, router
from optexity.inference.models.llm.claude import _closed, _payload
from optexity.inference.models.llm.types import Reply, Request
from optexity.inference.models.llm_model import _sanitize_schema_keys
from optexity.utils.llm_settings import llm_settings


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in ("LLM_MODEL", "LLM_MODEL_FALLBACK", "LLM_TIER_LITE", "LLM_TIER_STANDARD", "LLM_TIER_PRO"):
        monkeypatch.setattr(llm_settings, name, None)
    monkeypatch.setattr(config, "LOCATION", "global")
    monkeypatch.setattr(router.time, "sleep", lambda s: None)
    monkeypatch.setattr(prices, "_refresh_in_background", lambda: None)
    router._cooldown_until.clear()
    config.normalize.cache_clear()


def models(chain):
    return [s.model for s in chain]


# --- model resolution -------------------------------------------------------

def test_default_call_uses_lite_tier():
    assert models(config.resolve_chain(None)) == config.TIERS[Tier.LITE]


def test_tier_call_ignores_env_fallback(monkeypatch):
    monkeypatch.setattr(llm_settings, "LLM_MODEL_FALLBACK", "claude-sonnet-5-5")
    assert models(config.resolve_chain(Tier.PRO)) == config.TIERS[Tier.PRO]


def test_pinned_model_falls_back_to_rest_of_its_tier():
    assert models(config.resolve_chain("gemini/gemini-3.6-flash")) == ["gemini-3.6-flash", "claude-sonnet-5-5"]


def test_pinned_model_uses_env_fallback_when_set(monkeypatch):
    monkeypatch.setattr(llm_settings, "LLM_MODEL_FALLBACK", "anthropic/claude-opus-5-5")
    assert models(config.resolve_chain("gemini-3.6-flash")) == ["gemini-3.6-flash", "claude-opus-5-5"]


def test_llm_model_env_replaces_default_primary(monkeypatch):
    monkeypatch.setattr(llm_settings, "LLM_MODEL", "gemini-3.6-flash")
    assert models(config.resolve_chain(None)) == ["gemini-3.6-flash", "claude-sonnet-5-5"]


def test_retired_and_unavailable_models():
    assert config.normalize("gemini/gemini-2.5-pro") is Tier.PRO
    assert config.normalize("bedrock/anthropic.claude-sonnet-4-20250514-v1:0") == "claude-sonnet-5-5"
    assert config.normalize("openrouter/anthropic/claude-sonnet-4.6") == "claude-sonnet-5-5"
    assert config.normalize("openai/gpt-4.1-mini") is None
    assert models(config.resolve_chain("openai/gpt-4.1-mini")) == config.TIERS[Tier.LITE]


def test_pinned_model_from_another_location_uses_its_tier():
    # gemini-4-argon is only listed for "us", but it is a pro model everywhere.
    assert models(config.resolve_chain("gemini-4-argon")) == ["gemini-4-argon", *config.TIERS[Tier.PRO]]


def test_pinned_model_in_no_tier_falls_back_to_catch_all():
    chain = config.resolve_chain("gemini-2.0-flash")
    assert models(chain) == ["gemini-2.0-flash", *config.TIERS[config.CATCH_ALL_TIER]]
    assert chain[0].tier is config.CATCH_ALL_TIER  # gets that tier's timeout


def test_env_fallback_beats_catch_all(monkeypatch):
    monkeypatch.setattr(llm_settings, "LLM_MODEL_FALLBACK", "claude-opus-5-5")
    assert models(config.resolve_chain("gemini-2.0-flash")) == ["gemini-2.0-flash", "claude-opus-5-5"]


def test_location_and_env_tier_overrides(monkeypatch):
    monkeypatch.setattr(config, "LOCATION", "us")
    assert models(config.resolve_chain(Tier.PRO)) == ["gemini-4-argon", "claude-opus-5-5"]
    monkeypatch.setattr(llm_settings, "LLM_TIER_PRO", "claude-opus-5-5, gemini-4-argon")
    assert models(config.resolve_chain(Tier.PRO)) == ["claude-opus-5-5", "gemini-4-argon"]


# --- retries and fallback ---------------------------------------------------

class FakeProvider:
    """Plays back a script of outcomes per model: a Reply or an exception."""

    def __init__(self, script):
        self.script = {m: list(outcomes) for m, outcomes in script.items()}
        self.calls = []

    def call(self, request, model):
        self.calls.append(model)
        outcome = self.script[model].pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def classify(self, exc, model):
        return exc


@pytest.fixture
def fake(monkeypatch):
    def install(script):
        provider = FakeProvider(script)
        monkeypatch.setattr(router, "_provider", lambda model: provider)
        return provider
    return install


def ok(text="hi"):
    return Reply(text=text, prompt_tokens=10, output_tokens=5)


def test_rate_limit_goes_to_next_model_and_cools_down(fake):
    p = fake({"gemini-3.5-flash-lite": [LLMError("429", "rate_limit")], "claude-haiku-5-5": [ok(), ok()]})
    assert complete([], feature="t").model == "claude-haiku-5-5"
    assert complete([], feature="t").model == "claude-haiku-5-5"  # gemini skipped while cooling down
    assert p.calls == ["gemini-3.5-flash-lite", "claude-haiku-5-5", "claude-haiku-5-5"]


def test_transient_error_is_retried_once_on_the_same_model(fake):
    p = fake({"gemini-3.5-flash-lite": [LLMError("503", "transient"), ok()]})
    assert complete([]).model == "gemini-3.5-flash-lite"
    assert p.calls == ["gemini-3.5-flash-lite"] * 2


def test_invalid_output_retried_twice_then_next_model(fake):
    p = fake({"gemini-3.5-flash-lite": [ok("bad")] * 3, "claude-haiku-5-5": [ok("good")]})
    result = complete([], parse=lambda t: t if t == "good" else int("x"))
    assert result.parsed == "good"
    assert p.calls == ["gemini-3.5-flash-lite"] * 3 + ["claude-haiku-5-5"]
    assert result.usage.input_tokens == 40  # failed attempts are billed too


def test_fatal_error_stops_without_fallback(fake):
    p = fake({"gemini-3.5-flash-lite": [LLMError("403", "fatal")]})
    with pytest.raises(LLMError) as e:
        complete([])
    assert e.value.kind == "fatal" and p.calls == ["gemini-3.5-flash-lite"]


def test_all_rate_limited_reports_rate_limit(fake):
    fake({"gemini-3.5-flash-lite": [LLMError("429", "rate_limit")], "claude-haiku-5-5": [LLMError("429", "rate_limit")]})
    with pytest.raises(LLMError) as e:
        complete([])
    assert e.value.kind == "rate_limit"


# --- provider details and cost ----------------------------------------------

def test_claude_schema_closing():
    schema = {"type": "object", "properties": {"a": {"type": "object", "properties": {"b": {"type": "string"}}}}}
    closed = _closed(schema)
    assert closed["additionalProperties"] is False and closed["properties"]["a"]["additionalProperties"] is False
    assert "additionalProperties" not in schema  # input untouched
    assert _closed({"type": "object", "properties": {"m": {"type": "object", "additionalProperties": {"type": "string"}}}}) is None


def test_cost_uses_cached_price_and_regional_uplift(monkeypatch):
    monkeypatch.setattr(prices, "_prices", {})
    # sonnet 5.5 fallback: $2 in, $10 out, $0.10 cached, $2.50 cache write per 1M
    inp, out = prices.cost("claude-sonnet-5-5", prompt=1_000_000, cached=500_000, cache_write=0, output=100_000)
    assert inp == pytest.approx(1.05) and out == pytest.approx(1.0)
    monkeypatch.setattr(prices, "_UPLIFT", 1.1)
    assert prices.cost("claude-sonnet-5-5", 1_000_000, 0, 0, 0)[0] == pytest.approx(2.2)
    assert prices.cost("unknown-model", 100, 0, 0, 100) == (0.0, 0.0)


def test_sanitized_schema_keeps_required_in_sync():
    schema = {"properties": {"first page": {"type": "string"}}, "required": ["first page"]}
    assert _sanitize_schema_keys(schema) == {"properties": {"first_._page": {"type": "string"}}, "required": ["first_._page"]}


def test_claude_request_shape():
    req = Request(
        messages=[{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}],
        schema={"type": "object", "properties": {"a": {"type": "string"}}},
        temperature=0.2, max_tokens=None, reasoning="medium", timeout=60, labels={"feature": "x"},
    )
    kw = _payload(req, "claude-sonnet-5-5")
    assert kw["extra_body"]["output_format"]["schema"]["additionalProperties"] is False
    assert kw["extra_headers"]["anthropic-beta"] == "structured-outputs-2025-11-13"
    assert "X-Vertex-AI-Labels" in kw["extra_headers"]
    assert "temperature" not in kw and kw["system"] == "Be brief." and kw["max_tokens"] > 0
    # A schema with free-form maps can't use JSON mode: it goes into the system prompt.
    req.schema = {"type": "object", "additionalProperties": {"type": "string"}}
    kw = _payload(req, "claude-sonnet-5-5")
    assert "extra_body" not in kw and "JSON schema" in kw["system"]
