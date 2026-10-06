import pytest

from optexity.inference.models.openrouter import (
    LLMConfig,
    Tier,
    normalize_model_id,
    resolve_chain,
)

LITE = [
    "google/gemini-3.5-flash-lite",
    "anthropic/claude-haiku-4.5",
    "openai/gpt-5.6-luna",
]
STANDARD = [
    "google/gemini-3.6-flash",
    "anthropic/claude-sonnet-4.6",
    "openai/gpt-5.6-terra",
]
PRO = ["google/gemini-2.5-pro", "anthropic/claude-opus-4.7", "openai/gpt-5.6-sol"]
FALLBACK = "openai/gpt-5-mini"


def chain(config=None, **kw):
    return [e.model for e in resolve_chain(config or LLMConfig(api_key="k"), **kw)]


def test_override_with_env_fallback():
    cfg = LLMConfig(api_key="k", env_fallback=FALLBACK)
    assert chain(cfg, model="google/gemini-2.5-pro") == [
        "google/gemini-2.5-pro",
        FALLBACK,
    ]


def test_override_in_a_tier_falls_back_to_rest_of_its_tier():
    assert chain(model="anthropic/claude-sonnet-4.6") == [
        "anthropic/claude-sonnet-4.6",
        "openai/gpt-5.6-terra",
    ]


def test_override_not_in_any_tier_has_no_fallback():
    assert chain(model="mistralai/mistral-large") == ["mistralai/mistral-large"]


def test_default_with_llm_model_and_env_fallback():
    cfg = LLMConfig(
        api_key="k", default_model="google/gemini-3.6-flash", env_fallback=FALLBACK
    )
    assert chain(cfg) == ["google/gemini-3.6-flash", FALLBACK]


def test_default_with_llm_model_falls_back_to_lite_tier():
    cfg = LLMConfig(api_key="k", default_model="mistralai/mistral-large")
    assert chain(cfg) == ["mistralai/mistral-large", *LITE]


def test_default_without_llm_model_uses_lite_tier():
    assert chain() == LITE


def test_default_without_llm_model_with_env_fallback():
    cfg = LLMConfig(api_key="k", env_fallback=FALLBACK)
    assert chain(cfg) == [LITE[0], FALLBACK]


def test_tier_call_ignores_env_fallback():
    cfg = LLMConfig(api_key="k", env_fallback=FALLBACK)
    assert chain(cfg, model=Tier.PRO) == PRO
    assert chain(cfg, model=Tier.STANDARD) == STANDARD


def test_fallback_equal_to_primary_is_dropped():
    cfg = LLMConfig(api_key="k", env_fallback=LITE[0])
    assert chain(cfg) == [LITE[0]]


def test_tier_override_replaces_list():
    cfg = LLMConfig.with_tier_overrides({Tier.PRO: ["a/x", "b/y"]}, api_key="k")
    assert chain(cfg, model=Tier.PRO) == ["a/x", "b/y"]
    assert chain(cfg, model=Tier.LITE) == LITE


def test_reasoning_and_timeout_per_tier():
    lite = resolve_chain(LLMConfig(api_key="k"))[0]
    pro = resolve_chain(LLMConfig(api_key="k"), model=Tier.PRO)[0]
    assert (lite.reasoning, lite.timeout) == ("medium", 120.0)
    assert (pro.reasoning, pro.timeout) == (None, 300.0)


@pytest.mark.parametrize(
    "given",
    [
        "google/gemini-2.5-pro",
        "openrouter/google/gemini-2.5-pro",
        "  google/gemini-2.5-pro ",
    ],
)
def test_openrouter_prefix_and_whitespace_are_accepted(given):
    assert chain(model=given)[0] == "google/gemini-2.5-pro"


def test_env_values_accept_openrouter_prefix_too():
    cfg = LLMConfig(
        api_key="k",
        default_model="openrouter/google/gemini-3.6-flash",
        env_fallback="openrouter/openai/gpt-5-mini",
    )
    assert chain(cfg) == ["google/gemini-3.6-flash", "openai/gpt-5-mini"]


def test_plain_string_is_never_treated_as_a_tier():
    assert chain(model="pro") == ["pro"]  # sent as a model ID; OpenRouter rejects it


def test_every_model_in_a_tier_gets_the_tier_timeout():
    timeouts = {e.model: e.timeout for e in resolve_chain(LLMConfig(api_key="k"))}
    assert set(timeouts.values()) == {120.0}
    # Not in any tier: the standard timeout.
    pinned = resolve_chain(LLMConfig(api_key="k"), "anthropic/claude-3-haiku")[0]
    assert pinned.timeout == 180.0
    pro = resolve_chain(LLMConfig(api_key="k"), Tier.PRO)
    assert [e.timeout for e in pro] == [300.0, 300.0, 300.0]


@pytest.mark.parametrize(
    "old, new",
    [
        # values found in saved workflows
        ("gemini/gemini-2.5-pro", "google/gemini-2.5-pro"),
        ("gemini/gemini-2.5-flash", "google/gemini-2.5-flash"),
        ("gemini-2.5-pro", "google/gemini-2.5-pro"),
        ("anthropic/claude-opus-4-6", "anthropic/claude-opus-4.6"),
        ("openai/gpt-4.1", "openai/gpt-4.1"),
        ("openai/gpt-4o-mini", "openai/gpt-4o-mini"),
        # other LiteLLM / provider-native forms
        ("claude-haiku-4-5-20251001", "anthropic/claude-haiku-4.5"),
        ("claude-3-5-sonnet-20241022", "anthropic/claude-3.5-sonnet"),
        ("vertex_ai/claude-sonnet-4-5@20250929", "anthropic/claude-sonnet-4.5"),
        ("vertex_ai/gemini-2.5-pro", "google/gemini-2.5-pro"),
        (
            "bedrock/anthropic.claude-sonnet-4-20250514-v1:0",
            "anthropic/claude-sonnet-4",
        ),
        ("gpt-4o", "openai/gpt-4o"),
        ("o3-mini", "openai/o3-mini"),
        # already OpenRouter IDs, or unknown: unchanged
        ("openrouter/google/gemini-2.5-pro", "google/gemini-2.5-pro"),
        ("anthropic/claude-sonnet-4.6", "anthropic/claude-sonnet-4.6"),
        ("mistralai/mistral-large", "mistralai/mistral-large"),
        ("some-unknown-model", "some-unknown-model"),
    ],
)
def test_legacy_model_ids_are_converted(old, new):
    assert normalize_model_id(old) == new


def test_alias_map_wins_over_rules(monkeypatch):
    from optexity.inference.models.openrouter import core

    monkeypatch.setitem(
        core.LEGACY_MODEL_ALIASES, "old-custom", "google/gemini-3.6-flash"
    )
    assert normalize_model_id("old-custom") == "google/gemini-3.6-flash"


def test_legacy_pin_falls_back_like_its_new_id():
    assert chain(model="gemini/gemini-2.5-pro") == PRO
