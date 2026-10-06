import pytest

from optexity.inference.models.openrouter import (
    LLMConfig,
    LLMConfigError,
    Tier,
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


def test_context_too_large_skips_small_fallback():
    # Haiku 4.5 has a 200k context window.
    assert chain(model=Tier.LITE, est_tokens=300_000) == [LITE[0], LITE[2]]


def test_user_chosen_model_is_never_filtered():
    cfg = LLMConfig(api_key="k", default_model="anthropic/claude-haiku-4.5")
    assert chain(cfg, est_tokens=300_000)[0] == "anthropic/claude-haiku-4.5"


def test_reasoning_and_timeout_per_tier():
    lite = resolve_chain(LLMConfig(api_key="k"))[0]
    pro = resolve_chain(LLMConfig(api_key="k"), model=Tier.PRO)[0]
    assert (lite.reasoning, lite.timeout) == ("medium", 45.0)
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


def test_no_model_left_after_filtering_is_a_config_error():
    cfg = LLMConfig.with_tier_overrides(
        {Tier.LITE: ["anthropic/claude-haiku-4.5"]}, api_key="k"
    )
    with pytest.raises(LLMConfigError):
        chain(cfg, model=Tier.LITE, est_tokens=300_000)


def test_anthropic_models_get_a_minimum_timeout():
    timeouts = {e.model: e.timeout for e in resolve_chain(LLMConfig(api_key="k"))}
    assert timeouts == {
        "google/gemini-3.5-flash-lite": 45.0,
        "anthropic/claude-haiku-4.5": 120.0,
        "openai/gpt-5.6-luna": 45.0,
    }
    pinned = resolve_chain(LLMConfig(api_key="k"), "anthropic/claude-3-haiku")[0]
    assert pinned.timeout == 120.0
    pro = resolve_chain(LLMConfig(api_key="k"), Tier.PRO)
    assert [e.timeout for e in pro] == [300.0, 300.0, 300.0]
