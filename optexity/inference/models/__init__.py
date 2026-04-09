import logging

from .llm_model import (
    AnthropicModels,
    GeminiModels,
    HumanModels,
    LLMModel,
    OpenAIModels,
)

logger = logging.getLogger(__name__)

_model_cache: dict[tuple, LLMModel] = {}

FALLBACK_ORDER = ["gemini", "anthropic", "openai"]

# Maps every model value string to a tier name
_MODEL_TIER_MAP: dict[str, str] = {
    # Pro tier
    "gemini-2.5-pro": "pro",
    "claude-opus-4-6": "pro",
    "gpt-4.1": "pro",
    # Standard tier
    "gemini-2.5-flash": "standard",
    "claude-sonnet-4-6": "standard",
    "gpt-4.1-mini": "standard",
    "gpt-4o": "standard",
    # Light tier
    "gemini-2.0-flash": "light",
    "gemini-1.5-flash": "light",
    "gemini-2.5-flash-lite-preview-06-17": "light",
    "claude-haiku-4-5-20251001": "light",
    "gpt-4o-mini": "light",
}

# Maps (tier, provider) to the default model value string for that combination
_TIER_DEFAULTS: dict[tuple[str, str], str] = {
    ("pro", "gemini"): "gemini-2.5-pro",
    ("pro", "anthropic"): "claude-opus-4-6",
    ("pro", "openai"): "gpt-4.1",
    ("standard", "gemini"): "gemini-2.5-flash",
    ("standard", "anthropic"): "claude-sonnet-4-6",
    ("standard", "openai"): "gpt-4.1-mini",
    ("light", "gemini"): "gemini-2.0-flash",
    ("light", "anthropic"): "claude-haiku-4-5-20251001",
    ("light", "openai"): "gpt-4o-mini",
}


def resolve_model_name(
    provider: str, model_name: str
) -> GeminiModels | AnthropicModels | OpenAIModels:
    if provider == "gemini":
        return GeminiModels(model_name)
    elif provider == "anthropic":
        return AnthropicModels(model_name)
    elif provider == "openai":
        return OpenAIModels(model_name)
    else:
        raise ValueError(f"Invalid LLM provider: {provider}")


def get_equivalent_model(
    model_value: str, target_provider: str
) -> GeminiModels | AnthropicModels | OpenAIModels:
    """Get the equivalent model for a different provider based on tier mapping."""
    tier = _MODEL_TIER_MAP.get(model_value)
    if tier is None:
        # Unknown model, fall back to standard tier
        tier = "standard"
    target_value = _TIER_DEFAULTS.get((tier, target_provider))
    if target_value is None:
        raise ValueError(
            f"No equivalent model for tier={tier}, provider={target_provider}"
        )
    return resolve_model_name(target_provider, target_value)


def _try_create_model(
    model_name: GeminiModels | HumanModels | OpenAIModels | AnthropicModels,
    use_structured_output: bool,
) -> LLMModel | None:
    """Try to create a model. Returns instance on success, None on failure.
    Results are cached — successful models in _model_cache, failures in _failed_models.
    """
    cache_key = (model_name, use_structured_output)

    # Already cached
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    # Try creating
    try:
        if isinstance(model_name, GeminiModels):
            from .gemini import Gemini

            instance = Gemini(model_name, use_structured_output)

        elif isinstance(model_name, OpenAIModels):
            from .openai import OpenAI

            instance = OpenAI(model_name, use_structured_output)

        elif isinstance(model_name, AnthropicModels):
            from .anthropic import Anthropic

            instance = Anthropic(model_name, use_structured_output)

        else:
            raise ValueError(f"Invalid model type: {model_name}")

        _model_cache[cache_key] = instance
        logger.info(
            f"Created model {model_name.value} (structured={use_structured_output})"
        )
        return instance

    except Exception as e:
        logger.warning(f"Model {model_name.value} not available: {e}")
        return None


def get_llm_model(
    model_name: GeminiModels | HumanModels | OpenAIModels | AnthropicModels,
    use_structured_output: bool,
) -> LLMModel:
    model = _try_create_model(model_name, use_structured_output)
    if model is not None:
        return model
    raise ValueError(
        f"Model {model_name.value} (structured={use_structured_output}) not available"
    )


def _get_first_fallback(
    provider: str, model_name: str, use_structured_output: bool
) -> LLMModel | None:
    """Try fallback providers one by one, return the first that works."""
    for fallback_provider in FALLBACK_ORDER:
        if fallback_provider == provider:
            continue
        try:
            equiv = get_equivalent_model(model_name, fallback_provider)
            fb_model = _try_create_model(equiv, use_structured_output)
            if fb_model is not None:
                logger.info(
                    f"Fallback: {provider}/{model_name} "
                    f"-> {fallback_provider}/{equiv.value}"
                )
                return fb_model
            else:
                logger.warning(
                    f"Fallback model {fallback_provider}/{equiv.value} not available"
                )
        except Exception as e:
            logger.warning(f"No equivalent model for {fallback_provider}: {e}")
    return None


def get_llm_model_with_fallback(
    provider: str, model_name: str, use_structured_output: bool
) -> LLMModel:
    """Get an LLMModel with one fallback for inference-time failures."""
    from .fallback import FallbackLLMModel

    model_enum = resolve_model_name(provider, model_name)

    # Try primary
    primary = _try_create_model(model_enum, use_structured_output)

    if primary is not None:
        logger.info(f"Using primary model {provider}/{model_name}")

        # Try to add one fallback for inference-time safety
        fallback = _get_first_fallback(provider, model_name, use_structured_output)
        if fallback is not None:
            return FallbackLLMModel([primary, fallback])
        return primary

    # Primary failed at init — use first available fallback directly
    logger.warning(
        f"Primary model {provider}/{model_name} not available, trying fallbacks..."
    )
    fallback = _get_first_fallback(provider, model_name, use_structured_output)
    if fallback is not None:
        return fallback

    raise RuntimeError(
        f"No models available: primary {provider}/{model_name} "
        f"and all fallbacks failed"
    )
