import logging

from .llm_model import LLMModel, get_client, get_config
from .openrouter import Tier

logger = logging.getLogger(__name__)

_model_cache: dict[tuple[str | Tier | None, bool], LLMModel] = {}


def normalize_model(provider: str | None, model_name: str | None) -> str | None:
    """The model ID from a task/action's (provider, model) pair; None means the
    configured default.

    `llm_provider` is deprecated — a full OpenRouter ID in `llm_model_name` is
    preferred — but older workflow JSON still sets it.
    """
    if not model_name:
        return None
    if "/" in model_name or not provider:
        return model_name
    return f"{provider}/{model_name}"


def get_llm_model(
    model_name: str | Tier | None, use_structured_output: bool = True
) -> LLMModel:
    """``model_name``: OpenRouter model ID, a `Tier`, or None for the default."""
    cache_key = (model_name, use_structured_output)
    if cache_key not in _model_cache:
        _model_cache[cache_key] = LLMModel(model_name, use_structured_output)
        logger.info(f"Created model {model_name} (structured={use_structured_output})")
    return _model_cache[cache_key]


def get_llm_model_with_fallback(
    provider: str | None, model_name: str | None, use_structured_output: bool
) -> LLMModel:
    """Same as `get_llm_model`, from a (provider, model) pair. Fallback happens in
    the OpenRouter client."""
    return get_llm_model(normalize_model(provider, model_name), use_structured_output)


__all__ = [
    "LLMModel",
    "Tier",
    "get_client",
    "get_config",
    "get_llm_model",
    "get_llm_model_with_fallback",
    "normalize_model",
]
