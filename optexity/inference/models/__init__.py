import logging

from .llm import Tier
from .llm_model import LLMModel

logger = logging.getLogger(__name__)

_model_cache: dict[tuple[str | Tier | None, bool], LLMModel] = {}


def normalize_model(provider: str | None, model_name: str | None) -> str | None:
    """The task's (provider, model) pair as one model string, or None for the
    configured default.

    `llm_provider` is deprecated — a full model string in `llm_model_name` is
    preferred — but existing workflow JSON still sets it.
    """
    if not model_name:
        return None
    if "/" in model_name or not provider:
        return model_name
    return f"{provider}/{model_name}"


def get_llm_model(
    model_name: str | Tier | None, use_structured_output: bool
) -> LLMModel:
    cache_key = (model_name, use_structured_output)
    if cache_key not in _model_cache:
        _model_cache[cache_key] = LLMModel(model_name, use_structured_output)
        logger.info(f"Created model {model_name} (structured={use_structured_output})")
    return _model_cache[cache_key]


def get_llm_model_with_fallback(
    provider: str | None, model_name: str | None, use_structured_output: bool
) -> LLMModel:
    """Fallback (LLM_MODEL_FALLBACK, else the model's tier) happens in the LLM layer."""
    return get_llm_model(normalize_model(provider, model_name), use_structured_output)
