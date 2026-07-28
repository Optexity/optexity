import logging

from optexity.utils.llm_settings import llm_settings

from .litellm_model import LiteLLMModel
from .llm_model import LLMModel

logger = logging.getLogger(__name__)

_model_cache: dict[tuple[str, bool], LLMModel] = {}


def normalize_model(provider: str | None, model_name: str | None) -> str:
    """Build a litellm model string from the task's (provider, model) pair.

    `llm_provider` is deprecated — a full "provider/model" string in
    `llm_model_name` is preferred — but existing workflow JSON still sets it.
    """
    if not model_name:
        return llm_settings.LLM_MODEL
    if "/" in model_name:
        return model_name
    if provider:
        return f"{provider}/{model_name}"
    return model_name


def get_llm_model(model_name: str, use_structured_output: bool) -> LLMModel:
    cache_key = (model_name, use_structured_output)
    if cache_key not in _model_cache:
        _model_cache[cache_key] = LiteLLMModel(model_name, use_structured_output)
        logger.info(f"Created model {model_name} (structured={use_structured_output})")
    return _model_cache[cache_key]


def get_llm_model_with_fallback(
    provider: str | None, model_name: str | None, use_structured_output: bool
) -> LLMModel:
    """Fallback is handled inside litellm via llm_settings.LLM_MODEL_FALLBACK."""
    return get_llm_model(normalize_model(provider, model_name), use_structured_output)
