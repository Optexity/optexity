import logging
import time

from .llm_model import GeminiModels, HumanModels, LLMModel, OpenAIModels

logger = logging.getLogger(__name__)

_model_cache: dict[tuple, LLMModel] = {}

MAX_RETRIES = 5
INITIAL_BACKOFF_S = 1.0
BACKOFF_FACTOR = 2


def get_llm_model(
    model_name: GeminiModels | HumanModels | OpenAIModels, use_structured_output: bool
) -> LLMModel:
    cache_key = (model_name, use_structured_output)
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    model = _create_model_with_backoff(model_name, use_structured_output)
    _model_cache[cache_key] = model
    return model


def _create_model_with_backoff(
    model_name: GeminiModels | HumanModels | OpenAIModels, use_structured_output: bool
) -> LLMModel:
    if isinstance(model_name, GeminiModels):
        from .gemini import Gemini

        factory = lambda: Gemini(model_name, use_structured_output)

    # elif isinstance(model_name, OpenAIModels):
    #     from .openai import OpenAI
    #     factory = lambda: OpenAI(model_name, use_structured_output)

    # elif isinstance(model_name, HumanModels):
    #     from .human import HumanModel
    #     factory = lambda: HumanModel(model_name, use_structured_output)

    else:
        raise ValueError(f"Invalid model type: {model_name}")

    backoff = INITIAL_BACKOFF_S
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return factory()
        except Exception as e:
            if attempt == MAX_RETRIES:
                logger.warning(
                    f"Failed to create {model_name} (attempt {attempt}/{MAX_RETRIES}): {e}. "
                    f"Aborting..."
                )
                raise
            logger.warning(
                f"Failed to create {model_name} (attempt {attempt}/{MAX_RETRIES}): {e}. "
                f"Retrying in {backoff:.1f}s..."
            )
            time.sleep(backoff)
            backoff *= BACKOFF_FACTOR

    raise RuntimeError(
        f"Failed to create model {model_name} after {MAX_RETRIES} attempts"
    )
