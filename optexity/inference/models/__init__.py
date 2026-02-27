import logging
import time

from .llm_model import LLMModel, ModelEnum
from .registry import MODEL_REGISTRY

logger = logging.getLogger(__name__)

__all__ = ["GeminiModels"]

# Re-export for convenience
from .llm_model import GeminiModels

_model_cache: dict[ModelEnum, LLMModel] = {}

MAX_RETRIES = 5
INITIAL_BACKOFF_S = 1.0
BACKOFF_FACTOR = 2


def get_llm_model(model_name: ModelEnum) -> LLMModel:
    """Get or create an LLM model instance with caching and backoff."""
    if model_name in _model_cache:
        return _model_cache[model_name]

    model = _create_model_with_backoff(model_name)
    _model_cache[model_name] = model
    return model


def _create_model_with_backoff(model_name: ModelEnum) -> LLMModel:
    """Create a model instance with exponential backoff on failure."""
    # Find the model class from the registry
    model_cls = None
    for enum_type, cls in MODEL_REGISTRY.items():
        if isinstance(model_name, enum_type):
            model_cls = cls
            break

    if model_cls is None:
        raise ValueError(f"Invalid model type: {model_name}")

    backoff = INITIAL_BACKOFF_S
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return model_cls(model_name)
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
