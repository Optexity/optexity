from .llm_model import GeminiModels, ModelEnum
from .registry import MODEL_REGISTRY

__all__ = ["GeminiModels"]


def get_llm_model(model_name: ModelEnum):
    for enum_type, cls in MODEL_REGISTRY.items():
        if isinstance(model_name, enum_type):
            return cls(model_name)
    raise ValueError(f"Invalid model type: {model_name}")
