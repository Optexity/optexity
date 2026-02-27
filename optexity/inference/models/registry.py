from .llm_model import LLMModel, ModelEnum


ModelEnumType = type[ModelEnum]
MODEL_REGISTRY: dict[ModelEnumType, type[LLMModel]] = {}


def register_model(enum_type: ModelEnumType, cls: type[LLMModel]) -> None:
    MODEL_REGISTRY[enum_type] = cls
