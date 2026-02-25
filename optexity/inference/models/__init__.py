from .llm_model import GeminiModels, HumanModels, OpenAIModels


def get_llm_model(
    model_name: GeminiModels | HumanModels | OpenAIModels, use_structured_output: bool
):
    if isinstance(model_name, GeminiModels):
        from .gemini import Gemini

        return Gemini(model_name, use_structured_output)

    if isinstance(model_name, OpenAIModels):
        from .openai import OpenAI

        return OpenAI(model_name, use_structured_output)

    # if isinstance(model_name, HumanModels):
    #     from .human import HumanModel

    #     return HumanModel(model_name, use_structured_output)

    raise ValueError(f"Invalid model type: {model_name}")
