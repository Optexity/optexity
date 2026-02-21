from optexity.inference.models import GeminiModels, OpenAIModels, get_llm_model
from optexity.utils.settings import settings


def get_agent_model(use_structured_output: bool):
    if settings.AGENT_LLM_PROVIDER == "openai":
        return get_llm_model(OpenAIModels(settings.AGENT_LLM_MODEL), use_structured_output)
    return get_llm_model(GeminiModels(settings.AGENT_LLM_MODEL), use_structured_output)