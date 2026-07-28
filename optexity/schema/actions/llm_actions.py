from pydantic import BaseModel


class LLMAction(BaseModel):
    # Any litellm model string; unset falls through to the task's model, then
    # to settings.LLM_MODEL. llm_provider is deprecated but still honored.
    llm_provider: str | None = None
    llm_model_name: str | None = None
