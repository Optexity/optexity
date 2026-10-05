from pydantic import BaseModel


class LLMAction(BaseModel):
    # OpenRouter model ID; unset falls through to the task's model, then to the
    # configured default. llm_provider is deprecated.
    llm_provider: str | None = None
    llm_model_name: str | None = None
