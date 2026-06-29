from typing import Literal

from pydantic import BaseModel


class LLMAction(BaseModel):
    llm_provider: Literal["gemini", "anthropic", "openai"] = "gemini"
    llm_model_name: str = "gemini-2.5-flash"
