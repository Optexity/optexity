from typing import Literal

from pydantic import BaseModel


class LLMAction(BaseModel):
    llm_provider: Literal["gemini", "anthropic", "openai"] = "gemini"
    llm_model_name: str = "gemini-3.5-flash-lite"
