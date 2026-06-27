import logging
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from .llm_model import LLMModel, TokenUsage

logger = logging.getLogger(__name__)


class FallbackLLMModel(LLMModel):
    """Wraps multiple LLMModel instances and tries each in order on failure."""

    def __init__(self, models: list[LLMModel]):
        if not models:
            raise ValueError("FallbackLLMModel requires at least one model")
        primary = models[0]
        super().__init__(primary.model_name, primary.use_structured_output)
        self.models = models

    def get_model_response(
        self, prompt: str, system_instruction: Optional[str] = None
    ) -> tuple[str, TokenUsage]:
        last_error = None
        for i, model in enumerate(self.models):
            try:
                return model.get_model_response(prompt, system_instruction)
            except Exception as e:
                last_error = e
                if i < len(self.models) - 1:
                    logger.warning(
                        f"Model {model.model_name} failed: {e}. "
                        f"Falling back to {self.models[i + 1].model_name}"
                    )
        if last_error is not None:
            raise last_error
        raise RuntimeError(
            "FallbackLLMModel: all models exhausted with no error captured"
        )

    def get_model_response_with_structured_output(
        self,
        prompt: str,
        response_schema: type[BaseModel],
        recording_screenshot: Optional[str] = None,
        screenshot: Optional[str] = None,
        pdf_url: Optional[str | Path] = None,
        system_instruction: Optional[str] = None,
    ) -> tuple[BaseModel, TokenUsage]:
        last_error = None
        for i, model in enumerate(self.models):
            try:
                return model.get_model_response_with_structured_output(
                    prompt=prompt,
                    response_schema=response_schema,
                    recording_screenshot=recording_screenshot,
                    screenshot=screenshot,
                    pdf_url=pdf_url,
                    system_instruction=system_instruction,
                )
            except Exception as e:
                last_error = e
                if i < len(self.models) - 1:
                    logger.warning(
                        f"Model {model.model_name} failed: {e}. "
                        f"Falling back to {self.models[i + 1].model_name}"
                    )
        if last_error is not None:
            raise last_error
        raise RuntimeError(
            "FallbackLLMModel: all models exhausted with no error captured"
        )
