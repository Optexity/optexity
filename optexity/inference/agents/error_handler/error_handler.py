import logging
from typing import Literal

from pydantic import BaseModel

from optexity.inference.agents.error_handler.prompt import system_prompt
from optexity.inference.models import GeminiModels, get_llm_model
from optexity.schema.token_usage import TokenUsage

logger = logging.getLogger(__name__)


class ErrorHandlerOutput(BaseModel):
    error_type: Literal["website_not_loaded", "overlay_popup_blocking", "fatal_error"]
    detailed_reason: str


class ErrorHandlerAgent:
    def __init__(self):
        self.model = get_llm_model(GeminiModels.GEMINI_2_5_FLASH)

    def classify_error(
        self, command: str, screenshot: str
    ) -> tuple[str, ErrorHandlerOutput, TokenUsage]:

        final_prompt = f"""
        [INPUT]
        Command: {command}
        [/INPUT]
        """

        response, token_usage = self.model.get_model_response_with_structured_output(
            prompt=final_prompt,
            response_schema=ErrorHandlerOutput,
            screenshot=screenshot,
            system_instruction=system_prompt,
        )

        return final_prompt, response, token_usage
