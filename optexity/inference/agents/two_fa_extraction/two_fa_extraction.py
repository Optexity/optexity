import json
import logging

from pydantic import BaseModel, Field

from optexity.inference.agents.two_fa_extraction.prompt import system_prompt
from optexity.inference.models import GeminiModels, get_llm_model
from optexity.schema.inference import Message
from optexity.schema.token_usage import TokenUsage

logger = logging.getLogger(__name__)


class TwoFAExtractionOutput(BaseModel):
    code: str | list[str] | None = Field(
        description="The 2FA code extracted from the messages."
    )


class TwoFAExtraction:
    def __init__(self):
        self.model = get_llm_model(GeminiModels.GEMINI_2_5_FLASH, True)

    def extract_code(
        self, instructions: str | None, messages: list[Message]
    ) -> tuple[str, TwoFAExtractionOutput, TokenUsage]:

        final_prompt = ""

        if instructions is not None:
            final_prompt += f"""
            [EXTRACTION INSTRUCTIONS]
            {instructions}
            [/EXTRACTION INSTRUCTIONS]
            """
        final_prompt += f"""
        [MESSAGES] 
        {json.dumps([message.model_dump(include={"message_text"}) for message in messages], indent=2)} 
        [/MESSAGES]
        """

        response, token_usage = self.model.get_model_response_with_structured_output(
            prompt=final_prompt,
            response_schema=TwoFAExtractionOutput,
            system_instruction=system_prompt,
        )
        return final_prompt, response, token_usage
