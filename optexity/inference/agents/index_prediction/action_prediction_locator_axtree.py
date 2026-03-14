import logging
from typing import Optional

from pydantic import BaseModel, Field

from optexity.inference.agents.index_prediction.prompt import (
    can_return_negative_index_prompt,
    system_prompt,
)
from optexity.inference.models import GeminiModels, get_llm_model
from optexity.schema.token_usage import TokenUsage

logger = logging.getLogger(__name__)


class IndexPredictionOutput(BaseModel):
    index: int = Field(
        description="The index of the interactive element in the axtree that would achieve the desired outcome. It is always greater than 0."
    )


class ActionPredictionLocatorAxtree:
    def __init__(self):
        self.model = get_llm_model(GeminiModels.GEMINI_2_5_FLASH, True)

    def predict_action(
        self,
        goal: str,
        axtree: str,
        screenshot: Optional[str] = None,
        can_return_negative_index: bool = False,
    ) -> tuple[str, IndexPredictionOutput, TokenUsage]:

        final_prompt = f"""
        [INPUT]
        Goal: {goal}

        [AXTREE] 
        {axtree} 
        [/AXTREE]

        [/INPUT]
        """

        system_instruction = f"""
        {system_prompt}
        """
        if can_return_negative_index:
            system_instruction += f"\n{can_return_negative_index_prompt}"

        response, token_usage = self.model.get_model_response_with_structured_output(
            prompt=final_prompt,
            response_schema=IndexPredictionOutput,
            screenshot=screenshot,
            system_instruction=system_prompt,
        )

        return final_prompt, response, token_usage
