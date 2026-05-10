import logging
from typing import Optional

from pydantic import BaseModel, Field

from optexity.inference.agents.index_prediction.prompt import (
    can_return_negative_index_prompt,
    system_prompt,
)
from optexity.inference.models.llm_model import LLMModel
from optexity.schema.token_usage import TokenUsage

logger = logging.getLogger(__name__)


class IndexPredictionOutputAllowNegative(BaseModel):
    index: int = Field(
        description="The index of the interactive element in the axtree that would achieve the desired outcome. It is either a positive integer or -1 if the element is not found in the axtree."
    )


class IndexPredictionOutputPositiveOnly(BaseModel):
    index: int = Field(
        description="The index of the interactive element in the axtree that would achieve the desired outcome. It is a positive integer."
    )


class ActionPredictionLocatorAxtree:
    def __init__(self, model: LLMModel):
        self.model = model

    def predict_action(
        self,
        goal: str,
        axtree: str,
        screenshot: Optional[str] = None,
        can_return_negative_index: bool = False,
    ) -> tuple[
        str,
        IndexPredictionOutputAllowNegative | IndexPredictionOutputPositiveOnly,
        TokenUsage,
    ]:

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

        response_schema = (
            IndexPredictionOutputAllowNegative
            if can_return_negative_index
            else IndexPredictionOutputPositiveOnly
        )

        response, token_usage = self.model.get_model_response_with_structured_output(
            prompt=final_prompt,
            response_schema=response_schema,
            screenshot=screenshot,
            system_instruction=system_instruction,
        )

        return final_prompt, response, token_usage
