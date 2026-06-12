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
    reasoning: str = Field(
        description="Brief reasoning: the specific element the goal targets, whether the axtree contains an element whose own text/label/role clearly matches it, and why you chose this index (or -1)."
    )
    matched_element_text: str = Field(
        description='The visible text, label, or role of the element you selected, copied verbatim from the axtree. Set to "" when returning -1 because no element matches.'
    )
    index: int = Field(
        description="The index of the interactive element in the axtree that would achieve the desired outcome. It is either a positive integer or -1 if no element in the axtree clearly matches the goal."
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
