import logging
from typing import Optional

from pydantic import BaseModel, Field

from optexity.inference.agents.index_prediction.prompt import system_prompt
from optexity.inference.agents import get_agent_model
from optexity.schema.token_usage import TokenUsage

logger = logging.getLogger(__name__)


class IndexPredictionOutput(BaseModel):
    index: int = Field(
        description="The index of the interactive element in the axtree that would achieve the desired outcome. It is always greater than 0."
    )


class ActionPredictionLocatorAxtree:
    def __init__(self):
        self.model = get_agent_model(True)

    def predict_action(
        self, goal: str, axtree: str, screenshot: Optional[str] = None
    ) -> tuple[str, IndexPredictionOutput, TokenUsage]:

        final_prompt = f"""
        [INPUT]
        Goal: {goal}

        [AXTREE] 
        {axtree} 
        [/AXTREE]

        [/INPUT]
        """

        response, token_usage = self.model.get_model_response_with_structured_output(
            prompt=final_prompt,
            response_schema=IndexPredictionOutput,
            screenshot=screenshot,
            system_instruction=system_prompt,
        )

        return final_prompt, response, token_usage
