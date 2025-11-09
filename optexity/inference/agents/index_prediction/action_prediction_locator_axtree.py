import logging
from typing import Optional

from pydantic import BaseModel, Field

from optexity.inference.agents.index_prediction.prompt import system_prompt
from optexity.inference.models import GeminiModels, get_llm_model

logger = logging.getLogger(__name__)


class IndexPredictionOutput(BaseModel):
    index: int = Field(
        description="The index of the interactive element in the axtree that would achieve the desired outcome. It is always greater than 0."
    )


class ActionPredictionLocatorAxtree:
    def __init__(self):
        self.model = get_llm_model(GeminiModels.GEMINI_2_5_FLASH, True)

    def predict_action(self, goal: str, axtree: str, screenshot: Optional[str] = None):

        prompt = f"""
        [INSTRUCTIONS]
        {system_prompt}
        [/INSTRUCTIONS]

        [INPUT]
        Goal: {goal}

        [AXTREE] 
        {axtree} 
        [/AXTREE]

        [/INPUT]
        """

        response, token_usage = self.model.get_model_response_with_structured_output(
            prompt,
            IndexPredictionOutput,
            screenshot,
        )

        index = response.index

        return index, token_usage
