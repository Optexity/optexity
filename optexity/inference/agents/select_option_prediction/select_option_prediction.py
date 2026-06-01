from typing import Optional

from pydantic import BaseModel, Field

from optexity.inference.agents.select_option_prediction.prompt import system_prompt
from optexity.inference.models.llm_model import LLMModel
from optexity.schema.token_usage import TokenUsage


class SelectOptionPredictionOutput(BaseModel):
    select_values: list[str] = Field(
        description=(
            "Strings identifying which dropdown option(s) to select; "
            "matched later against option value and label."
        )
    )


class SelectOptionPredictionAgent:
    def __init__(self, model: LLMModel):
        self.model = model

    def predict_select_option(
        self,
        goal: str,
        axtree: str,
        screenshot: Optional[str] = None,
    ) -> tuple[str, SelectOptionPredictionOutput, TokenUsage]:

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
            response_schema=SelectOptionPredictionOutput,
            screenshot=screenshot,
            system_instruction=system_prompt,
        )

        return final_prompt, response, token_usage
