from typing import Optional

from pydantic import BaseModel, Field

from optexity.inference.agents.input_text_prediction.prompt import system_prompt
from optexity.inference.models.llm_model import LLMModel
from optexity.schema.token_usage import TokenUsage


class InputTextPredictionOutput(BaseModel):
    input_text: str = Field(
        description="The exact string to type into the target input field."
    )


class InputTextPredictionAgent:
    def __init__(self, model: LLMModel):
        self.model = model

    def predict_input_text(
        self,
        goal: str,
        axtree: str,
        screenshot: Optional[str] = None,
    ) -> tuple[str, InputTextPredictionOutput, TokenUsage]:

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
            response_schema=InputTextPredictionOutput,
            screenshot=screenshot,
            system_instruction=system_prompt,
        )

        return final_prompt, response, token_usage
