import json
import logging

from pydantic import BaseModel, Field

from optexity.inference.agents.select_value_prediction.prompt import system_prompt
from optexity.inference.models import GeminiModels, get_llm_model
from optexity.schema.token_usage import TokenUsage

logger = logging.getLogger(__name__)


class SelectValuePredictionOutput(BaseModel):
    matched_values: list[str] = Field(default_factory=list)


class SelectValuePredictionAgent:
    def __init__(self):
        self.model = get_llm_model(GeminiModels.GEMINI_2_5_FLASH)

    def predict_select_value(
        self, options: list[dict[str, str]], patterns: list[str]
    ) -> tuple[str, SelectValuePredictionOutput, TokenUsage]:

        final_prompt = f"""
        [Actual Select Options]
        {json.dumps(options, indent=4)}

        [User Provided Patterns]
        [{", ".join(patterns)}]
        """

        response, token_usage = self.model.get_model_response_with_structured_output(
            prompt=final_prompt,
            response_schema=SelectValuePredictionOutput,
            system_instruction=system_prompt,
        )

        return final_prompt, response, token_usage
