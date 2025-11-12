import ast
import logging
import re
import time
from enum import Enum, unique
from typing import Optional

import tokencost.costs
from pydantic import BaseModel

from optexity.schema.token_usage import TokenUsage

logger = logging.getLogger(__name__)


@unique
class HumanModels(Enum):
    TERMINAL_INPUT = "terminal-input"


@unique
class GeminiModels(Enum):
    GEMINI_1_5_FLASH = "gemini-1.5-flash"
    GEMINI_2_0_FLASH = "gemini-2.0-flash"
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_2_5_FLASH_LITE = "gemini-2.5-flash-lite-preview-06-17"
    GEMINI_2_5_PRO = "gemini-2.5-pro"


@unique
class OpenAIModels(Enum):
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4_1 = "gpt-4.1"
    GPT_4_1_MINI = "gpt-4.1-mini"


class LLMModel:
    def __init__(
        self,
        model_name: GeminiModels | HumanModels | OpenAIModels,
        use_structured_output: bool,
    ):

        self.model_name = model_name
        self.use_structured_output = use_structured_output

    def _get_model_response(self, prompt: str) -> tuple[str, TokenUsage]:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def _get_model_response_with_structured_output(
        self,
        prompt: str,
        response_schema: BaseModel,
        screenshot: Optional[str] = None,
        pdf_url: Optional[str] = None,
        system_instruction: Optional[str] = None,
    ) -> tuple[BaseModel, TokenUsage]:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_model_response(self, prompt: str) -> tuple[str, TokenUsage]:

        max_retries = 3
        for i in range(max_retries):
            try:
                return self._get_model_response(prompt)
            except Exception as e:
                logger.error(f"LLM Error during inference: {e}")
                if i < max_retries - 1:
                    logger.info(f"Retrying... {i + 1}/{max_retries}")
                    time.sleep(5)
                continue
        raise Exception("Max retries exceeded for LLM")

    def get_model_response_with_structured_output(
        self,
        prompt: str,
        response_schema: BaseModel,
        screenshot: Optional[str] = None,
        pdf_url: Optional[str] = None,
        system_instruction: Optional[str] = None,
    ) -> tuple[BaseModel, TokenUsage]:

        max_retries = 3
        last_exception = ""
        for i in range(max_retries):
            try:
                # raise Exception("Test error")
                return self._get_model_response_with_structured_output(
                    prompt=prompt,
                    response_schema=response_schema,
                    screenshot=screenshot,
                    pdf_url=pdf_url,
                    system_instruction=system_instruction,
                )
            except Exception as e:
                logger.error(f"LLM with structured output Error during inference: {e}")
                if i < max_retries - 1:
                    logger.info(f"Retrying... {i + 1}/{max_retries}")
                    time.sleep(20)
                last_exception = str(e)
                continue
        raise Exception(
            "Max retries exceeded for LLM with structured output"
            + "\n"
            + last_exception
        )

    def extract_json_objects(self, text):
        stack = []  # Stack to track `{` positions
        json_candidates = []  # Potential JSON substrings

        # Iterate through the text to find balanced { }
        for i, char in enumerate(text):
            if char == "{":
                stack.append(i)  # Store index of '{'
            elif char == "}" and stack:
                start = stack.pop()  # Get the last unmatched '{'
                json_candidates.append(text[start : i + 1])  # Extract substring

        return json_candidates

    def parse_from_completion(
        self, content: str, response_schema: BaseModel
    ) -> BaseModel:
        patterns = [r"```json\n(.*?)\n```"]
        json_blocks = []
        for pattern in patterns:
            json_blocks += re.findall(pattern, content, re.DOTALL)
        json_blocks += self.extract_json_objects(content)
        for block in json_blocks:
            block = block.strip()
            try:
                response = response_schema.model_validate_json(block)
                return response
            except Exception as e:
                try:
                    block_dict = ast.literal_eval(block)
                    response = response_schema.model_validate(block_dict)
                    return response
                except Exception as e:
                    continue

        raise ValueError("Could not parse response from completion.")

    def get_token_usage(self, input_tokens: int, output_tokens: int) -> TokenUsage:
        input_cost = tokencost.costs.calculate_cost_by_tokens(
            model=self.model_name.value,
            num_tokens=input_tokens,
            token_type="input",
        )
        output_cost = tokencost.costs.calculate_cost_by_tokens(
            model=self.model_name.value,
            num_tokens=output_tokens,
            token_type="output",
        )
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=input_cost + output_cost,
        )
