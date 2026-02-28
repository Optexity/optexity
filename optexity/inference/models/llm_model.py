import ast
import logging
import re
import time
from enum import Enum, unique
from pathlib import Path
from typing import Optional

import tokencost.costs
from pydantic import BaseModel, ValidationError

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


@unique
class AnthropicModels(Enum):
    CLAUDE_SONNET_4_6 = "claude-sonnet-4-6"
    CLAUDE_OPUS_4_6 = "claude-opus-4-6"
    CLAUDE_HAIKU_4_5 = "claude-haiku-4-5-20251001"


class LLMModel:
    def __init__(
        self,
        model_name: GeminiModels | HumanModels | OpenAIModels | AnthropicModels,
        use_structured_output: bool,
    ):

        self.model_name = model_name
        self.use_structured_output = use_structured_output

    def _get_model_response(
        self, prompt: str, system_instruction: str | None = None
    ) -> tuple[str, TokenUsage]:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def _get_model_response_with_structured_output(
        self,
        prompt: str,
        response_schema: type[BaseModel],
        screenshot: str | None = None,
        pdf_url: Optional[str | Path] = None,
        system_instruction: str | None = None,
    ) -> tuple[BaseModel, TokenUsage]:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_model_response(
        self, prompt: str, system_instruction: str | None = None
    ) -> tuple[str, TokenUsage]:

        max_retries = 3
        for i in range(max_retries):
            try:
                return self._get_model_response(prompt, system_instruction)
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
        response_schema: type[BaseModel],
        screenshot: str | None = None,
        pdf_url: str | Path | None = None,
        system_instruction: str | None = None,
    ) -> tuple[BaseModel, TokenUsage]:

        total_token_usage = TokenUsage()
        max_retries = 3
        last_exception = ""
        for i in range(max_retries):
            try:
                # raise Exception("Test error")
                parsed_response, token_usage = (
                    self._get_model_response_with_structured_output(
                        prompt=prompt,
                        response_schema=response_schema,
                        screenshot=screenshot,
                        pdf_url=pdf_url,
                        system_instruction=system_instruction,
                    )
                )
                total_token_usage += token_usage
                if parsed_response is not None:
                    return parsed_response, total_token_usage
            except Exception as e:
                logger.error(f"LLM with structured output Error during inference: {e}")
                if i < max_retries - 1:
                    logger.info(f"Retrying... {i + 1}/{max_retries}")
                    time.sleep(20)
                last_exception = str(e)

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
        self, content: str, response_schema: type[BaseModel]
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

        raise ValidationError("Could not parse response from completion.")

    def get_token_usage(
        self,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        tool_use_tokens: int | None = None,
        thoughts_tokens: int | None = None,
        total_tokens: Optional[int] = None,
    ) -> TokenUsage:
        if input_tokens is None:
            input_tokens = 0
        if output_tokens is None:
            output_tokens = 0
        if tool_use_tokens is None:
            tool_use_tokens = 0
        if thoughts_tokens is None:
            thoughts_tokens = 0
        if total_tokens is None:
            total_tokens = 0
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
        tool_use_cost = tokencost.costs.calculate_cost_by_tokens(
            model=self.model_name.value,
            num_tokens=tool_use_tokens,
            token_type="output",
        )
        thoughts_cost = tokencost.costs.calculate_cost_by_tokens(
            model=self.model_name.value,
            num_tokens=thoughts_tokens,
            token_type="output",
        )
        calculated_total_tokens = (
            input_tokens + output_tokens + tool_use_tokens + thoughts_tokens
        )
        total_cost = input_cost + output_cost + tool_use_cost + thoughts_cost
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_use_tokens=tool_use_tokens,
            thoughts_tokens=thoughts_tokens,
            total_tokens=total_tokens,
            calculated_total_tokens=calculated_total_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            tool_use_cost=tool_use_cost,
            thoughts_cost=thoughts_cost,
            total_cost=total_cost,
        )
