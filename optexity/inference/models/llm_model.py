import ast
import logging
import re
import time
from pathlib import Path
from typing import Optional

import litellm
from pydantic import BaseModel

from optexity.schema.token_usage import TokenUsage

logger = logging.getLogger(__name__)


def extract_json_objects(text: str) -> list[str]:
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


def _recover_partial_output(data: dict, schema: type[BaseModel]) -> BaseModel:
    """Recover when a model embeds state fields inside action items.

    Some models return:
      {"action": [{"thinking": "...", "evaluation_previous_goal": "...", "click": {...}}]}
    instead of:
      {"evaluation_previous_goal": "...", "action": [{"click": {...}}]}

    This function extracts string fields that belong at the top level from
    inside the first action item, promotes them, strips them from action items,
    and fills any remaining missing string fields with empty defaults.
    """
    # Collect names of string-typed fields in the schema
    top_str_fields: set[str] = set()
    for name, field in schema.model_fields.items():
        ann = field.annotation
        args = getattr(ann, "__args__", None)
        if ann is str or (args is not None and str in args):
            top_str_fields.add(name)

    recovered: dict = {}
    cleaned_actions: list = []
    for item in data.get("action", []):
        if not isinstance(item, dict):
            cleaned_actions.append(item)
            continue
        clean: dict = {}
        for k, v in item.items():
            if k in top_str_fields and k not in data:
                recovered[k] = v
            elif k == "thinking" and "thinking" not in schema.model_fields:
                # model adds free-form thinking inside action — promote or discard
                if "thinking" in top_str_fields:
                    recovered.setdefault("thinking", v)
            else:
                clean[k] = v
        cleaned_actions.append(clean)

    new_data = dict(data)
    new_data.update(recovered)
    new_data["action"] = cleaned_actions

    # Fill any still-missing string fields with empty defaults
    for name in top_str_fields:
        if name not in new_data:
            new_data[name] = ""

    return schema.model_validate(new_data)


def parse_json_from_completion(
    content: str, response_schema: type[BaseModel]
) -> BaseModel:
    """Recover a schema instance from a completion that isn't clean JSON.

    Covers providers that wrap JSON in markdown fences or prose, and those that
    drop `response_format` altogether.
    """
    patterns = [r"```json\n(.*?)\n```"]
    json_blocks = []
    for pattern in patterns:
        json_blocks += re.findall(pattern, content, re.DOTALL)
    json_blocks += extract_json_objects(content)
    for block in json_blocks:
        block = block.strip()
        try:
            return response_schema.model_validate_json(block)
        except Exception:
            try:
                return response_schema.model_validate(ast.literal_eval(block))
            except Exception:
                pass
        # Fallback: provider returned valid JSON but misplaced state fields.
        # Promote string fields from inside action items and fill empty defaults.
        try:
            import json as _json
            data = _json.loads(block)
            if isinstance(data, dict):
                return _recover_partial_output(data, response_schema)
        except Exception:
            continue

    raise ValueError("Could not parse response from completion.")


class LLMModel:
    def __init__(self, model_name: str, use_structured_output: bool):

        self.model_name = model_name
        self.use_structured_output = use_structured_output

    def _get_model_response(
        self, prompt: str, system_instruction: Optional[str] = None
    ) -> tuple[str, TokenUsage]:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def _get_model_response_with_structured_output(
        self,
        prompt: str,
        response_schema: type[BaseModel],
        screenshot: Optional[str] = None,
        pdf_url: Optional[str | Path] = None,
        system_instruction: Optional[str] = None,
    ) -> tuple[BaseModel, TokenUsage]:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_model_response(
        self, prompt: str, system_instruction: Optional[str] = None
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
        screenshot: Optional[str] = None,
        pdf_url: Optional[str | Path] = None,
        system_instruction: Optional[str] = None,
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
                    time.sleep(5)
                last_exception = str(e)

        raise Exception(
            "Max retries exceeded for LLM with structured output"
            + "\n"
            + last_exception
        )

    def extract_json_objects(self, text):
        return extract_json_objects(text)

    def parse_from_completion(
        self, content: str, response_schema: type[BaseModel]
    ) -> BaseModel:
        return parse_json_from_completion(content, response_schema)

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

        # litellm already counts reasoning/thinking inside completion_tokens, so
        # thoughts and tool-use are reported as tokens but never priced separately —
        # doing so would double-bill them.
        tool_use_cost = thoughts_cost = 0.0
        try:
            input_cost, output_cost = litellm.cost_per_token(
                model=self.model_name,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
            )
        except Exception as e:
            logger.warning(
                f"Model {self.model_name} has no litellm pricing data ({e}). "
                f"Cost will be reported as 0."
            )
            input_cost = output_cost = 0.0
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
