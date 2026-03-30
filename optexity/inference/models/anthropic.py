import base64
import logging
import os
from pathlib import Path
from typing import Optional

import anthropic
from pydantic import BaseModel, ValidationError

from .llm_model import AnthropicModels, LLMModel, TokenUsage

logger = logging.getLogger(__name__)

MAX_TOKENS = 4096


class Anthropic(LLMModel):

    def __init__(self, model_name: AnthropicModels, use_structured_output: bool):
        super().__init__(model_name, use_structured_output)

        try:
            api_key = os.environ["ANTHROPIC_API_KEY"]
            self.client = anthropic.Anthropic(api_key=api_key)
        except KeyError:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    def _get_model_response_with_structured_output(
        self,
        prompt: str,
        response_schema: type[BaseModel],
        screenshot: Optional[str] = None,
        pdf_url: Optional[str | Path] = None,
        system_instruction: Optional[str] = None,
    ) -> tuple[BaseModel, TokenUsage]:

        if pdf_url is not None and screenshot is not None:
            raise ValueError("Cannot use both screenshot and pdf_url")

        if screenshot is not None:
            content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": screenshot,
                    },
                },
                {"type": "text", "text": prompt},
            ]
        else:
            content = prompt

        token_usage = TokenUsage()
        parsed_response = None

        kwargs = {
            "model": self.model_name.value,
            "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": content}],
        }
        if system_instruction:
            kwargs["system"] = system_instruction

        try:
            if self.use_structured_output:
                import anthropic

                kwargs["tools"] = [
                    {
                        "name": "structured_output",
                        "description": "Return the structured response",
                        "input_schema": response_schema.model_json_schema(),
                    }
                ]
                kwargs["tool_choice"] = {"type": "tool", "name": "structured_output"}

                response = self.client.messages.create(**kwargs)

                for block in response.content:
                    if block.type == "tool_use" and block.name == "structured_output":
                        parsed_response = response_schema.model_validate(block.input)
                        break
            else:
                response = self.client.messages.create(**kwargs)
                text = response.content[0].text
                parsed_response = self.parse_from_completion(text, response_schema)

            if response.usage is not None:
                token_usage = self.get_token_usage(
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                )
        except ValidationError as e:
            logger.error(f"ValidationError in Anthropic model response: {e}")

        return parsed_response, token_usage

    def _get_model_response(
        self, prompt: str, system_instruction: Optional[str] = None
    ) -> tuple[str, TokenUsage]:

        kwargs = {
            "model": self.model_name.value,
            "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_instruction:
            kwargs["system"] = system_instruction

        response = self.client.messages.create(**kwargs)

        token_usage = TokenUsage()
        if response.usage is not None:
            token_usage = self.get_token_usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )

        return response.content[0].text, token_usage

    def _get_computer_use_model_response(
        self,
        prompt: str,
        screenshot: str,
        system_instruction: Optional[str] = None,
    ) -> tuple[tuple[int, int] | None, TokenUsage]:

        if screenshot is None:
            raise ValueError("Screenshot is required")

        content = [
            {"type": "text", "text": prompt},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": screenshot,
                },
            },
        ]

        token_usage = TokenUsage()
        coordinates = None

        try:
            kwargs = {
                "model": self.model_name.value,
                "max_tokens": MAX_TOKENS,
                "messages": [{"role": "user", "content": content}],
                "tools": [
                    {
                        "type": "computer_20251124",
                        "name": "computer",
                        "display_width_px": 1440,
                        "display_height_px": 900,
                        "display_number": 1,
                    }
                ],
            }
            if system_instruction:
                kwargs["system"] = system_instruction

            response = self.client.beta.messages.create(
                **kwargs,
                betas=["computer-use-2025-11-24"],
            )

            for block in response.content:
                if block.type == "tool_use" and block.name == "computer":
                    action = block.input.get("action")
                    if action in ("click", "left_click", "mouse_move"):
                        coord = block.input.get("coordinate")
                        if coord and len(coord) == 2:
                            coordinates = (int(coord[0]), int(coord[1]))
                            break

            if response.usage is not None:
                token_usage = self.get_token_usage(
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                )
        except ValidationError as e:
            logger.error(f"ValidationError in Anthropic computer use response: {e}")

        return coordinates, token_usage
