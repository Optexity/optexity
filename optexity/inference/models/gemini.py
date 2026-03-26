import base64
import logging
import os
from pathlib import Path
from typing import Optional

import httpx
from google import genai
from google.genai import types
from google.genai.types import Content, Part
from pydantic import BaseModel, ValidationError

from optexity.utils.settings import settings
from optexity.utils.utils import is_local_path, is_url

from .llm_model import GeminiModels, LLMModel, TokenUsage

logger = logging.getLogger(__name__)


class Gemini(LLMModel):

    def __init__(self, model_name: GeminiModels, use_structured_output: bool):
        super().__init__(model_name, use_structured_output)

        self.api_key = os.environ["GOOGLE_API_KEY"]
        try:
            self.client = genai.Client(api_key=self.api_key)
            self.client.models.list()
        except Exception as e:
            raise ValueError("Invalid GOOGLE_API_KEY")

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

        final_prompt = prompt

        if screenshot is not None:
            final_prompt = [
                types.Part.from_bytes(
                    data=base64.b64decode(screenshot),
                    mime_type="image/png",
                ),
                prompt,
            ]
        if pdf_url is not None:
            if is_local_path(pdf_url):
                final_prompt = [
                    types.Part.from_bytes(
                        data=Path(str(pdf_url)).read_bytes(),
                        mime_type="application/pdf",
                    ),
                    prompt,
                ]
            elif is_url(pdf_url):
                doc_data = httpx.get(str(pdf_url)).content
                final_prompt = [
                    types.Part.from_bytes(
                        data=doc_data,
                        mime_type="application/pdf",
                    ),
                    prompt,
                ]
        response = None
        parsed_response = None
        token_usage = TokenUsage()

        try:
            if self.use_structured_output:
                response = self.client.models.generate_content(
                    model=self.model_name.value,
                    contents=final_prompt,
                    config={
                        "response_mime_type": "application/json",
                        "system_instruction": system_instruction,
                        "response_json_schema": response_schema.model_json_schema(),
                    },
                )

                if isinstance(response.parsed, BaseModel):
                    parsed_response = response.parsed
                else:
                    parsed_response = response_schema.model_validate(response.parsed)
            else:
                response = self.client.models.generate_content(
                    model=self.model_name.value,
                    contents=final_prompt,
                    config={"system_instruction": system_instruction},
                )

                parsed_response: BaseModel = self.parse_from_completion(
                    str(response.candidates[0].content.parts[0].text), response_schema
                )

            if response.usage_metadata is not None:
                token_usage = self.get_token_usage(
                    input_tokens=response.usage_metadata.prompt_token_count,
                    output_tokens=response.usage_metadata.candidates_token_count,
                    tool_use_tokens=response.usage_metadata.tool_use_prompt_token_count,
                    thoughts_tokens=response.usage_metadata.thoughts_token_count,
                    total_tokens=response.usage_metadata.total_token_count,
                )
        except ValidationError as e:
            logger.error(f"ValidationError in Gemini model response: {e}")
            pass

        return parsed_response, token_usage

    def _get_model_response(
        self, prompt: str, system_instruction: Optional[str] = None
    ) -> tuple[str, TokenUsage]:

        response = self.client.models.generate_content(
            model=self.model_name.value,
            contents=prompt,
            config={"system_instruction": system_instruction},
        )
        if response.usage_metadata is not None:
            token_usage = self.get_token_usage(
                input_tokens=response.usage_metadata.prompt_token_count,
                output_tokens=response.usage_metadata.candidates_token_count,
            )
        else:
            token_usage = TokenUsage()
        return str(response.candidates[0].content.parts[0].text), token_usage

    def _get_computer_use_model_response(
        self,
        prompt: str,
        screenshot: str,
        system_instruction: Optional[str] = None,
    ) -> tuple[tuple[int, int] | None, TokenUsage]:

        if screenshot is None:
            raise ValueError("Screenshot is required")

        contents = [
            Content(
                role="user",
                parts=[
                    Part(text=prompt),
                    Part.from_bytes(
                        data=base64.b64decode(screenshot), mime_type="image/png"
                    ),
                ],
            )
        ]

        response = None
        coordinates = None
        token_usage = TokenUsage()

        try:
            excluded_predefined_functions = [
                "open_web_browser",
                "wait_5_seconds",
                "go_back",
                "go_forward",
                "search",
                "navigate",
                "hover_at",
                "key_combination",
                "scroll_document",
                "scroll_at",
                "drag_and_drop",
            ]
            generate_content_config = genai.types.GenerateContentConfig(
                tools=[
                    # 1. Computer Use tool with browser environment
                    types.Tool(
                        computer_use=types.ComputerUse(
                            environment=types.Environment.ENVIRONMENT_BROWSER,
                            # Optional: Exclude specific predefined functions
                            excluded_predefined_functions=excluded_predefined_functions,
                        )
                    ),
                    # 2. Optional: Custom user-defined functions
                    # types.Tool(
                    # function_declarations=custom_functions
                    #   )
                ],
                thinking_config=types.ThinkingConfig(include_thoughts=True),
            )

            response = self.client.models.generate_content(
                model=self.model_name.value,
                contents=contents,
                config=generate_content_config,
            )

            candidate = response.candidates[0]

            has_function_calls = any(
                part.function_call for part in candidate.content.parts
            )
            if not has_function_calls:
                text_response = " ".join(
                    [part.text for part in candidate.content.parts if part.text]
                )
                print("Agent finished:", text_response)
                return None, token_usage

            print("Executing actions...")
            results = self.execute_function_calls(
                candidate, screenshot, settings.SCREEN_WIDTH, settings.SCREEN_HEIGHT
            )

            # if response.usage_metadata is not None:
            #     token_usage = self.get_token_usage(
            #         input_tokens=response.usage_metadata.prompt_token_count,
            #         output_tokens=response.usage_metadata.candidates_token_count,
            #         tool_use_tokens=response.usage_metadata.tool_use_prompt_token_count,
            #         thoughts_tokens=response.usage_metadata.thoughts_token_count,
            #         total_tokens=response.usage_metadata.total_token_count,
            #     )
            return results, token_usage
        except ValidationError as e:
            logger.error(f"ValidationError in Gemini model response: {e}")
            pass

        return None, TokenUsage()

    def denormalize_x(self, x: int, screen_width: int) -> int:
        """Convert normalized x coordinate (0-1000) to actual pixel coordinate."""
        return int(x / 1000 * screen_width)

    def denormalize_y(self, y: int, screen_height: int) -> int:
        """Convert normalized y coordinate (0-1000) to actual pixel coordinate."""
        return int(y / 1000 * screen_height)

    def execute_function_calls(self, candidate, page, screen_width, screen_height):
        results = []
        function_calls = []
        for part in candidate.content.parts:
            if part.function_call:
                function_calls.append(part.function_call)

        for function_call in function_calls:
            action_result = {}
            fname = function_call.name
            args = function_call.args
            actual_x = self.denormalize_x(args["x"], screen_width)
            actual_y = self.denormalize_y(args["y"], screen_height)

            return (actual_x, actual_y)
        return None
