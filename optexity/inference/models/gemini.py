import base64
import logging
import os
from pathlib import Path
from typing import Optional

import httpx
from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

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
