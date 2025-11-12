import base64
import logging
import os
from typing import Optional

import httpx
from google import genai
from google.genai import types
from pydantic import BaseModel

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
        response_schema: BaseModel,
        screenshot: Optional[str] = None,
        pdf_url: Optional[str] = None,
        system_instruction: Optional[str] = None,
    ) -> tuple[BaseModel, TokenUsage]:

        if pdf_url is not None and screenshot is not None:
            raise ValueError("Cannot use both screenshot and pdf_url")

        if screenshot is not None:
            prompt = [
                types.Part.from_bytes(
                    data=base64.b64decode(screenshot),
                    mime_type="image/png",
                ),
                prompt,
            ]
        if pdf_url is not None:
            doc_data = httpx.get(pdf_url).content
            prompt = [
                types.Part.from_bytes(
                    data=doc_data,
                    mime_type="application/pdf",
                ),
                prompt,
            ]
        if self.use_structured_output:
            response = self.client.models.generate_content(
                model=self.model_name.value,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "system_instruction": system_instruction,
                    "response_json_schema": response_schema.model_json_schema(),
                },
            )
            parsed_response: response_schema = response.parsed
        else:
            response = self.client.models.generate_content(
                model=self.model_name.value, contents=prompt
            )

            parsed_response: response_schema = self.parse_from_completion(
                response.candidates[0].content.parts[0].text, response_schema
            )

        token_usage = self.get_token_usage(
            input_tokens=response.usage_metadata.prompt_token_count,
            output_tokens=response.usage_metadata.candidates_token_count,
        )
        return parsed_response, token_usage

    def _get_model_response(self, prompt: str) -> tuple[str, TokenUsage]:

        response = self.client.models.generate_content(
            model=self.model_name.value,
            contents=prompt,
        )
        token_usage = self.get_token_usage(
            input_tokens=response.usage_metadata.prompt_token_count,
            output_tokens=response.usage_metadata.candidates_token_count,
        )
        return response.candidates[0].content.parts[0].text, token_usage
