import base64
import logging
import os
from pathlib import Path
from typing import Optional

import httpx
from openai import OpenAI as OpenAIClient
from pydantic import BaseModel, ValidationError

from optexity.utils.utils import is_local_path, is_url

from .llm_model import LLMModel, OpenAIModels, TokenUsage, register_model

logger = logging.getLogger(__name__)


@register_model(OpenAIModels)
class OpenAI(LLMModel):

    def __init__(self, model_name: OpenAIModels, use_structured_output: bool):
        super().__init__(model_name, use_structured_output)

        self.api_key = os.environ["OPENAI_API_KEY"]
        self.client = OpenAIClient(api_key=self.api_key)

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

        messages = []

        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})

        if screenshot is not None:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{screenshot}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            )
        elif pdf_url is not None:
            if is_local_path(pdf_url):
                pdf_data = base64.standard_b64encode(
                    Path(str(pdf_url)).read_bytes()
                ).decode("utf-8")
            elif is_url(pdf_url):
                pdf_data = base64.standard_b64encode(
                    httpx.get(str(pdf_url)).content
                ).decode("utf-8")
            else:
                raise ValueError(f"Invalid pdf_url: {pdf_url}")

            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "file",
                            "file": {
                                "filename": "document.pdf",
                                "file_data": f"data:application/pdf;base64,{pdf_data}",
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            )
        else:
            messages.append({"role": "user", "content": prompt})

        parsed_response = None
        token_usage = TokenUsage()

        try:
            if self.use_structured_output:
                response = self.client.chat.completions.parse(
                    model=self.model_name.value,
                    messages=messages,
                    response_format=response_schema,
                )
                parsed_response = response.choices[0].message.parsed
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name.value,
                    messages=messages,
                )
                content = response.choices[0].message.content or ""
                parsed_response = self.parse_from_completion(content, response_schema)

            if response.usage is not None:
                token_usage = self.get_token_usage(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )
        except ValidationError as e:
            logger.error(f"ValidationError in OpenAI model response: {e}")

        return parsed_response, token_usage

    def _get_model_response(
        self, prompt: str, system_instruction: Optional[str] = None
    ) -> tuple[str, TokenUsage]:

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model_name.value,
            messages=messages,
        )
        content = response.choices[0].message.content or ""
        if response.usage is not None:
            token_usage = self.get_token_usage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )
        else:
            token_usage = TokenUsage()
        return content, token_usage
