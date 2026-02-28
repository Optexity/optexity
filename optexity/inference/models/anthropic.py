import base64
import logging
import os
from pathlib import Path
from typing import TypeVar
from typing_extensions import override

import httpx
from anthropic import Anthropic as AnthropicClient, Omit
from anthropic.types import (
    ContentBlockParam,
    DocumentBlockParam,
    ImageBlockParam,
    TextBlockParam,
)
from pydantic import BaseModel, ValidationError

from optexity.schema.token_usage import TokenUsage

from .llm_model import AnthropicModels, LLMModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class Anthropic(LLMModel):
    api_key: str
    client: AnthropicClient

    def __init__(self, model_name: AnthropicModels, use_structured_output: bool):
        super().__init__(model_name, use_structured_output)
        self.api_key = os.environ["ANTHROPIC_API_KEY"]

        try:
            self.client = AnthropicClient(api_key=self.api_key)
            _ = self.client.models.list()  # Test connection

        except Exception:
            raise ValueError("Invalid ANTHROPIC_API_KEY")

    @override
    def _get_model_response_with_structured_output(
        self,
        prompt: str,
        response_schema: type[T],
        screenshot: str | None = None,
        pdf_url: str | Path | None = None,
        system_instruction: str | None = "",
    ) -> tuple[T, TokenUsage]:

        if pdf_url is not None and screenshot is not None:
            raise ValueError("Cannot use both screenshot and pdf_url")

        system = system_instruction if system_instruction is not None else Omit()

        content: list[ContentBlockParam] = []
        text_block: TextBlockParam = {"type": "text", "text": prompt}
        content.append(text_block)

        if screenshot is not None:
            image_block: ImageBlockParam = {
                "type": "image",
                "source": {
                    "data": screenshot,
                    "media_type": "image/png",
                    "type": "base64",
                },
            }
            content.append(image_block)

        if pdf_url is not None:
            pdf_data = httpx.get(
                pdf_url.as_uri() if isinstance(pdf_url, Path) else pdf_url
            ).content
            doc_block: DocumentBlockParam = {
                "type": "document",
                "source": {
                    "type": "base64",
                    "data": base64.b64encode(pdf_data).decode("utf-8"),
                    "media_type": "application/pdf",
                },
            }
            content.append(doc_block)

        try:
            response = self.client.messages.parse(
                model=self.model_name.value,
                max_tokens=8192,
                system=system,
                messages=[{"role": "user", "content": content}],
                output_format=response_schema,
            )
            parsed_response = response_schema.model_validate(response.parsed_output)

        except ValidationError as e:
            raise e  # or wrap it — but don't return None

        token_usage = self.get_token_usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

        return parsed_response, token_usage

    @override
    def _get_model_response(
        self, prompt: str, system_instruction: str | None = None
    ) -> tuple[str, TokenUsage]:
        response = self.client.messages.create(
            model=self.model_name.value,
            max_tokens=8192,
            system=system_instruction if system_instruction is not None else Omit(),
            messages=[{"role": "user", "content": prompt}],
        )

        response_content = response.content[0]
        if response_content.type != "text":
            raise ValueError("Model did not return text content")

        token_usage = self.get_token_usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

        return response_content.text, token_usage
