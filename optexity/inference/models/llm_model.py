import ast
import base64
import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

import httpx
from pydantic import BaseModel

from optexity.schema.token_usage import TokenUsage
from optexity.utils.utils import is_local_path, is_url

from .llm import Tier, complete

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
                continue

    raise ValueError("Could not parse response from completion.")


_SPACE_PLACEHOLDER = "_._"


def _sniff_image_media_type(screenshot_b64: str) -> str:
    """Detect the real encoding of a base64 screenshot from its magic bytes.

    Screenshots reach here from more than one capture path (our own
    Playwright page.screenshot(), which defaults to PNG, vs. browser_use's
    ScreenshotWatchdog, which explicitly captures JPEG) — declaring a fixed
    media type regardless of the source causes providers like Anthropic to
    reject the image outright when it doesn't match.
    """
    try:
        header = base64.b64decode(screenshot_b64[:16])
    except Exception:
        return "png"
    if header.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    return "png"


def _sanitize_schema_keys(obj):
    """Recursively replace spaces in dict keys with _._

    Anthropic rejects schemas with spaces in property names, and extraction
    schemas come from user-authored workflow JSON where spaces are common.
    """
    if isinstance(obj, dict):
        return {
            k.replace(" ", _SPACE_PLACEHOLDER): (
                # "required" names properties, so it must match the renamed keys.
                [n.replace(" ", _SPACE_PLACEHOLDER) for n in v]
                if k == "required" and isinstance(v, list)
                else _sanitize_schema_keys(v)
            )
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [_sanitize_schema_keys(item) for item in obj]
    return obj


def _restore_schema_keys(obj):
    """Recursively replace _._ in dict keys back to spaces."""
    if isinstance(obj, dict):
        return {
            k.replace(_SPACE_PLACEHOLDER, " "): _restore_schema_keys(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [_restore_schema_keys(item) for item in obj]
    return obj


def _pdf_to_base64(pdf_url: str | Path) -> str:
    if is_local_path(pdf_url):
        raw = Path(str(pdf_url)).read_bytes()
    elif is_url(pdf_url):
        raw = httpx.get(str(pdf_url)).content
    else:
        raise ValueError(f"Invalid pdf_url: {pdf_url}")
    return base64.standard_b64encode(raw).decode("utf-8")


def _build_messages(
    prompt: str,
    system_instruction: Optional[str] = None,
    screenshot: Optional[str] = None,
    pdf_url: Optional[str | Path] = None,
) -> list[dict[str, Any]]:

    if pdf_url is not None and screenshot is not None:
        raise ValueError("Cannot use both screenshot and pdf_url")

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]

    if screenshot is not None:
        media_type = _sniff_image_media_type(screenshot)
        content.insert(
            0,
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/{media_type};base64,{screenshot}"},
            },
        )
    elif pdf_url is not None:
        content.insert(
            0,
            {
                "type": "file",
                "file": {
                    "file_data": (
                        f"data:application/pdf;base64,{_pdf_to_base64(pdf_url)}"
                    )
                },
            },
        )

    messages: list[dict[str, Any]] = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": content})
    return messages


class LLMModel:
    """An LLM call target: a model ID (action / task override), a `Tier`
    (hardcoded call sites) or None for the configured default. Retries, fallback
    and cost tracking happen in `optexity.inference.models.llm`."""

    def __init__(self, model_name: str | Tier | None, use_structured_output: bool):

        self.model_name = model_name
        self.use_structured_output = use_structured_output

    def get_model_response_with_structured_output(
        self,
        prompt: str,
        response_schema: type[BaseModel],
        screenshot: Optional[str] = None,
        pdf_url: Optional[str | Path] = None,
        system_instruction: Optional[str] = None,
        feature: str = "unknown",
    ) -> tuple[BaseModel, TokenUsage]:

        def parse(content: str) -> BaseModel:
            if self.use_structured_output:
                try:
                    restored = _restore_schema_keys(json.loads(content))
                    return response_schema.model_validate(restored)
                except Exception as e:
                    logger.warning(
                        f"Structured output was not valid JSON for the schema "
                        f"({type(e).__name__}); falling back to completion parsing."
                    )
            return self.parse_from_completion(content, response_schema)

        schema = (
            _sanitize_schema_keys(response_schema.model_json_schema())
            if self.use_structured_output
            else None
        )
        result = complete(
            _build_messages(prompt, system_instruction, screenshot, pdf_url),
            model=self.model_name,
            schema=schema,
            parse=parse,
            feature=feature,
        )
        return result.parsed, result.usage

    def extract_json_objects(self, text):
        return extract_json_objects(text)

    def parse_from_completion(
        self, content: str, response_schema: type[BaseModel]
    ) -> BaseModel:
        return parse_json_from_completion(content, response_schema)
