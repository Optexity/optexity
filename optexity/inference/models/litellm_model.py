import base64
import json
import logging
from pathlib import Path
from typing import Any, Optional

import httpx
import litellm
from pydantic import BaseModel

from optexity.utils.llm_settings import llm_settings, resolve_llm_api_key
from optexity.utils.utils import is_local_path, is_url

from .llm_model import LLMModel, TokenUsage

logger = logging.getLogger(__name__)

_SPACE_PLACEHOLDER = "_._"


def _sanitize_schema_keys(obj):
    """Recursively replace spaces in dict keys with _._

    Anthropic rejects tool schemas with spaces in property names, and extraction
    schemas come from user-authored workflow JSON where spaces are common.
    """
    if isinstance(obj, dict):
        return {
            k.replace(" ", _SPACE_PLACEHOLDER): _sanitize_schema_keys(v)
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


# Gemini 3.x thinks by default. litellm has no thinking_level support, so this
# goes through reasoning_effort, which it maps to a thinkingBudget: "minimal" is
# 128 tokens and "disable"/"none" are 0, which Gemini 3.x rejects with a 400.
# 128 is therefore the floor.
_GEMINI_3_REASONING_EFFORT = "medium"


def reasoning_effort_for(model: str) -> str | None:
    """The reasoning_effort to force on a model, or None to leave it unset.

    Scoped to gemini-3* on purpose. litellm turns reasoning_effort into
    per-provider thinking config, and "minimal" becomes budget_tokens=128 on
    Anthropic — under its 1024 floor, so it would 400 every Claude call. Gemini
    2.x is left on the SDK default, as it was before 3.x became the default.
    """
    if model.split("/")[-1].startswith("gemini-3"):
        return _GEMINI_3_REASONING_EFFORT
    return None


def litellm_fallbacks(model: str) -> list[dict[str, Any]]:
    """LLM_MODEL_FALLBACK as a litellm dict so it carries its own api_key.

    Rebuilt on every call: litellm pops "model" off these dicts. api_key and
    reasoning_effort are always set, even to None, because litellm merges the
    primary call's kwargs into each fallback — leaving them out would send the
    primary provider's key, and a Gemini-shaped reasoning_effort, to a fallback
    on a different provider.
    """
    fallback = llm_settings.LLM_MODEL_FALLBACK
    if not fallback or fallback == model:
        return []
    return [
        {
            "model": fallback,
            "api_key": resolve_llm_api_key(fallback),
            "reasoning_effort": reasoning_effort_for(fallback),
        }
    ]


def _pdf_to_base64(pdf_url: str | Path) -> str:
    if is_local_path(pdf_url):
        raw = Path(str(pdf_url)).read_bytes()
    elif is_url(pdf_url):
        raw = httpx.get(str(pdf_url)).content
    else:
        raise ValueError(f"Invalid pdf_url: {pdf_url}")
    return base64.standard_b64encode(raw).decode("utf-8")


class LiteLLMModel(LLMModel):
    """Single provider-agnostic backend. `model_name` is any litellm model string."""

    def _build_messages(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        screenshot: Optional[str] = None,
        pdf_url: Optional[str | Path] = None,
    ) -> list[dict[str, Any]]:

        if pdf_url is not None and screenshot is not None:
            raise ValueError("Cannot use both screenshot and pdf_url")

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]

        if screenshot is not None:
            content.insert(
                0,
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{screenshot}"},
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

    def _completion(self, messages: list[dict[str, Any]], **kwargs):
        fallbacks = litellm_fallbacks(self.model_name)
        if fallbacks:
            # litellm only routes through its fallback path when this is present,
            # and that path hops to a worker thread — skip it when unconfigured.
            kwargs["fallbacks"] = fallbacks
        kwargs.setdefault("reasoning_effort", reasoning_effort_for(self.model_name))
        return litellm.completion(
            model=self.model_name,
            messages=messages,
            api_key=resolve_llm_api_key(self.model_name),
            # LLMModel.get_model_response already retries 3x; letting litellm retry
            # too would multiply that out to 9 attempts.
            num_retries=0,
            drop_params=True,
            **kwargs,
        )

    def _token_usage_from(self, response) -> TokenUsage:
        usage = getattr(response, "usage", None)
        if usage is None:
            return TokenUsage()
        details = getattr(usage, "completion_tokens_details", None)
        return self.get_token_usage(
            input_tokens=getattr(usage, "prompt_tokens", 0),
            output_tokens=getattr(usage, "completion_tokens", 0),
            thoughts_tokens=getattr(details, "reasoning_tokens", 0) if details else 0,
            total_tokens=getattr(usage, "total_tokens", 0),
        )

    def _get_model_response(
        self, prompt: str, system_instruction: Optional[str] = None
    ) -> tuple[str, TokenUsage]:

        response = self._completion(self._build_messages(prompt, system_instruction))
        return (response.choices[0].message.content or ""), self._token_usage_from(
            response
        )

    def _get_model_response_with_structured_output(
        self,
        prompt: str,
        response_schema: type[BaseModel],
        screenshot: Optional[str] = None,
        pdf_url: Optional[str | Path] = None,
        system_instruction: Optional[str] = None,
    ) -> tuple[BaseModel | None, TokenUsage]:

        messages = self._build_messages(prompt, system_instruction, screenshot, pdf_url)

        kwargs: dict[str, Any] = {}
        if self.use_structured_output:
            # Pass the schema as a dict rather than the pydantic class so the
            # space-in-key sanitization survives. drop_params=True means litellm
            # silently drops this for providers that can't honor it, and we fall
            # back to parsing the raw completion below.
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "schema": _sanitize_schema_keys(
                        response_schema.model_json_schema()
                    ),
                    "strict": False,
                },
            }

        response = self._completion(messages, **kwargs)
        token_usage = self._token_usage_from(response)
        content = response.choices[0].message.content or ""

        if self.use_structured_output:
            try:
                restored = _restore_schema_keys(json.loads(content))
                return response_schema.model_validate(restored), token_usage
            except Exception as e:
                logger.warning(
                    f"Structured output from {self.model_name} was not valid JSON "
                    f"for the schema ({e}); falling back to completion parsing."
                )

        return self.parse_from_completion(content, response_schema), token_usage
