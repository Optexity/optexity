import logging
from pathlib import Path
from typing import Optional

import httpx
from pydantic import BaseModel

from optexity.schema.token_usage import TokenUsage
from optexity.utils.llm_settings import llm_settings
from optexity.utils.utils import is_local_path, is_url

from .openrouter import (
    LLMConfig,
    OpenRouterClient,
    Tier,
    Usage,
    parse_model_list,
    user_message,
)

logger = logging.getLogger(__name__)

_config: LLMConfig | None = None
_client: OpenRouterClient | None = None


def get_config() -> LLMConfig:
    """OpenRouter client config, built from `llm_settings` on first use."""
    global _config
    if _config is None:
        s = llm_settings
        _config = LLMConfig.with_tier_overrides(
            {
                Tier.LITE: parse_model_list(s.LLM_TIER_LITE),
                Tier.STANDARD: parse_model_list(s.LLM_TIER_STANDARD),
                Tier.PRO: parse_model_list(s.LLM_TIER_PRO),
            },
            api_key=s.OPENROUTER_API_KEY,
            service=s.LLM_SERVICE_NAME,
            environment=s.LLM_ENVIRONMENT,
            default_model=s.LLM_MODEL,
            env_fallback=s.LLM_MODEL_FALLBACK,
        )
    return _config


def get_client() -> OpenRouterClient:
    """The shared OpenRouter client. Raises LLMConfigError on first use if
    OPENROUTER_API_KEY isn't set."""
    global _client
    if _client is None:
        _client = OpenRouterClient(get_config())
    return _client


def _load_pdf(pdf_url: str | Path) -> bytes:
    if is_local_path(pdf_url):
        return Path(str(pdf_url)).read_bytes()
    if is_url(pdf_url):
        response = httpx.get(str(pdf_url), timeout=60, follow_redirects=True)
        response.raise_for_status()
        return response.content
    raise ValueError(f"Invalid pdf_url: {pdf_url}")


def to_token_usage(usage: Usage) -> TokenUsage:
    # Reasoning tokens are already inside completion_tokens, so they are reported
    # but not added to the total again. OpenRouter returns one total cost.
    total = usage.prompt_tokens + usage.completion_tokens
    return TokenUsage(
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        thoughts_tokens=usage.reasoning_tokens,
        total_tokens=total,
        calculated_total_tokens=total + usage.reasoning_tokens,
        total_cost=usage.cost,
    )


class LLMModel:
    """An LLM call target: an OpenRouter model ID, a `Tier`, or None for the
    configured default (`LLM_MODEL`, else the lite tier)."""

    def __init__(
        self, model_name: str | Tier | None, use_structured_output: bool = True
    ):
        self.model_name = model_name
        # Kept for existing callers; responses are always requested as structured output.
        self.use_structured_output = use_structured_output

    def get_model_response_with_structured_output(
        self,
        prompt: str,
        response_schema: type[BaseModel],
        screenshot: Optional[str] = None,
        pdf_url: Optional[str | Path] = None,
        system_instruction: Optional[str] = None,
        feature: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> tuple[BaseModel, TokenUsage]:
        """Ask the model for ``response_schema``; retries and fallback happen in
        the OpenRouter client. ``screenshot`` is a base64 PNG/JPEG; ``pdf_url`` a
        local path or URL. Returns the parsed object and its token usage."""
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append(
            user_message(
                prompt,
                images=[screenshot] if screenshot else (),
                pdfs=[_load_pdf(pdf_url)] if pdf_url else (),
            )
        )
        result = get_client().complete(
            messages,
            model=self.model_name,
            schema=response_schema,
            feature=feature,
            session_id=session_id,
        )
        return result.parsed, to_token_usage(result.usage)
