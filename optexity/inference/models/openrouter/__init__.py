"""LLM calls through OpenRouter with model tiers and fallback.

client = OpenRouterClient(LLMConfig(api_key=..., service="optexity-worker"))
result = client.complete([user_message("hi")], tier=Tier.PRO)
"""

from .client import OpenRouterClient
from .core import (
    DEFAULT_TIERS,
    LEGACY_MODEL_ALIASES,
    Attempt,
    ChainEntry,
    Citation,
    LLMAllModelsFailed,
    LLMAuthError,
    LLMBadRequest,
    LLMConfig,
    LLMConfigError,
    LLMError,
    LLMInvalidOutput,
    LLMModelRejected,
    LLMOutOfCredits,
    LLMRateLimited,
    LLMResult,
    LLMTimeout,
    LLMUnavailable,
    Tier,
    ToolCall,
    Usage,
    WebSearch,
    normalize_model_id,
    parse_model_list,
    resolve_chain,
)
from .messages import image_part, pdf_part, user_message

__version__ = "0.7.0"

__all__ = [
    "DEFAULT_TIERS",
    "Attempt",
    "ChainEntry",
    "Citation",
    "LLMAllModelsFailed",
    "LLMAuthError",
    "LLMBadRequest",
    "LLMConfig",
    "LLMConfigError",
    "LLMError",
    "LLMInvalidOutput",
    "LLMModelRejected",
    "LLMOutOfCredits",
    "LLMRateLimited",
    "LLMResult",
    "LLMTimeout",
    "LEGACY_MODEL_ALIASES",
    "LLMUnavailable",
    "OpenRouterClient",
    "Tier",
    "ToolCall",
    "Usage",
    "WebSearch",
    "image_part",
    "parse_model_list",
    "pdf_part",
    "resolve_chain",
    "user_message",
    "normalize_model_id",
]
