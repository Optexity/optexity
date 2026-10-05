"""LLM calls through OpenRouter with model tiers and fallback.

client = OpenRouterClient(LLMConfig(api_key=..., service="optexity-worker"))
result = client.complete([user_message("hi")], tier=Tier.PRO)
"""

from .client import OpenRouterClient
from .core import (
    DEFAULT_TIERS,
    MODEL_SPECS,
    Attempt,
    ChainEntry,
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
    ModelSpec,
    Tier,
    Usage,
    normalize_model_id,
    resolve_chain,
)
from .messages import image_part, pdf_part, user_message

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_TIERS",
    "MODEL_SPECS",
    "Attempt",
    "ChainEntry",
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
    "LLMUnavailable",
    "ModelSpec",
    "OpenRouterClient",
    "Tier",
    "Usage",
    "image_part",
    "pdf_part",
    "resolve_chain",
    "user_message",
    "normalize_model_id",
]
