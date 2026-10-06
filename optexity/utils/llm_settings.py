"""Model routing config, in its own module so that importing it does not
construct the task-runtime `Settings` singleton.

Keeping these fields here lets an embedder that uses just the model layer
(opcloud's recording processor) import it without supplying OPTEXITY_API_KEY or
a DEPLOYMENT this package recognises — importing `optexity.utils.settings` would
run `Settings()` and fail on both.
"""

import logging
import os

from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

env_path = os.getenv("ENV_PATH")
if not env_path:
    logger.warning("ENV_PATH is not set, using default values")


class LLMSettings(BaseSettings):
    # Every LLM call goes through OpenRouter with this key.
    OPENROUTER_API_KEY: str | None = None

    # OpenRouter model IDs, e.g. "google/gemini-3.6-flash" (an "openrouter/"
    # prefix is accepted). LLM_MODEL is used by calls that don't pick a model;
    # unset means the lite tier. LLM_MODEL_FALLBACK replaces the tier fallback
    # for those calls and for workflow-pinned models.
    LLM_MODEL: str | None = None
    LLM_MODEL_FALLBACK: str | None = None

    # Comma-separated OpenRouter model IDs replacing a tier's default list, in
    # fallback order.
    LLM_TIER_LITE: str | None = None
    LLM_TIER_STANDARD: str | None = None
    LLM_TIER_PRO: str | None = None

    # Tags sent with every request so usage can be split by service in the
    # OpenRouter dashboard.
    LLM_SERVICE_NAME: str = "optexity-worker"
    LLM_ENVIRONMENT: str | None = None

    class Config:
        env_file = env_path if env_path else None
        extra = "allow"


llm_settings = LLMSettings()
