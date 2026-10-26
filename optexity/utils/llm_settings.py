"""Model routing config, in its own module so that importing it does not
construct the task-runtime `Settings` singleton.

`optexity.inference.models` needs only these fields. Keeping them here lets
an embedder that uses just the model layer (opcloud's recording processor) import
it without supplying OPTEXITY_API_KEY or a DEPLOYMENT this package recognises —
importing `optexity.utils.settings` would run `Settings()` and fail on both.
"""

import logging
import os

from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

env_path = os.getenv("ENV_PATH")
if not env_path:
    logger.warning("ENV_PATH is not set, using default values")


class LLMSettings(BaseSettings):
    # Every LLM call runs on Google Vertex AI (Gemini and Claude), authenticated
    # with Application Default Credentials. Models are chosen per call (a tier or
    # a model ID); the tiers' default models live in inference/models/llm/config.py.
    # These settings are optional overrides, e.g.:
    #
    #   LLM_MODEL=gemini-3.6-flash                    model for calls that don't pick one
    #   LLM_MODEL_FALLBACK=claude-sonnet-5-5          backup for LLM_MODEL / llm_model_name
    #   LLM_TIER_PRO=claude-opus-5-5,gemini-4-argon   replaces the pro tier's models
    #
    # Every field has a default, so this validates under any environment.
    LLM_MODEL: str | None = None
    LLM_MODEL_FALLBACK: str | None = None
    LLM_TIER_LITE: str | None = None
    LLM_TIER_STANDARD: str | None = None
    LLM_TIER_PRO: str | None = None

    GOOGLE_CLOUD_PROJECT: str | None = None
    GOOGLE_CLOUD_LOCATION: str = "global"

    # Usage tags on every request (billing labels and the per-call log entry).
    LLM_SERVICE_NAME: str = "optexity"
    LLM_ENVIRONMENT: str = "dev"
    # Also send the per-call log entries to Google Cloud Logging.
    LLM_CLOUD_LOGGING: bool = False

    class Config:
        env_file = env_path if env_path else None
        extra = "allow"


llm_settings = LLMSettings()
