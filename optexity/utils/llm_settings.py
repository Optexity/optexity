"""Model routing config, in its own module so that importing it does not
construct the task-runtime `Settings` singleton.

`optexity.inference.models` needs only these four fields. Keeping them here lets
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
    # LiteLLM model routing: one primary model and one fallback, each with its own
    # key so the two can live on different providers.
    #
    #   LLM_MODEL=anthropic/claude-sonnet-4-6
    #   LLM_MODEL_API_KEY=...
    #   LLM_MODEL_FALLBACK=openai/gpt-4.1-mini
    #   LLM_MODEL_FALLBACK_API_KEY=...
    #
    # Model strings are any litellm model ("provider/model", or a bare name for
    # openai). A key may be omitted, in which case litellm reads the provider's own
    # env var (GEMINI_API_KEY / GOOGLE_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY).
    #
    # Every field has a default, so this validates under any environment.
    LLM_MODEL: str = "gemini/gemini-3.5-flash-lite"
    LLM_MODEL_API_KEY: str | None = None
    LLM_MODEL_FALLBACK: str | None = None
    LLM_MODEL_FALLBACK_API_KEY: str | None = None

    def llm_api_key_for(self, model: str) -> str | None:
        """The configured key for an arbitrary litellm model string.

        An exact model match wins, then any configured model from the same
        provider — so a task overriding LLM_MODEL with a sibling model still gets
        the right key. No match means litellm resolves the provider's env var.
        """
        configured = [
            (m, k)
            for m, k in (
                (self.LLM_MODEL, self.LLM_MODEL_API_KEY),
                (self.LLM_MODEL_FALLBACK, self.LLM_MODEL_FALLBACK_API_KEY),
            )
            if m and k
        ]
        for configured_model, key in configured:
            if configured_model == model:
                return key
        provider = model.split("/")[0] if "/" in model else ""
        if provider:
            for configured_model, key in configured:
                if configured_model.split("/")[0] == provider:
                    return key
        return None

    class Config:
        env_file = env_path if env_path else None
        extra = "allow"


llm_settings = LLMSettings()


def resolve_llm_api_key(model: str) -> str | None:
    """The configured key for this litellm model string, else the provider env var.

    Both halves are load-bearing: `LLM_MODEL_API_KEY` is read out of the env file
    into `llm_settings` only and never lands in `os.environ`, while a bare
    `GOOGLE_API_KEY` in that same file only reaches `os.environ` via the
    `load_dotenv` in `cli.py`. Callers get the key either way.

    Special case: litellm reads GEMINI_API_KEY for the gemini/ route, but this
    codebase and both opcloud deploy paths set GOOGLE_API_KEY.
    """
    key = llm_settings.llm_api_key_for(model)
    if key:
        return key
    if model.startswith("gemini/"):
        return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    return None
