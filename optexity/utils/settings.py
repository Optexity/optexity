import logging
import os
from typing import Literal

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

env_path = os.getenv("ENV_PATH")
if not env_path:
    logger.warning("ENV_PATH is not set, using default values")


class Settings(BaseSettings):
    SERVER_URL: str = "https://api.optexity.com"
    HEALTH_ENDPOINT: str = "api/v1/health"
    INFERENCE_ENDPOINT: str = "api/v1/inference"
    ADD_EXAMPLE_ENDPOINT: str = "api/v1/add_example"
    UPDATE_EXAMPLE_ENDPOINT: str = "api/v1/update_example"
    START_TASK_ENDPOINT: str = "api/v1/start_task"
    COMPLETE_TASK_ENDPOINT: str = "api/v1/complete_task"
    SAVE_OUTPUT_DATA_ENDPOINT: str = "api/v1/save_output_data"
    REQUEST_DOWNLOAD_UPLOAD_URLS_ENDPOINT: str = "api/v1/request_download_upload_urls"
    CONFIRM_DOWNLOADS_ENDPOINT: str = "api/v1/confirm_downloads"
    SAVE_TRAJECTORY_ENDPOINT: str = "api/v1/save_trajectory"
    INITIATE_CALLBACK_ENDPOINT: str = "api/v1/initiate_callback"
    GET_CALLBACK_DATA_ENDPOINT: str = "api/v1/get_callback_data"
    FETCH_EMAIL_MESSAGES_ENDPOINT: str = "api/v1/fetch_email_messages"
    FETCH_SLACK_MESSAGES_ENDPOINT: str = "api/v1/fetch_slack_messages"
    FETCH_SMS_MESSAGES_ENDPOINT: str = "api/v1/fetch_sms_messages"
    INTEGRATION_SECRETS_ENDPOINT: str = "api/v1/integration-secrets/{type}/encrypt"
    HUMAN_IN_LOOP_ENDPOINT: str = "api/v1/human_in_loop"

    FERNET_SECRET_KEY: str | None = None  # required when using integration secrets

    OPTEXITY_API_KEY: str = Field(
        validation_alias=AliasChoices("OPTEXITY_API_KEY", "API_KEY")
    )

    CHILD_PORT_OFFSET: int = 9000
    WEBSOCKIFY_PORT: int = 8080
    DEPLOYMENT: Literal["dev", "prod"]
    LOCAL_CALLBACK_URL: str | None = None

    USE_PLAYWRIGHT_BROWSER: bool = True

    PROXY_URL: str | None = None
    PROXY_USERNAME: str | None = None
    PROXY_PASSWORD: str | None = None
    PROXY_COUNTRY: str | None = None
    PROXY_PROVIDER: Literal["oxylabs", "brightdata", "other"] | None = None

    BROWSER_USE_API_KEY: str | None = None

    DOWNLOAD_TIMEOUT_SECONDS: float = 200.0

    UPLOAD_CONNECT_TIMEOUT_SECONDS: float = 30.0
    UPLOAD_WRITE_TIMEOUT_SECONDS: float = 300.0
    UPLOAD_READ_TIMEOUT_SECONDS: float = 600.0
    UPLOAD_POOL_TIMEOUT_SECONDS: float = 30.0

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
    LLM_MODEL: str = "gemini/gemini-2.5-flash"
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

    @model_validator(mode="after")
    def validate_local_callback_url(self):
        if self.DEPLOYMENT == "prod" and self.LOCAL_CALLBACK_URL is not None:
            raise ValueError("LOCAL_CALLBACK_URL is not allowed in prod mode")

        if self.PROXY_PROVIDER == "oxylabs":
            if self.PROXY_COUNTRY is None:
                self.PROXY_COUNTRY = "US"
        return self

    class Config:
        env_file = env_path if env_path else None
        extra = "allow"


settings = Settings()  # pyright: ignore[reportCallIssue]
