import logging
import os
from typing import Literal

from pydantic import model_validator
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
    SAVE_DOWNLOADS_ENDPOINT: str = "api/v1/save_downloads"
    SAVE_TRAJECTORY_ENDPOINT: str = "api/v1/save_trajectory"
    INITIATE_CALLBACK_ENDPOINT: str = "api/v1/initiate_callback"
    GET_CALLBACK_DATA_ENDPOINT: str = "api/v1/get_callback_data"
    FETCH_EMAIL_MESSAGES_ENDPOINT: str = "api/v1/fetch_email_messages"
    FETCH_SLACK_MESSAGES_ENDPOINT: str = "api/v1/fetch_slack_messages"

    API_KEY: str

    CHILD_PORT_OFFSET: int = 9000
    DEPLOYMENT: Literal["dev", "prod"]
    LOCAL_CALLBACK_URL: str | None = None

    USE_PLAYWRIGHT_BROWSER: bool = True

    PROXY_URL: str | None = None
    PROXY_USERNAME: str | None = None
    PROXY_PASSWORD: str | None = None
    PROXY_COUNTRY: str | None = None
    PROXY_PROVIDER: Literal["oxylabs", "brightdata", "other"] | None = None

    @model_validator(mode="after")
    def validate_local_callback_url(self):
        if self.DEPLOYMENT == "prod" and self.LOCAL_CALLBACK_URL is not None:
            raise ValueError("LOCAL_CALLBACK_URL is not allowed in prod mode")
        return self

    class Config:
        env_file = env_path if env_path else None
        extra = "allow"


settings = Settings()
