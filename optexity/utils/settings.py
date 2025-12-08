import logging
import os
from typing import Literal

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
    CALLBACK_ENDPOINT: str = "api/v1/callback"

    API_KEY: str

    CHILD_PORT_OFFSET: int = 9000
    DEPLOYMENT: Literal["dev", "prod"]

    class Config:
        env_file = env_path if env_path else None
        extra = "allow"


settings = Settings()
