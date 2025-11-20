import logging
import os
from typing import Literal

from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

env_path = os.getenv("ENV_PATH")
if not env_path:
    logger.warning("ENV_PATH is not set, using default values")


class Settings(BaseSettings):
    SERVER_URL: str = "http://localhost:8000"
    HEALTH_ENDPOINT: str = "api/v1/health"

    CREATE_TASK_ENDPOINT: str = "api/v1/create_task"
    START_TASK_ENDPOINT: str = "api/v1/start_task"
    COMPLETE_TASK_ENDPOINT: str = "api/v1/complete_task"
    SAVE_OUTPUT_DATA_ENDPOINT: str = "api/v1/save_output_data"
    SAVE_DOWNLOADS_ENDPOINT: str = "api/v1/save_downloads"
    SAVE_TRAJECTORY_ENDPOINT: str = "api/v1/save_trajectory"

    API_KEY: str

    DEPLOYMENT: Literal["local", "cloud"]
    CHILD_PORT_OFFSET: int = 9000

    class Config:
        env_file = env_path if env_path else None
        extra = "allow"


settings = Settings()
