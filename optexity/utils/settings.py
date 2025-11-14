import os

from pydantic_settings import BaseSettings

env_path = os.getenv("ENV_PATH")
if not env_path:
    raise ValueError("ENV_PATH is not set")


class Settings(BaseSettings):
    SERVER_URL: str = "http://localhost:8000"

    CREATE_TASK_ENDPOINT: str = "api/v1/create_task"
    START_TASK_ENDPOINT: str = "api/v1/start_task"
    COMPLETE_TASK_ENDPOINT: str = "api/v1/complete_task"
    SAVE_OUTPUT_DATA_ENDPOINT: str = "api/v1/save_output_data"
    SAVE_DOWNLOADS_ENDPOINT: str = "api/v1/save_downloads"
    SAVE_TRAJECTORY_ENDPOINT: str = "api/v1/save_trajectory"

    API_KEY: str

    class Config:
        env_file = env_path


settings = Settings()
