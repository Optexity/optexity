import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from optexity.schema.token_usage import TokenUsage


class NetworkRequest(BaseModel):
    url: str = Field(...)
    method: str = Field(...)
    status: int = Field(...)
    headers: dict = Field(...)
    body: str = Field(...)


class NetworkError(BaseModel):
    url: str = Field(...)
    message: str = Field(...)
    stack_trace: str = Field(...)


class NetworkResponse(BaseModel):
    url: str = Field(...)
    status: int = Field(...)
    headers: dict = Field(...)
    body: dict | str | None | bytes | Any = Field(default=None)
    method: str = Field(...)
    content_length: int = Field(...)


class AutomationState(BaseModel):
    step_index: int = Field(default_factory=lambda: -1)
    try_index: int = Field(default_factory=lambda: -1)
    start_2fa_time: float | None = Field(default=None)


class BrowserState(BaseModel):
    url: str = Field(...)
    title: str | None = Field(default=None)
    screenshot: str | None = Field(default=None)
    html: str | None = Field(default=None)
    axtree: str | None = Field(default=None)
    final_prompt: str | None = Field(default=None)
    llm_response: str | dict | None = Field(default=None)


class Variables(BaseModel):
    input_variables: dict[str, list[str]]
    output_data: list = Field(default_factory=list)
    generated_variables: dict = Field(default_factory=dict)


class Memory(BaseModel):
    task_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    save_directory: Path = Field(default=Path("/tmp/optexity"))
    logs_directory: Path = Field(default=Path("/tmp/optexity/logs"))
    downloads_directory: Path = Field(default=Path("/tmp/optexity/downloads"))
    log_file_path: Path = Field(default=Path("/tmp/optexity/logs/optexity.log"))
    variables: Variables
    automation_state: AutomationState = Field(default_factory=AutomationState)
    browser_states: list[BrowserState] = Field(default_factory=list)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    downloaded_files: list[Path] = Field(default_factory=list)

    @model_validator(mode="after")
    def set_dependent_paths(self):
        self.logs_directory = self.save_directory / str(self.task_id) / "logs"
        self.downloads_directory = self.save_directory / str(self.task_id) / "downloads"
        self.log_file_path = self.logs_directory / "optexity.log"

        self.logs_directory.mkdir(parents=True, exist_ok=True)
        self.downloads_directory.mkdir(parents=True, exist_ok=True)
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)

        return self
