import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

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


class ScreenshotData(BaseModel):
    filename: str = Field(...)
    base64: str = Field(...)


class OutputData(BaseModel):
    json_data: dict | None = Field(default=None)
    screenshot: ScreenshotData = Field(default=None)
    text: str | None = Field(default=None)


class Variables(BaseModel):
    input_variables: dict[str, list[str]]
    output_data: list[OutputData] = Field(default_factory=list)
    generated_variables: dict = Field(default_factory=dict)
    unique_parameters: list[str] = Field(default_factory=list)


class Memory(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    recording_id: str

    created_at: datetime
    started_at: datetime | None = Field(default=None)
    completed_at: datetime | None = Field(default=None)
    status: Literal["running", "success", "failed", "cancelled"] | None = Field(
        default=None
    )
    error: str | None = Field(default=None)

    save_directory: Path = Field(default=Path("/tmp/optexity"))
    task_directory: Path | None = Field(default=None)
    logs_directory: Path | None = Field(default=None)
    downloads_directory: Path | None = Field(default=None)
    log_file_path: Path | None = Field(default=None)

    variables: Variables
    automation_state: AutomationState = Field(default_factory=AutomationState)
    browser_states: list[BrowserState] = Field(default_factory=list)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    downloads: list[Path] = Field(default_factory=list)
    final_screenshot: str | None = Field(default=None)

    @model_validator(mode="after")
    def set_dependent_paths(self):
        self.task_directory = self.save_directory / str(self.task_id)
        self.logs_directory = self.task_directory / "logs"
        self.downloads_directory = self.task_directory / "downloads"
        self.log_file_path = self.logs_directory / "optexity.log"

        self.logs_directory.mkdir(parents=True, exist_ok=True)
        self.downloads_directory.mkdir(parents=True, exist_ok=True)
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)

        return self
