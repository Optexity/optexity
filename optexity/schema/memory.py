import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from playwright.async_api import Download
from pydantic import BaseModel, Field, model_validator

from optexity.schema.token_usage import TokenUsage


class NetworkRequest(BaseModel):
    url: str
    method: str
    headers: dict
    body: str | bytes | None | dict | Any


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
    start_2fa_time: datetime | None = Field(default=None)

    @model_validator(mode="after")
    def validate_start_2fa_time(self):
        if self.start_2fa_time is not None:
            assert (
                self.start_2fa_time.tzinfo is not None
            ), "start_2fa_time must be timezone-aware"
        return self


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
    unique_identifier: str | None = None
    json_data: dict | None = Field(default=None)
    screenshot: ScreenshotData = Field(default=None)
    text: str | None = Field(default=None)


class ForLoopStatus(BaseModel):
    variable_name: str
    index: int
    value: str | int | float | bool
    error: str | None = None
    status: Literal["success", "error", "skipped"]


class Variables(BaseModel):
    output_data: list[OutputData] = Field(default_factory=list)
    for_loop_status: list[list[ForLoopStatus]] = Field(default_factory=list)
    generated_variables: dict = Field(default_factory=dict)


class Memory(BaseModel):
    variables: Variables = Field(default_factory=Variables)
    automation_state: AutomationState = Field(default_factory=AutomationState)
    browser_states: list[BrowserState] = Field(default_factory=list)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    download_lock: asyncio.Lock = Field(default_factory=asyncio.Lock)
    raw_downloads: dict[Path, tuple[bool, Download | None]] = Field(
        default_factory=dict
    )
    urls_to_downloads: list[tuple[str, str]] = Field(default_factory=list)
    downloads: list[Path] = Field(default_factory=list)
    final_screenshot: str | None = Field(default=None)

    model_config = {
        "arbitrary_types_allowed": True,
        "exclude": {"download_lock"},
    }
