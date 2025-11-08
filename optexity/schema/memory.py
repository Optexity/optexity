import uuid
from pathlib import Path

from optexity.schema.token_usage import TokenUsage
from pydantic import BaseModel, Field


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
    body: str = Field(...)


class AutomationState(BaseModel):
    step_index: int = Field(default_factory=lambda: -1)
    try_index: int = Field(default_factory=lambda: -1)


class BrowserState(BaseModel):
    url: str = Field(...)
    title: str | None = Field(default=None)
    screenshot: str | None = Field(default=None)
    html: str | None = Field(default=None)
    axtree: str | None = Field(default=None)


class Variables(BaseModel):
    input_variables: dict[str, list[str]]
    output_data: list = Field(default_factory=list)
    generated_variables: dict = Field(default_factory=dict)


class Memory(BaseModel):
    task_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    save_directory: Path = Field(default=Path("/tmp/optexity"))
    variables: Variables
    automation_state: AutomationState = Field(default_factory=AutomationState)
    browser_states: list[BrowserState] = Field(default_factory=list)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
