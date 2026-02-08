import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import psutil
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
    method: str | None = Field(default=None)
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


class SystemInfo(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_system_memory: float = Field(
        default_factory=lambda: SystemInfo.get_effective_memory_mb()[1]
    )  # convert to MB
    total_system_memory_used: float = Field(
        default_factory=lambda: SystemInfo.get_effective_memory_mb()[0]
    )  # convert to MB

    @staticmethod
    def get_effective_memory_mb():
        """
        Returns (used_mb, total_mb)
        - If running inside Docker → container memory
        - Else → system memory
        Works on Linux (Ubuntu) and macOS.
        """

        # ---------- Linux: try cgroups (Docker / K8s) ----------
        if Path("/sys/fs/cgroup").exists():
            # cgroup v2
            mem_current = Path("/sys/fs/cgroup/memory.current")
            mem_max = Path("/sys/fs/cgroup/memory.max")

            if mem_current.exists() and mem_max.exists():
                try:
                    used = int(mem_current.read_text().strip())
                    limit_raw = mem_max.read_text().strip()
                    if limit_raw != "max":
                        limit = int(limit_raw)
                        return used / (1024**2), limit / (1024**2)
                except Exception:
                    pass

            # cgroup v1
            mem_used = Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")
            mem_limit = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")

            if mem_used.exists() and mem_limit.exists():
                try:
                    used = int(mem_used.read_text().strip())
                    limit = int(mem_limit.read_text().strip())
                    # very large limit means "no limit"
                    if limit < (1 << 60):
                        return used / (1024**2), limit / (1024**2)
                except Exception:
                    pass

        # ---------- Fallback: system memory (macOS or non-docker) ----------
        vm = psutil.virtual_memory()
        return vm.used / (1024**2), vm.total / (1024**2)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat() if v is not None else None}


class BrowserState(BaseModel):
    url: str = Field(...)
    title: str | None = Field(default=None)
    screenshot: str | None = Field(default=None)
    html: str | None = Field(default=None)
    axtree: str | None = Field(default=None)
    final_prompt: str | None = Field(default=None)
    llm_response: str | dict | None = Field(default=None)
    system_info: SystemInfo = Field(default_factory=SystemInfo)


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
    system_info_tracking: list[SystemInfo] = Field(default_factory=list)
    unique_child_arn: str

    model_config = {
        "arbitrary_types_allowed": True,
        "exclude": {"download_lock"},
    }

    def update_system_info(self):
        self.system_info_tracking.append(SystemInfo())
