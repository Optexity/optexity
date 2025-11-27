import base64
import json
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Literal, Optional

from PIL import Image
from pydantic import BaseModel, Field, model_validator

from optexity.schema.automation import Automation
from optexity.schema.token_usage import TokenUsage


class Task(BaseModel):
    task_id: str
    user_id: str
    recording_id: str
    automation: Automation
    input_parameters: dict[str, list[str]]
    unique_parameter_names: list[str]
    unique_parameters: dict[str, list[str]] | None = None
    created_at: datetime
    allocated_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    status: Literal["queued", "allocated", "running", "success", "failed", "cancelled"]

    save_directory: Path = Field(default=Path("/tmp/optexity"))
    task_directory: Path | None = Field(default=None)
    logs_directory: Path | None = Field(default=None)
    downloads_directory: Path | None = Field(default=None)
    log_file_path: Path | None = Field(default=None)

    dedup_key: str = str(uuid.uuid4())
    retry_count: int = 0
    max_retries: int = 1

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat() if v is not None else None}

    @model_validator(mode="after")
    def validate_unique_parameters(self):
        if len(self.unique_parameter_names) > 0:
            self.unique_parameters = {
                unique_parameter_name: self.input_parameters[unique_parameter_name]
                for unique_parameter_name in self.unique_parameter_names
            }
            self.dedup_key = json.dumps(self.unique_parameters, sort_keys=True)

        return self

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


class TaskCreateRequest(BaseModel):
    task_id: str
    recording_id: str
    input_parameters: dict
    unique_parameter_names: list[str]
    created_at: datetime

    @model_validator(mode="after")
    def must_have_timezone(self):
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must include timezone information")

        for unique_parameter_name in self.unique_parameter_names:
            if unique_parameter_name not in self.input_parameters:
                raise ValueError(
                    f"unique_parameter_name {unique_parameter_name} not found in input_parameters"
                )
        return self


class TaskStartedRequest(BaseModel):
    task_id: str
    started_at: datetime
    allocated_at: Optional[datetime] = None

    @model_validator(mode="after")
    def must_have_timezone(self):
        if self.started_at.tzinfo is None:
            raise ValueError("started_at must include timezone information")
        if self.allocated_at is not None and self.allocated_at.tzinfo is None:
            raise ValueError("allocated_at must include timezone information")
        return self


class TaskCompleteRequest(BaseModel):
    task_id: str
    child_process_id: int

    status: Literal["success", "failed", "cancelled"]
    error: str | None
    completed_at: datetime
    token_usage: TokenUsage

    @model_validator(mode="after")
    def must_have_timezone(self):
        if self.completed_at.tzinfo is None:
            raise ValueError("completed_at must include timezone information")
        return self


class TaskOutputDataRequest(BaseModel):
    task_id: str
    output_data: list[dict]
    final_screenshot: str | None

    @model_validator(mode="after")
    def must_have_valid_final_screenshot(self):
        if self.final_screenshot is not None and not self.is_valid_base64_image(
            self.final_screenshot
        ):
            raise ValueError("final_screenshot must be a valid base64 encoded image")
        return self

    def is_valid_base64_image(self, data: str) -> bool:
        try:
            # Decode the base64 string
            decoded = base64.b64decode(data, validate=True)
            # Try to open it as an image
            Image.open(BytesIO(decoded))
            return True
        except Exception as e:
            return False
