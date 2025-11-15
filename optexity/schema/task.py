import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from optexity.schema.automation import Automation


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

    dedup_key: str = Field(init=False)

    @model_validator(mode="after")
    def validate_unique_parameters(self):
        if len(self.unique_parameter_names) > 0:
            self.unique_parameters = {
                unique_parameter_name: self.input_parameters[unique_parameter_name]
                for unique_parameter_name in self.unique_parameter_names
            }
            self.dedup_key = json.dumps(self.unique_parameters, sort_keys=True)
        else:
            self.dedup_key = str(uuid.uuid4())
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
