import base64
import json
import string
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Literal, Optional

from PIL import Image
from pydantic import BaseModel, Field, computed_field, model_validator

from optexity.schema.automation import Automation, SecureParameter
from optexity.schema.memory import ForLoopStatus, SystemInfo
from optexity.schema.token_usage import TokenUsage

BASE62 = string.digits + string.ascii_lowercase + string.ascii_uppercase


def uuid_str_to_base62(uuid_str: str) -> str:
    n = uuid.UUID(uuid_str).int
    out = []
    while n:
        n, r = divmod(n, 62)
        out.append(BASE62[r])
    return "".join(reversed(out))


class CallbackUrl(BaseModel):
    url: str
    api_key: str | None = None
    username: str | None = None
    password: str | None = None

    @model_validator(mode="after")
    def validate_callback_url(self):

        if self.api_key is not None and (
            self.username is not None or self.password is not None
        ):
            raise ValueError(
                "api_key and username/password cannot be used together. Please provide only one of them."
            )

        return self


class Task(BaseModel):
    task_id: str
    user_id: str
    recording_id: str
    endpoint_name: str
    automation: Automation
    input_parameters: dict[str, list[str | int | float | bool]]
    secure_parameters: dict[str, list[SecureParameter]]
    unique_parameter_names: list[str]
    unique_parameters: dict[str, list[str]] | None = None
    created_at: datetime
    allocated_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    status: Literal[
        "queued", "allocated", "running", "success", "failed", "cancelled", "killed"
    ]
    is_cloud: bool = False
    save_directory: Path = Field(default=Path("/tmp/optexity"))
    use_proxy: bool = False

    dedup_key: str = Field(default_factory=lambda: str(uuid.uuid4()))
    retry_count: int = 0
    max_retries: int = 1
    api_key: str
    callback_url: CallbackUrl | None = None
    is_dedicated: bool = False

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat() if v is not None else None}

    @computed_field
    @property
    def task_directory(self) -> Path:
        return self.save_directory / str(self.task_id)

    @computed_field
    @property
    def logs_directory(self) -> Path:
        return self.task_directory / "logs"

    @computed_field
    @property
    def downloads_directory(self) -> Path:
        return self.task_directory / "downloads"

    @computed_field
    @property
    def log_file_path(self) -> Path:
        return self.logs_directory / "optexity.log"

    @model_validator(mode="after")
    def validate_unique_parameters(self):
        ## TODO: we do not do dedup using secure parameters yet, need to add support for that
        if len(self.unique_parameter_names) > 0:
            self.unique_parameters = {
                unique_parameter_name: self.input_parameters[unique_parameter_name]
                for unique_parameter_name in self.unique_parameter_names
            }
            self.dedup_key = (
                json.dumps(self.unique_parameters, sort_keys=True) + self.user_id
            )

        for a, b in [
            (self.automation.parameters.input_parameters, self.input_parameters),
            (self.automation.parameters.secure_parameters, self.secure_parameters),
        ]:
            if a.keys() != b.keys():
                missing_keys = a.keys() - b.keys()
                extra_keys = b.keys() - a.keys()
                raise ValueError(
                    f"Please provide exactly the same {a} as the automation. Missing keys: {missing_keys}, Extra keys: {extra_keys}"
                )

        return self

    @model_validator(mode="after")
    def set_dependent_paths(self):

        self.logs_directory.mkdir(parents=True, exist_ok=True)
        self.downloads_directory.mkdir(parents=True, exist_ok=True)
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)

        return self

    def proxy_session_id(
        self, proxy_provider: Literal["oxylabs", "brightdata", "other"] | None
    ) -> str | None:
        if not self.use_proxy:
            return None
        if proxy_provider == "oxylabs":
            return uuid_str_to_base62(self.task_id)
        else:
            return "default"


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
    for_loop_status: list[list[ForLoopStatus]] | None = None
    system_info: list[SystemInfo] | None = None
    unique_child_arn: str | None = None

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
