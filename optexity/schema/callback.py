from typing import Literal

from pydantic import BaseModel


class CallbackResponse(BaseModel):
    task_id: str
    recording_id: str
    output_data: list[dict | str] | None
    status: Literal["queued", "allocated", "running", "success", "failed", "cancelled"]
    error: str | None
    final_screenshot: str | None = None
    endpoint_name: str
    downloads: list[dict] | None = None
    input_parameters: dict[str, list[str | int | float | bool]] | None = None
    unique_parameter_names: list[str] | None = None
