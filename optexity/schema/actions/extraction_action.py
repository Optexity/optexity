from typing import Any, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_BB_VARS_LENGTH = 4  # [x1_var, y1_var, x2_var, y2_var]

from optexity.schema.actions.two_fa_action import TwoFAAction
from optexity.utils.utils import build_model, deep_replace


class LLMExtraction(BaseModel):
    source: list[Literal["axtree", "screenshot"]] = ["axtree"]
    extraction_format: dict
    extraction_instructions: str
    recording_screenshot: str | None = None
    output_variable_names: list[str] | None = None
    llm_provider: Literal["gemini", "anthropic", "openai"] = "gemini"
    llm_model_name: str = "gemini-2.5-flash"
    include_full_page: bool = False

    def build_model(self):
        return build_model(self.extraction_format)

    @field_validator("extraction_format")
    def validate_extraction_format(cls, v):
        if isinstance(v, dict):
            try:
                build_model(v)
            except Exception as e:
                raise ValueError(f"Invalid extraction_format dict: {e}")
            return v
        raise ValueError("extraction_format must be either a string or a dict")

    @model_validator(mode="after")
    def validate_output_var_in_format(self):

        if self.output_variable_names is not None:
            for key in self.output_variable_names:
                if key not in self.extraction_format:
                    raise ValueError(
                        f"Output variable {key} not found in extraction_format"
                    )
                ## TODO: fix this
                # if eval(self.extraction_format[key]) not in [
                #     int,
                #     float,
                #     bool,
                #     str,
                #     None,
                #     list[str | int | float | bool | None],
                #     List[str | int | float | bool | None],
                # ]:
                #     raise ValueError(
                #         f"Output variable {key} must be a string, int, float, bool, or a list of strings, ints, floats, or bools"
                #     )

        return self

    def replace(self, pattern: str, replacement: str):
        self.extraction_instructions = self.extraction_instructions.replace(
            pattern, replacement
        )
        return self


class NetworkCallExtraction(BaseModel):
    url_pattern: Optional[str] = None
    extract_from: None | Literal["request", "response"] = "response"
    download_from: None | Literal["request", "response"] = "response"
    download_filename: str | None = None

    @model_validator(mode="before")
    def download_filename_if_download_from_is_set(cls, data: dict[str, Any]):
        if (
            "downlowd_from" in data
            and data["download_from"] is not None
            and ("download_filename" not in data or data["download_filename"] is None)
        ):
            data["download_filename"] = str(uuid4())

        return data

    def replace(self, pattern: str, replacement: str):
        return self


class PythonScriptExtraction(BaseModel):
    script: str
    ## TODO: add output to memory variables

    @field_validator("script")
    @classmethod
    def validate_script(cls, v: str):
        if not v.strip():
            raise ValueError("Script cannot be empty")
        return v

    def replace(self, pattern: str, replacement: str):
        self.script = self.script.replace(pattern, replacement)
        return self


class ScreenshotExtraction(BaseModel):
    filename: str
    full_page: bool = True


class StateExtraction(BaseModel):
    pass


class PDFExtraction(BaseModel):
    filename: str
    extraction_format: dict
    extraction_instructions: str
    llm_provider: Literal["gemini", "anthropic", "openai"] = "gemini"
    llm_model_name: str = "gemini-2.5-flash"

    def build_model(self):
        return build_model(self.extraction_format)

    @field_validator("extraction_format")
    def validate_extraction_format(cls, v):
        if isinstance(v, dict):
            try:
                build_model(v)
            except Exception as e:
                raise ValueError(f"Invalid extraction_format dict: {e}")
            return v
        raise ValueError("extraction_format must be either a string or a dict")

    def replace(self, pattern: str, replacement: str):
        self.extraction_instructions = self.extraction_instructions.replace(
            pattern, replacement
        )
        return self


class OCRCoordinatesExtraction(BaseModel):
    source_variable: str
    output_x_variable: str = "coords_x"
    output_y_variable: str = "coords_y"
    bounding_box_variables: list[str] | None = None

    @model_validator(mode="after")
    def validate_bounding_box_variables_length(self):
        if (
            self.bounding_box_variables is not None
            and len(self.bounding_box_variables) != _BB_VARS_LENGTH
        ):
            raise ValueError(
                f"bounding_box_variables must have exactly {_BB_VARS_LENGTH} elements: [x1_var, y1_var, x2_var, y2_var]"
            )
        return self

    def replace(self, pattern: str, replacement: str):
        return self


class VisionExtraction(BaseModel):
    prompt: str
    output_variable_names: list[str]

    def replace(self, pattern: str, replacement: str):
        self.prompt = self.prompt.replace(pattern, replacement)
        return self


class LocatorExtraction(BaseModel):
    command: str
    output_variable_name: str
    extraction_format: dict
    extraction_instructions: str | None = None
    llm_provider: Literal["gemini"] = "gemini"
    llm_model_name: str = "gemini-2.5-flash"

    @model_validator(mode="after")
    def validate_variable_in_format(self):
        if self.output_variable_name not in self.extraction_format:
            raise ValueError(
                f"Variable {self.output_variable_name!r} not found in extraction_format"
            )
        return self

    def replace(self, pattern: str, replacement: str):
        self.command = self.command.replace(pattern, replacement)
        return self


class APICallExtraction(BaseModel):
    url: str
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"] = "GET"
    headers: dict[str, str] = Field(default_factory=dict)
    body: dict | str | None = None
    query_params: dict[str, str] = Field(default_factory=dict)
    output_variable_names: list[str] = Field(default_factory=lambda: ["api_result"])
    timeout: float = 30.0
    poll_condition: str | None = None
    poll_interval: float = 5.0
    max_poll_attempts: int = 10

    def replace(self, pattern: str, replacement: str):
        self.url = self.url.replace(pattern, replacement)
        self.headers = {
            k: v.replace(pattern, replacement) for k, v in self.headers.items()
        }
        if isinstance(self.body, str):
            self.body = self.body.replace(pattern, replacement)
        elif isinstance(self.body, dict):
            self.body = deep_replace(self.body, pattern, replacement)
        self.query_params = {
            k: v.replace(pattern, replacement) for k, v in self.query_params.items()
        }
        if self.poll_condition:
            self.poll_condition = self.poll_condition.replace(pattern, replacement)
        return self


class ExtractionAction(BaseModel):
    unique_identifier: str | None = None
    network_call: Optional[NetworkCallExtraction] = None
    llm: Optional[LLMExtraction] = None
    python_script: Optional[PythonScriptExtraction] = None
    screenshot: Optional[ScreenshotExtraction] = None
    state: Optional[StateExtraction] = None
    two_fa_action: TwoFAAction | None = None
    pdf: Optional[PDFExtraction] = None
    ocr_coordinates: Optional[OCRCoordinatesExtraction] = None
    locator: Optional[LocatorExtraction] = None
    vision: Optional[VisionExtraction] = None
    api_call: Optional[APICallExtraction] = None

    @model_validator(mode="after")
    def validate_one_extraction(self):
        """Ensure exactly one of the extraction types is set and matches the type."""
        provided = {
            "llm": self.llm,
            "network_call": self.network_call,
            "python_script": self.python_script,
            "screenshot": self.screenshot,
            "state": self.state,
            "two_fa_action": self.two_fa_action,
            "pdf": self.pdf,
            "ocr_coordinates": self.ocr_coordinates,
            "locator": self.locator,
            "vision": self.vision,
            "api_call": self.api_call,
        }
        non_null = [k for k, v in provided.items() if v is not None]

        if len(non_null) != 1:
            raise ValueError(
                "Exactly one of llm, network_call, python_script, screenshot, state, two_fa_action, pdf, ocr_coordinates, locator, vision, or api_call must be provided"
            )

        return self

    def replace(self, pattern: str, replacement: str):
        if self.network_call:
            self.network_call.replace(pattern, replacement)
        if self.llm:
            self.llm.replace(pattern, replacement)
        if self.python_script:
            self.python_script.replace(pattern, replacement)
        if self.unique_identifier:
            self.unique_identifier = self.unique_identifier.replace(
                pattern, replacement
            )
        if self.two_fa_action:
            self.two_fa_action.replace(pattern, replacement)
        if self.locator:
            self.locator.replace(pattern, replacement)
        if self.api_call:
            self.api_call.replace(pattern, replacement)

        return self
