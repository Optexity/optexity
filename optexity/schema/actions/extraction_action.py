from typing import Any, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

from optexity.schema.actions.two_fa_action import TwoFAAction
from optexity.utils.settings import settings
from optexity.utils.utils import build_model


class LLMExtraction(BaseModel):
    source: list[Literal["axtree", "screenshot"]] = ["axtree"]
    extraction_format: dict
    extraction_instructions: str
    output_variable_names: list[str] | None = None
    llm_provider: Literal["gemini", "openai"] = settings.AGENT_LLM_PROVIDER
    llm_model_name: str = settings.AGENT_LLM_MODEL

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
    llm_provider: Literal["gemini", "openai"] = settings.AGENT_LLM_PROVIDER
    llm_model_name: str = settings.AGENT_LLM_MODEL

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


class ExtractionAction(BaseModel):
    unique_identifier: str | None = None
    network_call: Optional[NetworkCallExtraction] = None
    llm: Optional[LLMExtraction] = None
    python_script: Optional[PythonScriptExtraction] = None
    screenshot: Optional[ScreenshotExtraction] = None
    state: Optional[StateExtraction] = None
    two_fa_action: TwoFAAction | None = None
    pdf: Optional[PDFExtraction] = None

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
        }
        non_null = [k for k, v in provided.items() if v is not None]

        if len(non_null) != 1:
            raise ValueError(
                "Exactly one of llm, networkcall, python_script, screenshot, state, or two_fa_action must be provided"
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
        return self
