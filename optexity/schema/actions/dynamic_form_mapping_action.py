from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from optexity.schema.actions.llm_actions import LLMAction
from optexity.utils.utils import build_model


class FormMappingCallbackUrl(BaseModel):
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
                "api_key and username/password cannot be used together. "
                "Please provide only one of them."
            )
        if (self.username is None) != (self.password is None):
            raise ValueError("username and password must be provided together")

        # Literal URLs are SSRF-checked at authoring time. Templated URLs
        # (containing `{...}`) are checked at runtime after substitution.
        if "{" not in self.url:
            from optexity.schema.task import validate_callback_url_ssrf

            validate_callback_url_ssrf(self.url)
        return self

    def replace(self, pattern: str, replacement: str):
        self.url = self.url.replace(pattern, replacement)
        if self.api_key is not None:
            self.api_key = self.api_key.replace(pattern, replacement)
        if self.username is not None:
            self.username = self.username.replace(pattern, replacement)
        if self.password is not None:
            self.password = self.password.replace(pattern, replacement)
        return self


class DynamicFormMappingAction(LLMAction):
    """Extract form field keys from the current page, POST them to a customer
    endpoint, and store the returned mapping for later nodes (typically
    ``agentic_task``).
    """

    extraction_instructions: str
    extraction_format: dict
    source: list[Literal["axtree", "screenshot"]] = ["axtree"]
    callback_url: FormMappingCallbackUrl
    include_screenshot: bool = True
    full_page_screenshot: bool = False
    include_axtree: bool = False
    include_live_stream_url: bool = False
    max_wait_time: float = Field(default=60.0, gt=0, le=120)
    output_variable_name: str = "form_values"

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
        raise ValueError("extraction_format must be a dict")

    @field_validator("output_variable_name")
    def validate_output_variable_name(cls, v: str):
        if not v.isidentifier():
            raise ValueError(f"output_variable_name {v!r} is not a valid variable name")
        return v

    @field_validator("source")
    def validate_source(cls, v: list):
        if not v:
            raise ValueError(
                "source must contain at least one of 'axtree', 'screenshot'"
            )
        return v

    def replace(self, pattern: str, replacement: str):
        self.extraction_instructions = self.extraction_instructions.replace(
            pattern, replacement
        )
        self.callback_url.replace(pattern, replacement)
        self.output_variable_name = self.output_variable_name.replace(
            pattern, replacement
        )
        return self
