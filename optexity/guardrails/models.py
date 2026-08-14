import re
from typing import Literal

from pydantic import BaseModel, Field, model_validator

DEFAULT_ALLOWED_ACTIONS = {
    "agentic_task",
    "api_call",
    "assertion",
    "check",
    "click",
    "close_tab",
    "download",
    "extract",
    "go_back",
    "hover",
    "input",
    "key_press",
    "llm_query",
    "navigate",
    "scroll",
    "select",
    "set_variable",
    "sleep",
    "switch_tab",
    "uncheck",
    "upload",
}


class PromptInjectionPolicy(BaseModel):
    enabled: bool = True
    action: Literal["block", "redact", "audit"] = "block"
    # Additional case-insensitive regular expressions owned by the workflow.
    additional_patterns: list[str] = Field(default_factory=list)
    max_untrusted_text_chars: int = Field(default=250_000, ge=1_000, le=2_000_000)

    @model_validator(mode="after")
    def validate_patterns(self):
        for pattern in self.additional_patterns:
            try:
                re.compile(pattern)
            except re.error as error:
                raise ValueError(
                    f"Invalid prompt-injection pattern: {pattern}"
                ) from error
        return self


class DataProtectionPolicy(BaseModel):
    redact_secrets_from_llm: bool = True
    allow_screenshots_to_llm: bool = False
    allow_files_to_llm: bool = False
    store_screenshots_in_trajectory: bool = False
    store_llm_conversations: bool = False
    block_sensitive_data_cross_domain: bool = True
    sensitive_parameter_patterns: list[str] = Field(
        default_factory=lambda: [
            r"password",
            r"passwd",
            r"secret",
            r"token",
            r"api[_-]?key",
            r"credential",
            r"authorization",
            r"ssn",
        ]
    )
    # Empty means the normal allowed_domains set is used.
    sensitive_data_domains: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_patterns(self):
        for pattern in self.sensitive_parameter_patterns:
            try:
                re.compile(pattern)
            except re.error as error:
                raise ValueError(
                    f"Invalid sensitive-parameter pattern: {pattern}"
                ) from error
        return self


class ResourceLimits(BaseModel):
    max_llm_calls: int = Field(default=200, ge=1, le=10_000)
    max_api_calls: int = Field(default=100, ge=0, le=10_000)
    max_agentic_steps: int = Field(default=25, ge=1, le=500)
    max_navigations: int = Field(default=100, ge=1, le=10_000)
    max_tabs: int = Field(default=20, ge=1, le=500)
    max_uploads: int = Field(default=20, ge=0, le=1_000)
    max_downloads: int = Field(default=50, ge=0, le=5_000)


class GuardrailPolicy(BaseModel):
    """A workflow's explicit capability manifest.

    An empty ``allowed_domains`` list means "derive the starting hostname from
    Automation.url".  It never means all domains.
    """

    enabled: bool = True
    mode: Literal["enforce", "audit"] = "enforce"
    allowed_domains: list[str] = Field(default_factory=list)
    allow_subdomains: bool = True
    allowed_schemes: set[Literal["http", "https"]] = Field(
        default_factory=lambda: {"http", "https"}
    )
    allowed_actions: set[str] = Field(
        default_factory=lambda: set(DEFAULT_ALLOWED_ACTIONS)
    )
    blocked_actions: set[str] = Field(
        default_factory=lambda: {
            "evaluate",
            "powershell",
            "private_node",
            "python_script",
            "read_file",
            "replace_file",
            "write_file",
        }
    )
    allowed_upload_roots: list[str] = Field(default_factory=list)
    prompt_injection: PromptInjectionPolicy = Field(
        default_factory=PromptInjectionPolicy
    )
    data_protection: DataProtectionPolicy = Field(default_factory=DataProtectionPolicy)
    limits: ResourceLimits = Field(default_factory=ResourceLimits)

    @model_validator(mode="after")
    def validate_policy(self):
        self.allowed_actions = {
            action.strip().lower() for action in self.allowed_actions
        }
        self.blocked_actions = {
            action.strip().lower() for action in self.blocked_actions
        }
        if "" in self.allowed_actions or "" in self.blocked_actions:
            raise ValueError("Action names must not be empty")
        overlap = self.allowed_actions & self.blocked_actions
        if overlap:
            raise ValueError(
                f"Actions cannot be both allowed and blocked: {sorted(overlap)}"
            )
        return self
