from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from optexity.schema.automation import Automation


class ConversionModel(BaseModel):
    """Base model for the cache-to-Automation conversion contract."""

    model_config = ConfigDict(extra="forbid")


class AutomationConversionStatus(str, Enum):
    DRAFT_REQUIRES_REPLAY_VALIDATION = "draft_requires_replay_validation"


class LocatorSource(str, Enum):
    CHOSEN_CACHE_LOCATOR = "chosen_cache_locator"
    HIGHEST_RANKED_UNVALIDATED_LOCATOR = "highest_ranked_unvalidated_cache_locator"


class ConvertedStep(ConversionModel):
    """Traceability from one observed Browser Use action to one Optexity node."""

    source_step_number: int = Field(ge=1)
    deterministic_candidate_number: int = Field(ge=1)
    browser_use_action: str
    optexity_action: str
    playwright_command: str
    locator_source: LocatorSource
    prompt_fallback_enabled: bool
    recorded_page_transition_after_step: bool


class ExcludedStep(ConversionModel):
    """A source action deliberately retained in the report but not replayed."""

    source_step_number: int = Field(ge=1)
    browser_use_action: str
    cache_decision: str
    explanation: str


class UnconvertedStep(ConversionModel):
    """A source action that prevents complete Automation generation."""

    source_step_number: int | None = Field(default=None, ge=1)
    browser_use_action: str | None = None
    cache_decision: str | None = None
    explanation: str


class AutomationConversionResult(ConversionModel):
    """Validated Automation plus the evidence needed to audit its conversion."""

    status: AutomationConversionStatus
    automation: Automation
    converted_steps: list[ConvertedStep]
    excluded_steps: list[ExcludedStep]
    uses_unvalidated_locators: bool
    unresolved_select_option_steps: list[int]
    literal_password_input_steps: list[int]


class ActionCacheConversionError(ValueError):
    """Raised when a cache cannot be converted without losing workflow behavior."""

    def __init__(self, message: str, problems: list[UnconvertedStep] | None = None):
        self.problems = tuple(problems or [])
        details = "; ".join(
            (
                f"step {problem.source_step_number}: {problem.explanation}"
                if problem.source_step_number is not None
                else problem.explanation
            )
            for problem in self.problems
        )
        super().__init__(f"{message}: {details}" if details else message)
