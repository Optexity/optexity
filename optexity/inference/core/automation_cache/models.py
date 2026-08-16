from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from optexity.schema.automation import ActionNode, Automation


class ConversionModel(BaseModel):
    """Base model for the cache-to-Automation conversion contract."""

    model_config = ConfigDict(extra="forbid")


class AutomationConversionStatus(str, Enum):
    DRAFT_REQUIRES_REPLAY_VALIDATION = "draft_requires_replay_validation"
    HYBRID_DRAFT_REQUIRES_REPLAY_VALIDATION = "hybrid_draft_requires_replay_validation"


class AutomationConversionPlanStatus(str, Enum):
    COMPLETE_DRAFT = "complete_draft"
    COMPLETE_HYBRID_DRAFT = "complete_hybrid_draft"
    PARTIAL_REQUIRES_RESOLUTION = "partial_requires_resolution"


class PlannedStepStatus(str, Enum):
    CONVERTED = "converted"
    EXCLUDED = "excluded"
    UNRESOLVED = "unresolved"


class LocatorSource(str, Enum):
    CHOSEN_CACHE_LOCATOR = "chosen_cache_locator"
    HIGHEST_RANKED_UNVALIDATED_LOCATOR = "highest_ranked_unvalidated_cache_locator"


class ConversionMode(str, Enum):
    CACHED_LOCATOR = "cached_locator"
    NATIVE_DETERMINISTIC = "native_deterministic"
    LLM_LOCATOR_ASSISTED = "llm_locator_assisted"
    LLM_AGENTIC_FALLBACK = "llm_agentic_fallback"


class ConvertedStep(ConversionModel):
    """Traceability from one observed Browser Use action to one Optexity node."""

    source_step_number: int = Field(ge=1)
    deterministic_candidate_number: int | None = Field(default=None, ge=1)
    browser_use_action: str
    optexity_action: str
    conversion_mode: ConversionMode = ConversionMode.CACHED_LOCATOR
    playwright_command: str | None = None
    locator_source: LocatorSource | None = None
    prompt_fallback_enabled: bool
    recorded_page_transition_after_step: bool
    parameter_references: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_conversion_provenance(self) -> ConvertedStep:
        if self.conversion_mode == ConversionMode.CACHED_LOCATOR:
            if (
                self.deterministic_candidate_number is None
                or self.playwright_command is None
                or self.locator_source is None
            ):
                raise ValueError(
                    "A cached-locator step requires candidate and locator provenance"
                )
        elif self.conversion_mode == ConversionMode.NATIVE_DETERMINISTIC:
            if self.deterministic_candidate_number is None:
                raise ValueError(
                    "A native deterministic step requires candidate provenance"
                )
            if self.playwright_command is not None or self.locator_source is not None:
                raise ValueError(
                    "A native deterministic step cannot claim locator provenance"
                )
        elif self.deterministic_candidate_number is not None or any(
            value is not None
            for value in (self.playwright_command, self.locator_source)
        ):
            raise ValueError(
                "An LLM-resolved step cannot claim cached candidate provenance"
            )
        return self


class StepResolution(ConversionModel):
    """One externally resolved source step, normally proposed by the LLM resolver."""

    source_step_number: int = Field(ge=1)
    node: ActionNode
    optexity_action: str
    explanation: str
    conversion_mode: ConversionMode = ConversionMode.LLM_AGENTIC_FALLBACK


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


class PlannedStep(ConversionModel):
    """One source action's ordered outcome during cache conversion planning."""

    source_step_number: int = Field(ge=1)
    browser_use_action: str
    status: PlannedStepStatus
    converted_step: ConvertedStep | None = None
    node: ActionNode | None = None
    excluded_step: ExcludedStep | None = None
    unconverted_step: UnconvertedStep | None = None

    @model_validator(mode="after")
    def validate_outcome_payload(self) -> PlannedStep:
        expected_payload = {
            PlannedStepStatus.CONVERTED: (
                self.converted_step is not None
                and self.node is not None
                and self.excluded_step is None
                and self.unconverted_step is None
            ),
            PlannedStepStatus.EXCLUDED: (
                self.converted_step is None
                and self.node is None
                and self.excluded_step is not None
                and self.unconverted_step is None
            ),
            PlannedStepStatus.UNRESOLVED: (
                self.converted_step is None
                and self.node is None
                and self.excluded_step is None
                and self.unconverted_step is not None
            ),
        }[self.status]
        if not expected_payload:
            raise ValueError(
                "A planned step must contain exactly the payload matching its status"
            )
        return self


class AutomationConversionPlan(ConversionModel):
    """Ordered, lossless plan produced before runnable Automation materialization."""

    status: AutomationConversionPlanStatus
    starting_url: str
    input_parameters: dict[str, list[str | int | float | bool]] = Field(
        default_factory=dict
    )
    ordered_steps: list[PlannedStep]
    global_problems: list[UnconvertedStep] = Field(default_factory=list)
    uses_unvalidated_locators: bool
    unresolved_select_option_steps: list[int]
    literal_password_input_steps: list[int]

    @model_validator(mode="after")
    def validate_plan(self) -> AutomationConversionPlan:
        step_numbers = [step.source_step_number for step in self.ordered_steps]
        if step_numbers != list(range(1, len(step_numbers) + 1)):
            raise ValueError(
                "A conversion plan must preserve every source step in order"
            )
        if self.problems or not self.converted_steps:
            expected_status = AutomationConversionPlanStatus.PARTIAL_REQUIRES_RESOLUTION
        elif any(
            step.conversion_mode
            in {
                ConversionMode.LLM_LOCATOR_ASSISTED,
                ConversionMode.LLM_AGENTIC_FALLBACK,
            }
            for step in self.converted_steps
        ):
            expected_status = AutomationConversionPlanStatus.COMPLETE_HYBRID_DRAFT
        else:
            expected_status = AutomationConversionPlanStatus.COMPLETE_DRAFT
        if self.status != expected_status:
            raise ValueError("Conversion-plan status does not match its outcomes")
        return self

    @property
    def complete(self) -> bool:
        return self.status in {
            AutomationConversionPlanStatus.COMPLETE_DRAFT,
            AutomationConversionPlanStatus.COMPLETE_HYBRID_DRAFT,
        }

    @property
    def converted_steps(self) -> list[ConvertedStep]:
        return [
            step.converted_step
            for step in self.ordered_steps
            if step.converted_step is not None
        ]

    @property
    def excluded_steps(self) -> list[ExcludedStep]:
        return [
            step.excluded_step
            for step in self.ordered_steps
            if step.excluded_step is not None
        ]

    @property
    def nodes(self) -> list[ActionNode]:
        return [step.node for step in self.ordered_steps if step.node is not None]

    @property
    def problems(self) -> list[UnconvertedStep]:
        return [
            *(
                step.unconverted_step
                for step in self.ordered_steps
                if step.unconverted_step is not None
            ),
            *self.global_problems,
        ]


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

    def __init__(
        self,
        message: str,
        problems: list[UnconvertedStep] | None = None,
        *,
        plan: AutomationConversionPlan | None = None,
    ):
        self.problems = tuple(problems or [])
        self.plan = plan
        details = "; ".join(
            (
                f"step {problem.source_step_number}: {problem.explanation}"
                if problem.source_step_number is not None
                else problem.explanation
            )
            for problem in self.problems
        )
        super().__init__(f"{message}: {details}" if details else message)
