from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from optexity.schema.automation import ActionNode
from optexity.schema.token_usage import TokenUsage


class ResolutionModel(BaseModel):
    """Strict base model for cache-resolution inputs and audit output."""

    model_config = ConfigDict(extra="forbid")


class ResolutionStrategy(str, Enum):
    DETERMINISTIC_ONLY = "deterministic_only"
    DETERMINISTIC_THEN_LLM = "deterministic_then_llm"


class LocatorAssistedAction(str, Enum):
    CLICK_ELEMENT = "click_element"
    INPUT_TEXT = "input_text"
    SELECT_OPTION = "select_option"
    CHECK = "check"
    UNCHECK = "uncheck"
    HOVER = "hover"


class UnresolvedStepReason(str, Enum):
    INELIGIBLE_SOURCE_STEP = "ineligible_source_step"
    LLM_RESOLUTION_DISABLED = "llm_resolution_disabled"
    LLM_RESOLUTION_FAILED = "llm_resolution_failed"
    MANUAL_REVIEW = "manual_review"


class LLMResolverConfig(ResolutionModel):
    """Policy for resolving only steps deterministic adapters could not map."""

    strategy: ResolutionStrategy = ResolutionStrategy.DETERMINISTIC_THEN_LLM
    provider: str | None = None
    model_name: str | None = None
    agentic_fallback_max_steps: int = Field(default=8, ge=1, le=12)
    action_max_tries: int = Field(default=10, ge=1)
    action_timeout_seconds: float = Field(default=1.0, gt=0)
    end_sleep_time: float = Field(default=0.0, ge=0, le=30)

    @model_validator(mode="after")
    def validate_model_override(self) -> LLMResolverConfig:
        if self.provider is not None and self.model_name is None:
            raise ValueError("provider requires an explicit model_name")
        if self.provider is not None and not self.provider.strip():
            raise ValueError("provider must not be blank")
        if self.model_name is not None and not self.model_name.strip():
            raise ValueError("model_name must not be blank")
        return self


class ResolutionProposalBase(ResolutionModel):
    source_step_number: int = Field(ge=1)
    explanation: str = Field(min_length=1, max_length=1000)


class LocatorAssistedProposal(ResolutionProposalBase):
    resolution_type: Literal["locator_assisted"]
    optexity_action: LocatorAssistedAction


class AgenticFallbackProposal(ResolutionProposalBase):
    resolution_type: Literal["agentic_fallback"]


class ManualReviewProposal(ResolutionProposalBase):
    resolution_type: Literal["manual_review"]


ResolutionProposal = Annotated[
    LocatorAssistedProposal | AgenticFallbackProposal | ManualReviewProposal,
    Field(discriminator="resolution_type"),
]


class ResolutionResponse(ResolutionModel):
    """Provider-friendly structured response for one unresolved source step.

    Some providers do not reliably honor a nested discriminated ``oneOf`` JSON
    schema.  The wire schema is deliberately flat; ``proposal`` rebuilds the
    strict discriminated model after Pydantic validates the conditional fields.
    """

    source_step_number: int = Field(ge=1)
    resolution_type: Literal["locator_assisted", "agentic_fallback", "manual_review"]
    explanation: str = Field(min_length=1, max_length=1000)
    optexity_action: LocatorAssistedAction | None = None

    @model_validator(mode="before")
    @classmethod
    def normalise_provider_shape(cls, value: object) -> object:
        """Accept only closed aliases for known provider schema quirks."""

        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)
        nested = normalized.pop("proposal", None)
        if nested is not None:
            if isinstance(nested, BaseModel):
                nested = nested.model_dump(mode="python")
            if not isinstance(nested, Mapping) or normalized:
                return value
            normalized = dict(nested)
        if "resolution_type" not in normalized and "outcome" in normalized:
            normalized["resolution_type"] = normalized.pop("outcome")
        return normalized

    @model_validator(mode="after")
    def validate_resolution_fields(self) -> ResolutionResponse:
        if self.resolution_type == "locator_assisted":
            if self.optexity_action is None:
                raise ValueError("locator_assisted requires optexity_action")
        elif self.optexity_action is not None:
            raise ValueError("Only locator_assisted may include an Optexity action")
        return self

    @property
    def proposal(self) -> ResolutionProposal:
        common = {
            "source_step_number": self.source_step_number,
            "resolution_type": self.resolution_type,
            "explanation": self.explanation,
        }
        if self.resolution_type == "locator_assisted":
            assert self.optexity_action is not None
            return LocatorAssistedProposal(
                **common,
                optexity_action=self.optexity_action,
            )
        if self.resolution_type == "agentic_fallback":
            return AgenticFallbackProposal(**common)
        return ManualReviewProposal(**common)


class ResolvedStep(ResolutionModel):
    """One executable Optexity node derived from one source action."""

    source_step_number: int = Field(ge=1)
    source_browser_action: str
    resolution_type: Literal["locator_assisted", "agentic_fallback"]
    optexity_action: str
    explanation: str
    model_name: str
    node: ActionNode


class UnresolvedStepResolution(ResolutionModel):
    """A source action retained because it could not be converted safely."""

    source_step_number: int = Field(ge=1)
    source_browser_action: str
    reason: UnresolvedStepReason
    explanation: str


class LLMResolutionResult(ResolutionModel):
    """Resolved nodes, retained blockers, and conversion-time model usage."""

    resolved_steps: list[ResolvedStep] = Field(default_factory=list)
    unresolved_steps: list[UnresolvedStepResolution] = Field(default_factory=list)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    model_name: str | None = None

    @model_validator(mode="after")
    def validate_non_overlapping_steps(self) -> LLMResolutionResult:
        step_numbers = [
            step.source_step_number
            for step in [*self.resolved_steps, *self.unresolved_steps]
        ]
        if len(step_numbers) != len(set(step_numbers)):
            raise ValueError("A source step may have only one resolution outcome")
        return self
