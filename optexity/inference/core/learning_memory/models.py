from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from optexity.schema.automation import Automation
from optexity.schema.token_usage import TokenUsage


def utc_now() -> datetime:
    return datetime.now(UTC)


class LearningMemoryModel(BaseModel):
    """Strict base model for persisted learning-memory data."""

    model_config = ConfigDict(extra="forbid")


class WorkflowVersionStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    DEGRADED = "degraded"
    REJECTED = "rejected"
    QUARANTINED = "quarantined"
    SUPERSEDED = "superseded"


class LearnedStepStrategy(str, Enum):
    LOCATOR = "locator"
    DIRECT = "direct"
    UNSUPPORTED = "unsupported"


class LocatorCapability(str, Enum):
    CLICK = "click"
    INPUT = "input"
    SELECT = "select"
    CHECK = "check"
    UNCHECK = "uncheck"
    HOVER = "hover"
    UPLOAD = "upload"


class LocatorCandidateState(str, Enum):
    TRIAL = "trial"
    ACTIVE = "active"
    DEGRADED = "degraded"


class LocatorValidationOutcome(str, Enum):
    PASSED = "passed"
    NO_MATCH = "no_match"
    MULTIPLE_MATCHES = "multiple_matches"
    CAPABILITY_MISMATCH = "capability_mismatch"
    INVALID_COMMAND = "invalid_command"
    TIMED_OUT = "timed_out"
    ERROR = "error"


class ReplayOutcome(str, Enum):
    PASSED = "passed"
    ACTION_FAILED = "action_failed"
    SIGNATURE_MISMATCH = "signature_mismatch"
    WORKFLOW_FAILED = "workflow_failed"
    DISCOVERY_REGISTERED = "discovery_registered"
    MEMORY_MISS = "memory_miss"
    INFRASTRUCTURE_FAILED = "infrastructure_failed"


class WorkflowIdentity(LearningMemoryModel):
    company_id: str
    workspace_id: str | None = None
    user_id: str
    recording_id: str
    endpoint_name: str
    source_automation_version: str | None = None
    node_path: str


class SourceCompatibility(LearningMemoryModel):
    source_node_fingerprint: str
    source_automation_fingerprint: str
    input_binding_fingerprint: str
    starting_origin: str
    entry_url_fingerprint: str


class PageSignature(LearningMemoryModel):
    """Sensitive-value-free signature used to prove replay equivalence."""

    url: str
    title: str | None = None
    body_text_sha256: str
    body_character_count: int = Field(ge=0)

    def matches(self, other: PageSignature) -> bool:
        return (
            self.url == other.url
            and self.title == other.title
            and self.body_text_sha256 == other.body_text_sha256
            and self.body_character_count == other.body_character_count
        )


class LearningPolicy(LearningMemoryModel):
    soft_validation_target_ms: float = Field(default=50.0, gt=0)
    candidate_timeout_ms: float = Field(default=250.0, gt=0)
    repair_budget_ms: float = Field(default=750.0, gt=0)
    max_alternatives: int = Field(default=2, ge=0, le=10)
    max_versions: int = Field(default=5, ge=1, le=50)


class LocatorCandidateMemory(LearningMemoryModel):
    command: str
    locator_kind: str
    built_from: list[str] = Field(default_factory=list)
    original_rank: int = Field(ge=0)
    appears_dynamic: bool = False
    state: LocatorCandidateState = LocatorCandidateState.TRIAL
    validation_successes: int = Field(default=0, ge=0)
    validation_failures: int = Field(default=0, ge=0)
    full_run_successes: int = Field(default=0, ge=0)
    last_latency_ms: float | None = Field(default=None, ge=0)
    last_failure_reason: str | None = None
    last_validated_at: datetime | None = None


class LearnedStep(LearningMemoryModel):
    node_index: int = Field(ge=0)
    source_step_number: int | None = Field(default=None, ge=1)
    browser_use_action: str | None = None
    optexity_action: str
    strategy: LearnedStepStrategy
    capability: LocatorCapability | None = None
    candidates: list[LocatorCandidateMemory] = Field(default_factory=list)
    chosen_candidate_index: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_locator_contract(self) -> LearnedStep:
        if self.strategy == LearnedStepStrategy.LOCATOR:
            if self.capability is None or not self.candidates:
                raise ValueError(
                    "A locator step requires a capability and locator candidates"
                )
            if self.chosen_candidate_index is not None and (
                self.chosen_candidate_index >= len(self.candidates)
            ):
                raise ValueError("chosen_candidate_index is outside candidates")
        elif self.capability is not None or self.candidates:
            raise ValueError(
                "Only locator steps may carry a capability or locator candidates"
            )
        return self


class VersionStats(LearningMemoryModel):
    successful_full_runs: int = Field(default=0, ge=0)
    failed_full_runs: int = Field(default=0, ge=0)
    consecutive_failures: int = Field(default=0, ge=0)
    last_task_id: str | None = None
    last_run_at: datetime | None = None


class WorkflowVersion(LearningMemoryModel):
    generation: int = Field(ge=1)
    status: WorkflowVersionStatus
    source_task_id: str
    compatibility: SourceCompatibility
    cache_format_version: str
    conversion_status: str
    automation: Automation
    steps: list[LearnedStep]
    source_final_signature: PageSignature
    stats: VersionStats = Field(default_factory=VersionStats)
    created_at: datetime = Field(default_factory=utc_now)
    promoted_at: datetime | None = None
    updated_at: datetime = Field(default_factory=utc_now)
    last_failure_reason: str | None = None

    @model_validator(mode="after")
    def validate_step_order(self) -> WorkflowVersion:
        indexes = [step.node_index for step in self.steps]
        if indexes != list(range(len(indexes))):
            raise ValueError("Learned steps must use consecutive node indexes")
        if len(self.automation.nodes) != len(self.steps):
            raise ValueError("Every learned Automation node requires one step record")
        return self


class WorkflowMemoryDocument(LearningMemoryModel):
    format_version: Literal["1.0"] = "1.0"
    revision: int = Field(default=0, ge=0)
    workflow: WorkflowIdentity
    active_generation: int | None = Field(default=None, ge=1)
    next_generation: int = Field(default=1, ge=1)
    versions: list[WorkflowVersion] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_generations(self) -> WorkflowMemoryDocument:
        generations = [version.generation for version in self.versions]
        if len(generations) != len(set(generations)):
            raise ValueError("Workflow-memory generations must be unique")
        if generations and self.next_generation <= max(generations):
            raise ValueError("next_generation must be newer than every stored version")
        if self.active_generation is not None:
            active = next(
                (
                    version
                    for version in self.versions
                    if version.generation == self.active_generation
                ),
                None,
            )
            if active is None or active.status != WorkflowVersionStatus.ACTIVE:
                raise ValueError("active_generation must reference an active version")
        return self


class LocatorValidationEvent(LearningMemoryModel):
    node_index: int = Field(ge=0)
    candidate_index: int = Field(ge=0)
    command: str
    capability: LocatorCapability
    outcome: LocatorValidationOutcome
    elapsed_ms: float = Field(ge=0)
    exceeded_soft_target: bool
    matched_count: int | None = Field(default=None, ge=0)
    explanation: str | None = None


class RunObservation(LearningMemoryModel):
    task_id: str
    workflow: WorkflowIdentity
    generation: int | None = Field(default=None, ge=1)
    run_kind: Literal["memory_miss", "discovery", "draft_replay", "active_replay"]
    outcome: ReplayOutcome
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime = Field(default_factory=utc_now)
    wall_time_ms: float = Field(default=0, ge=0)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    signature_matches: bool | None = None
    locator_events: list[LocatorValidationEvent] = Field(default_factory=list)
    selected_commands: dict[int, str] = Field(default_factory=dict)
    failure_reason: str | None = None


class RunObservationFile(LearningMemoryModel):
    format_version: Literal["1.0"] = "1.0"
    observations: list[RunObservation] = Field(default_factory=list)
