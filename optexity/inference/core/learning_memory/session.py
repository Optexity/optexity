from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qsl, unquote, urlparse

from browser_use.agent.history_compiler import (
    BrowserUseActionCache,
    DeterministicStepCandidate,
)

from optexity.inference.core.automation_cache.automatic_conversion import (
    automatically_convert_action_cache,
    write_automatic_conversion_artifacts,
)
from optexity.inference.core.automation_cache.models import (
    ActionCacheConversionError,
    AutomationConversionResult,
    ConversionMode,
)
from optexity.inference.core.automation_cache.resolution_models import (
    LLMResolverConfig,
    ResolutionStrategy,
)
from optexity.inference.core.learning_memory.capabilities import (
    LocatorResolutionError,
    locator_action,
    prepare_action_node,
    verify_action_effect,
)
from optexity.inference.core.learning_memory.judge import (
    ReplayJudgeUnavailable,
    judge_learning_replay,
)
from optexity.inference.core.learning_memory.models import (
    LearnedStep,
    LearnedStepStrategy,
    LearningPolicy,
    LocatorCandidateMemory,
    LocatorCandidateState,
    LocatorValidationEvent,
    LocatorValidationOutcome,
    PageSignature,
    ReplayOutcome,
    RunObservation,
    SourceCompatibility,
    WorkflowIdentity,
    WorkflowVersion,
    WorkflowVersionStatus,
    utc_now,
)
from optexity.inference.core.learning_memory.store import (
    LearningMemoryStoreError,
    LocalLearningMemoryStore,
)
from optexity.inference.infra.browser import Browser
from optexity.inference.infra.browser_health import (
    is_browser_session_poisoned_error,
    is_driver_closed_error,
)
from optexity.schema.automation import ActionNode, Automation
from optexity.schema.memory import Memory
from optexity.schema.task import Task
from optexity.schema.token_usage import TokenUsage
from optexity.utils.settings import settings

logger = logging.getLogger(__name__)

LEARNING_SESSION_STATE_KEY = "learning_memory_session"
ACTION_CACHE_FILENAME = "browser_use_action_cache.json"
_PARAMETER_REFERENCE_PATTERN = re.compile(
    r"\{(?P<name>[A-Za-z_]\w*)\[(?P<index>\d+)\]\}"
)


class LearningReplayError(RuntimeError):
    """A learned replay failed and must not continue from its dirty page state."""


@dataclass(slots=True)
class PendingDiscovery:
    workflow: WorkflowIdentity
    compatibility: SourceCompatibility
    cache_path: Path
    source_task_id: str
    started_at: datetime
    started_monotonic: float
    token_usage_before: TokenUsage


@dataclass(slots=True)
class PendingReplay:
    workflow: WorkflowIdentity
    version: WorkflowVersion
    run_kind: Literal["draft_replay", "active_replay", "rollback_replay"]
    task_instruction: str
    started_at: datetime
    started_monotonic: float
    replay_token_usage: TokenUsage
    locator_events: list[LocatorValidationEvent] = field(default_factory=list)
    selected_candidate_indexes: dict[int, int] = field(default_factory=dict)
    selected_commands: dict[int, str] = field(default_factory=dict)


def learning_memory_enabled() -> bool:
    return bool(settings.LEARNING_MEMORY_ENABLED)


def create_learning_session(task: Task) -> LearningMemorySession | None:
    if not learning_memory_enabled():
        return None
    root = settings.LEARNING_MEMORY_DIRECTORY or (
        task.save_directory / "_learning_memory"
    )
    policy = LearningPolicy(
        soft_validation_target_ms=settings.LEARNING_MEMORY_SOFT_TARGET_MS,
        candidate_timeout_ms=settings.LEARNING_MEMORY_CANDIDATE_TIMEOUT_MS,
        readiness_wait_ms=settings.LEARNING_MEMORY_READINESS_WAIT_MS,
        repair_budget_ms=settings.LEARNING_MEMORY_REPAIR_BUDGET_MS,
        max_alternatives=settings.LEARNING_MEMORY_MAX_ALTERNATIVES,
        max_versions=settings.LEARNING_MEMORY_MAX_VERSIONS,
    )
    return LearningMemorySession(
        task=task,
        store=LocalLearningMemoryStore(root, max_versions=policy.max_versions),
        policy=policy,
    )


def get_learning_session(memory: Memory) -> LearningMemorySession | None:
    session = memory.state.get(LEARNING_SESSION_STATE_KEY)
    return session if isinstance(session, LearningMemorySession) else None


def is_cacheable_agentic_node(node: ActionNode) -> bool:
    interaction = node.interaction_action
    return bool(
        interaction is not None
        and interaction.agentic_task is not None
        and interaction.agentic_task.backend == "browser_use"
    )


class LearningMemorySession:
    """Coordinates discovery, strict replay, and post-run promotion.

    A source-page signature proves that replay reproduced the accepted discovery
    run. It does not independently prove the user's semantic intent. A future
    version should allow explicit deterministic task postconditions; until then
    exact signature matching is intentionally conservative.
    """

    def __init__(
        self,
        *,
        task: Task,
        store: LocalLearningMemoryStore,
        policy: LearningPolicy,
    ):
        self.task = task
        self.store = store
        self.policy = policy
        self.pending_discoveries: list[PendingDiscovery] = []
        self.pending_replays: list[PendingReplay] = []
        self.failed_generations: set[tuple[str, int]] = set()
        self.memory_misses: list[tuple[WorkflowIdentity, datetime]] = []
        self._finalized = False

        assert task.automation is not None
        self.source_automation_fingerprint = _fingerprint(
            _parameter_agnostic_automation_payload(task.automation)
        )

    def workflow_context(
        self,
        node: ActionNode,
        node_path: str,
        memory: Memory,
        entry_url: str,
    ) -> tuple[WorkflowIdentity, SourceCompatibility]:
        assert self.task.automation is not None
        workflow = WorkflowIdentity(
            company_id=str(self.task.company_id),
            workspace_id=(
                str(self.task.workspace_id)
                if self.task.workspace_id is not None
                else None
            ),
            user_id=str(self.task.user_id),
            recording_id=str(self.task.recording_id),
            endpoint_name=self.task.endpoint_name,
            source_automation_version=self.task.version,
            node_path=node_path,
        )
        parsed_url = urlparse(self.task.automation.url)
        origin = f"{parsed_url.scheme}://{parsed_url.netloc}"
        compatibility = SourceCompatibility(
            source_node_fingerprint=_fingerprint(node.model_dump(mode="json")),
            source_automation_fingerprint=self.source_automation_fingerprint,
            parameter_schema_fingerprint=_fingerprint(
                {
                    "input_parameters": _parameter_contract(self.task.input_parameters),
                    "secure_parameters": _parameter_contract(
                        self.task.secure_parameters
                    ),
                    "generated_parameters": _parameter_contract(
                        memory.variables.generated_variables
                    ),
                }
            ),
            starting_origin=origin,
            entry_url_fingerprint=_fingerprint(
                _stable_entry_context(
                    self.task.automation.url,
                    entry_url,
                    input_parameters=self.task.input_parameters,
                    generated_parameters=memory.variables.generated_variables,
                )
            ),
        )
        return workflow, compatibility

    async def replay_if_available(
        self,
        *,
        node_path: str,
        workflow: WorkflowIdentity,
        compatibility: SourceCompatibility,
        source_node: ActionNode,
        memory: Memory,
        browser: Browser,
        full_automation: list,
        execute_node: Callable[[ActionNode, int], Awaitable[None]],
    ) -> bool:
        try:
            versions = self.store.select_replay_versions(workflow, compatibility)
        except LearningMemoryStoreError:
            # A stale/corrupt local memory must never block the original agent.
            logger.exception(
                "Ignoring unreadable learning memory for %s %s",
                workflow.recording_id,
                node_path,
            )
            versions = []
        if not versions:
            self.memory_misses.append((workflow, utc_now()))
            return False

        task_instruction = await _resolved_agentic_instruction(
            source_node,
            self.task,
            memory,
        )
        for version in versions:
            replay_outcome = await self._replay_version(
                workflow=workflow,
                version=version,
                task_instruction=task_instruction,
                memory=memory,
                browser=browser,
                full_automation=full_automation,
                execute_node=execute_node,
            )
            if replay_outcome == "passed":
                return True
            if replay_outcome == "fallback":
                return False
        return False

    async def _replay_version(
        self,
        *,
        workflow: WorkflowIdentity,
        version: WorkflowVersion,
        task_instruction: str,
        memory: Memory,
        browser: Browser,
        full_automation: list,
        execute_node: Callable[[ActionNode, int], Awaitable[None]],
    ) -> Literal["passed", "try_next", "fallback"]:

        run_kind: Literal["draft_replay", "active_replay", "rollback_replay"]
        if version.status == WorkflowVersionStatus.DRAFT:
            run_kind = "draft_replay"
        elif version.status == WorkflowVersionStatus.ACTIVE:
            run_kind = "active_replay"
        else:
            run_kind = "rollback_replay"
        started_at = utc_now()
        started_monotonic = time.monotonic()
        token_usage_before = memory.token_usage.model_copy(deep=True)
        events: list[LocatorValidationEvent] = []
        selected_indexes: dict[int, int] = {}
        selected_commands: dict[int, str] = {}
        action_started = False
        current_node_index: int | None = None

        try:
            for node_index, cached_node in enumerate(version.automation.nodes):
                current_node_index = node_index
                if not isinstance(cached_node, ActionNode):
                    raise LearningReplayError(
                        "Learning replay currently supports ordered ActionNode entries"
                    )
                learned_step = version.steps[node_index]
                prepared = await prepare_action_node(
                    cached_node,
                    learned_step,
                    browser,
                    self.policy,
                )
                events.extend(prepared.events)
                if prepared.selected_candidate_index is not None:
                    selected_indexes[node_index] = prepared.selected_candidate_index
                if prepared.selected_command is not None:
                    selected_commands[node_index] = prepared.selected_command
                full_automation.append(prepared.node.model_dump())
                action_started = True
                await execute_node(prepared.node, 1)
                await verify_action_effect(
                    prepared.node,
                    prepared.selected_command,
                    browser,
                    timeout_ms=self.policy.candidate_timeout_ms,
                )
        except Exception as exc:
            if isinstance(exc, LocatorResolutionError):
                events.extend(exc.events)
            reason = f"{type(exc).__name__}: {exc}"
            infrastructure_failure = is_driver_closed_error(exc)
            retry_events = [
                event
                for event in events
                if event.validation_attempt == "after_readiness_wait"
            ]
            page_not_ready = bool(retry_events) and all(
                event.outcome
                in {
                    LocatorValidationOutcome.PAGE_NOT_READY,
                    LocatorValidationOutcome.TIMED_OUT,
                }
                for event in retry_events
            )
            if not infrastructure_failure and not page_not_ready:
                try:
                    self.store.record_failure(
                        workflow,
                        version.generation,
                        task_id=str(self.task.task_id),
                        reason=reason,
                        locator_events=events,
                        selected_candidate_indexes=selected_indexes,
                        failed_node_index=(
                            current_node_index if action_started else None
                        ),
                        expected_version=version,
                    )
                except Exception:
                    logger.exception("Failed to persist learned replay failure")
                self.failed_generations.add((workflow.node_path, version.generation))
            self._append_observation(
                RunObservation(
                    task_id=str(self.task.task_id),
                    workflow=workflow,
                    generation=version.generation,
                    run_kind=run_kind,
                    outcome=(
                        ReplayOutcome.INFRASTRUCTURE_FAILED
                        if infrastructure_failure
                        else (
                            ReplayOutcome.PAGE_NOT_READY
                            if page_not_ready
                            else ReplayOutcome.ACTION_FAILED
                        )
                    ),
                    started_at=started_at,
                    completed_at=utc_now(),
                    wall_time_ms=(time.monotonic() - started_monotonic) * 1000,
                    token_usage=memory.token_usage - token_usage_before,
                    locator_events=events,
                    selected_commands=selected_commands,
                    failure_reason=reason,
                )
            )
            if infrastructure_failure:
                raise
            if isinstance(exc, LocatorResolutionError) and not action_started:
                logger.info(
                    "Cached locator validation failed before execution for "
                    "generation %d (%s)",
                    version.generation,
                    "page not ready" if page_not_ready else "hard locator miss",
                )
                return "fallback" if page_not_ready else "try_next"
            raise LearningReplayError(
                "Cached replay failed; the next run will use fresh agentic discovery"
            ) from exc

        pending = PendingReplay(
            workflow=workflow,
            version=version,
            run_kind=run_kind,
            task_instruction=task_instruction,
            started_at=started_at,
            started_monotonic=started_monotonic,
            replay_token_usage=memory.token_usage - token_usage_before,
            locator_events=events,
            selected_candidate_indexes=selected_indexes,
            selected_commands=selected_commands,
        )
        self.pending_replays.append(pending)
        return "passed"

    def record_discovery(
        self,
        *,
        workflow: WorkflowIdentity,
        compatibility: SourceCompatibility,
        cache_path: Path,
        started_at: datetime,
        started_monotonic: float,
        token_usage_before: TokenUsage,
    ) -> None:
        if not cache_path.exists():
            logger.warning(
                "Browser Use completed but no action cache exists at %s", cache_path
            )
            return
        self.pending_discoveries.append(
            PendingDiscovery(
                workflow=workflow,
                compatibility=compatibility,
                cache_path=cache_path,
                source_task_id=str(self.task.task_id),
                started_at=started_at,
                started_monotonic=started_monotonic,
                token_usage_before=token_usage_before,
            )
        )

    async def finalize_success(self, browser: Browser, memory: Memory) -> None:
        if self._finalized:
            return
        if not self.pending_replays and not self.pending_discoveries:
            self._append_memory_misses()
            self._finalized = True
            return
        signature: PageSignature | None = None
        try:
            signature = await capture_page_signature(browser)
        except Exception:
            logger.exception(
                "Could not capture supplemental learning-memory page signature"
            )
            if not self.pending_replays:
                self._append_memory_misses()
                self._finalized = True
                return

        for pending in self.pending_replays:
            signature_matches = (
                signature is not None
                and pending.version.source_final_signature.matches(signature)
            )
            try:
                judge_evidence = await judge_learning_replay(
                    task_instruction=pending.task_instruction,
                    task=self.task,
                    browser=browser,
                    memory=memory,
                    model_name=settings.LEARNING_MEMORY_JUDGE_LLM_MODEL,
                )
            except ReplayJudgeUnavailable as exc:
                reason = str(exc)
                self._append_observation(
                    _replay_observation(
                        pending,
                        task_id=str(self.task.task_id),
                        outcome=ReplayOutcome.JUDGE_UNAVAILABLE,
                        signature_matches=signature_matches,
                        failure_reason=reason,
                    )
                )
                self._finalized = True
                raise LearningReplayError(reason) from exc

            pending.replay_token_usage += judge_evidence.token_usage
            judgement = judge_evidence.judgement
            if not judgement.successful:
                reason = f"Replay judge rejected the workflow: {judgement.reasoning}"
                try:
                    self.store.record_failure(
                        pending.workflow,
                        pending.version.generation,
                        task_id=str(self.task.task_id),
                        reason=reason,
                        expected_version=pending.version,
                    )
                except Exception:
                    logger.exception("Failed to persist replay-judge rejection")
                self.failed_generations.add(
                    (pending.workflow.node_path, pending.version.generation)
                )
                self._append_observation(
                    _replay_observation(
                        pending,
                        task_id=str(self.task.task_id),
                        outcome=ReplayOutcome.JUDGE_REJECTED,
                        signature_matches=signature_matches,
                        judge_verdict=False,
                        judge_reasoning=judgement.reasoning,
                        failure_reason=reason,
                    )
                )
                self._finalized = True
                raise LearningReplayError(reason)

            updated = _apply_successful_locator_choices(pending)
            store_failure: str | None = None
            try:
                self.store.record_success(
                    pending.workflow,
                    updated,
                    task_id=str(self.task.task_id),
                    promote=True,
                )
            except Exception as exc:  # noqa: BLE001 - persistence is best-effort
                # The replay and final signature already passed. A concurrent
                # memory update must not change the user-visible task outcome.
                store_failure = f"Learning-memory update skipped: {exc}"
                logger.warning(store_failure)
            self._append_observation(
                _replay_observation(
                    pending,
                    task_id=str(self.task.task_id),
                    outcome=ReplayOutcome.PASSED,
                    signature_matches=signature_matches,
                    judge_verdict=True,
                    judge_reasoning=judgement.reasoning,
                    failure_reason=store_failure,
                )
            )

        for discovery in self.pending_discoveries:
            if signature is None:
                logger.warning(
                    "Skipping learning-memory draft without a discovery signature"
                )
                continue
            try:
                version = await self._compile_discovery(
                    discovery,
                    signature,
                    memory,
                )
                created = self.store.create_draft(discovery.workflow, version)
                self._append_observation(
                    RunObservation(
                        task_id=str(self.task.task_id),
                        workflow=discovery.workflow,
                        generation=created.generation,
                        run_kind="discovery",
                        outcome=ReplayOutcome.DISCOVERY_REGISTERED,
                        started_at=discovery.started_at,
                        completed_at=utc_now(),
                        wall_time_ms=(time.monotonic() - discovery.started_monotonic)
                        * 1000,
                        token_usage=(memory.token_usage - discovery.token_usage_before),
                    )
                )
                logger.info(
                    "Registered learning-memory draft generation %d for %s %s",
                    created.generation,
                    discovery.workflow.recording_id,
                    discovery.workflow.node_path,
                )
            except Exception:
                # The Browser Use task itself already succeeded. Learning is
                # best-effort and must not turn that user-visible run into failure.
                logger.exception(
                    "Failed to register learning-memory draft from %s",
                    discovery.cache_path,
                )

        self._append_memory_misses()
        self._finalized = True

    async def finalize_failure(self, error: BaseException) -> None:
        if self._finalized:
            return
        reason = f"{type(error).__name__}: {error}"
        infrastructure_failure = is_browser_session_poisoned_error(error)
        for pending in self.pending_replays:
            key = (pending.workflow.node_path, pending.version.generation)
            if key in self.failed_generations:
                continue
            if not infrastructure_failure:
                try:
                    self.store.record_failure(
                        pending.workflow,
                        pending.version.generation,
                        task_id=str(self.task.task_id),
                        reason=reason,
                        locator_events=pending.locator_events,
                        selected_candidate_indexes=pending.selected_candidate_indexes,
                        expected_version=pending.version,
                    )
                except Exception:
                    logger.exception("Failed to persist workflow-level replay failure")
            self._append_observation(
                _replay_observation(
                    pending,
                    task_id=str(self.task.task_id),
                    outcome=(
                        ReplayOutcome.INFRASTRUCTURE_FAILED
                        if infrastructure_failure
                        else ReplayOutcome.WORKFLOW_FAILED
                    ),
                    signature_matches=None,
                    failure_reason=reason,
                )
            )
        self._finalized = True

    async def _compile_discovery(
        self,
        discovery: PendingDiscovery,
        signature: PageSignature,
        memory: Memory,
    ) -> WorkflowVersion:
        cache = BrowserUseActionCache.model_validate_json(
            discovery.cache_path.read_text(encoding="utf-8")
        )
        resolver_config = LLMResolverConfig(
            strategy=ResolutionStrategy(settings.LEARNING_MEMORY_RESOLUTION_STRATEGY),
            model_name=settings.LEARNING_MEMORY_RESOLVER_LLM_MODEL,
            agentic_fallback_max_steps=(
                settings.LEARNING_MEMORY_AGENTIC_FALLBACK_MAX_STEPS
            ),
        )
        outcome = await asyncio.to_thread(
            automatically_convert_action_cache,
            cache,
            resolver_config=resolver_config,
            source_input_parameters=self.task.input_parameters,
            inherited_provider=self.task.llm_provider,
            inherited_model_name=self.task.llm_model_name,
            allow_unvalidated_locators=True,
            allow_unresolved_select_options=(
                settings.LEARNING_MEMORY_ALLOW_UNRESOLVED_SELECT_OPTIONS
            ),
            allow_literal_password_inputs=(
                settings.LEARNING_MEMORY_ALLOW_LITERAL_PASSWORD_INPUTS
            ),
        )
        memory.token_usage += outcome.resolution.token_usage
        await asyncio.to_thread(
            write_automatic_conversion_artifacts,
            outcome,
            discovery.cache_path.parent,
        )
        if outcome.conversion is None:
            raise ActionCacheConversionError(
                "Automatic conversion retained unresolved source steps",
                outcome.final_plan.problems,
                plan=outcome.final_plan,
            )
        conversion = outcome.conversion
        _validate_parameterized_conversion(conversion, self.task)
        strict_automation = _strict_automation(conversion.automation)
        missing_runtime_parameters = (
            strict_automation.parameters.input_parameters.keys()
            - self.task.input_parameters.keys()
        )
        if missing_runtime_parameters:
            raise ActionCacheConversionError(
                "Generated Automation requires runtime input parameters that the "
                f"source workflow does not declare: {sorted(missing_runtime_parameters)}"
            )
        candidates_by_number = {
            candidate.candidate_number: candidate
            for candidate in cache.deterministic_step_candidates
        }
        learned_steps: list[LearnedStep] = []
        if len(conversion.converted_steps) != len(strict_automation.nodes):
            raise ValueError(
                "Conversion report and generated Automation have different lengths"
            )

        for node_index, (node, converted) in enumerate(
            zip(strict_automation.nodes, conversion.converted_steps, strict=True)
        ):
            if not isinstance(node, ActionNode):
                raise TypeError("Learning memory currently requires ActionNode output")
            action_data = locator_action(node)
            candidate_number = getattr(
                converted, "deterministic_candidate_number", None
            )
            candidate = (
                candidates_by_number.get(candidate_number)
                if candidate_number is not None
                else None
            )
            if action_data is None:
                interaction = node.interaction_action
                strategy = (
                    LearnedStepStrategy.AGENTIC
                    if interaction is not None and interaction.agentic_task is not None
                    else LearnedStepStrategy.DIRECT
                )
                learned_steps.append(
                    LearnedStep(
                        node_index=node_index,
                        source_step_number=getattr(
                            converted, "source_step_number", None
                        ),
                        browser_use_action=getattr(
                            converted, "browser_use_action", None
                        ),
                        optexity_action=converted.optexity_action,
                        strategy=strategy,
                    )
                )
                continue
            if not isinstance(candidate, DeterministicStepCandidate):
                if (
                    candidate is not None
                    or converted.conversion_mode != ConversionMode.LLM_LOCATOR_ASSISTED
                ):
                    raise ValueError(
                        f"Locator node {node_index} has no cache candidate provenance"
                    )
                learned_steps.append(
                    LearnedStep(
                        node_index=node_index,
                        source_step_number=converted.source_step_number,
                        browser_use_action=converted.browser_use_action,
                        optexity_action=converted.optexity_action,
                        strategy=LearnedStepStrategy.LOCATOR_LLM,
                    )
                )
                continue
            _, _, requirements = action_data
            commands = [
                option.playwright_command
                for option in candidate.playwright_locator_options
            ]
            selected_command = converted.playwright_command
            if selected_command not in commands:
                raise ValueError(
                    f"Converted locator for node {node_index} is not cache-derived"
                )
            learned_steps.append(
                LearnedStep(
                    node_index=node_index,
                    source_step_number=converted.source_step_number,
                    browser_use_action=converted.browser_use_action,
                    optexity_action=converted.optexity_action,
                    strategy=LearnedStepStrategy.LOCATOR,
                    capability=requirements.capability,
                    candidates=[
                        LocatorCandidateMemory(
                            command=option.playwright_command,
                            locator_kind=option.locator_name,
                            built_from=option.built_from,
                            original_rank=index,
                            appears_dynamic=option.appears_dynamic,
                        )
                        for index, option in enumerate(
                            candidate.playwright_locator_options
                        )
                    ],
                    chosen_candidate_index=commands.index(selected_command),
                )
            )

        return WorkflowVersion(
            generation=1,
            status=WorkflowVersionStatus.DRAFT,
            source_task_id=discovery.source_task_id,
            compatibility=discovery.compatibility,
            cache_format_version=cache.cache_format_version,
            conversion_status=conversion.status.value,
            automation=strict_automation,
            steps=learned_steps,
            source_final_signature=signature,
        )

    def _append_observation(self, observation: RunObservation) -> None:
        try:
            path = self.store.append_observation(self.task.logs_directory, observation)
            logger.info("Saved learning-memory observation to %s", path)
        except Exception:
            logger.exception("Failed to persist learning-memory observation")

    def _append_memory_misses(self) -> None:
        for workflow, started_at in self.memory_misses:
            self._append_observation(
                RunObservation(
                    task_id=str(self.task.task_id),
                    workflow=workflow,
                    run_kind="memory_miss",
                    outcome=ReplayOutcome.MEMORY_MISS,
                    started_at=started_at,
                    completed_at=utc_now(),
                )
            )


async def capture_page_signature(browser: Browser) -> PageSignature:
    page = await browser.get_current_page()
    if page is None:
        raise LearningReplayError("Cannot capture signature without an active page")
    body_text = await page.locator("body").inner_text(timeout=2000)
    normalized_body = " ".join(body_text.split())
    return PageSignature(
        url=await browser.get_current_page_url(),
        title=await browser.get_current_page_title(),
        body_text_sha256=hashlib.sha256(normalized_body.encode("utf-8")).hexdigest(),
        body_character_count=len(normalized_body),
    )


def _strict_automation(automation: Automation) -> Automation:
    strict = automation.model_copy(deep=True)
    for node in strict.nodes:
        if not isinstance(node, ActionNode):
            continue
        action_data = locator_action(node)
        if action_data is None:
            continue
        _, action, _ = action_data
        if action.command is None:
            if (
                action.skip_command
                and not action.skip_prompt
                and action.prompt_instructions
            ):
                # A schema-constrained resolver may deliberately leave one
                # element action to Optexity's narrow locator LLM. It remains a
                # hybrid draft and cannot be promoted as a zero-token replay.
                continue
            raise ValueError(
                "A learned locator action requires either a cache command or a "
                "prompt-only locator strategy"
            )
        action.skip_command = False
        action.skip_prompt = True
        action.assert_locator_presence = True
        # Candidate selection already performs bounded locator/actionability
        # checks. Repeating the same command ten times would only add latency;
        # one real execution attempt is enough before the version is rejected.
        assert node.interaction_action is not None
        node.interaction_action.max_tries = 1
    return Automation.model_validate(strict.model_dump(mode="json"))


def _apply_successful_locator_choices(pending: PendingReplay) -> WorkflowVersion:
    updated = pending.version.model_copy(deep=True)
    now = utc_now()
    events_by_node: dict[int, list[LocatorValidationEvent]] = {}
    for event in pending.locator_events:
        events_by_node.setdefault(event.node_index, []).append(event)

    for step in updated.steps:
        for event in events_by_node.get(step.node_index, []):
            candidate = step.candidates[event.candidate_index]
            candidate.last_latency_ms = event.elapsed_ms
            candidate.last_validated_at = now
            if event.outcome == LocatorValidationOutcome.PASSED:
                candidate.validation_successes += 1
            else:
                candidate.validation_failures += 1
                candidate.last_failure_reason = event.explanation or event.outcome.value

        selected_index = pending.selected_candidate_indexes.get(step.node_index)
        if selected_index is None:
            continue
        step.chosen_candidate_index = selected_index
        for candidate_index, candidate in enumerate(step.candidates):
            candidate.state = (
                LocatorCandidateState.ACTIVE
                if candidate_index == selected_index
                else LocatorCandidateState.TRIAL
            )
        selected = step.candidates[selected_index]
        selected.full_run_successes += 1
        node = updated.automation.nodes[step.node_index]
        assert isinstance(node, ActionNode)
        action_data = locator_action(node)
        assert action_data is not None
        _, action, _ = action_data
        action.command = selected.command
    return updated


def _replay_observation(
    pending: PendingReplay,
    *,
    task_id: str,
    outcome: ReplayOutcome,
    signature_matches: bool | None,
    judge_verdict: bool | None = None,
    judge_reasoning: str | None = None,
    failure_reason: str | None = None,
) -> RunObservation:
    return RunObservation(
        task_id=task_id,
        workflow=pending.workflow,
        generation=pending.version.generation,
        run_kind=pending.run_kind,
        outcome=outcome,
        started_at=pending.started_at,
        completed_at=utc_now(),
        wall_time_ms=(time.monotonic() - pending.started_monotonic) * 1000,
        token_usage=pending.replay_token_usage,
        signature_matches=signature_matches,
        locator_events=pending.locator_events,
        selected_commands=pending.selected_commands,
        judge_verdict=judge_verdict,
        judge_reasoning=judge_reasoning,
        failure_reason=failure_reason,
    )


async def _resolved_agentic_instruction(
    source_node: ActionNode,
    task: Task,
    memory: Memory,
) -> str:
    """Resolve current non-secret values for the final semantic judge."""

    resolved = source_node.model_copy(deep=True)
    await resolved.replace_variables(task.input_parameters)
    await resolved.replace_variables(memory.variables.generated_variables)
    interaction = resolved.interaction_action
    if interaction is None or interaction.agentic_task is None:
        raise LearningReplayError("Learning replay lost its source agentic task")
    instruction = interaction.agentic_task.task.strip()
    if not instruction:
        raise LearningReplayError("Learning replay source task is blank")
    return instruction


def _fingerprint(payload: Any) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _parameter_contract(parameters: dict[str, list[Any]]) -> dict[str, Any]:
    """Fingerprint binding shape while excluding all runtime parameter values."""

    return {
        name: {
            "arity": len(values),
            "types": [_parameter_value_type(value) for value in values],
        }
        for name, values in sorted(parameters.items())
    }


def _parameter_agnostic_automation_payload(automation: Automation) -> dict[str, Any]:
    """Fingerprint workflow structure while excluding declared parameter values."""

    payload = automation.model_dump(mode="json")
    parameters = automation.parameters
    payload["parameters"] = {
        "input_parameters": _parameter_contract(parameters.input_parameters),
        "secure_parameters": _parameter_contract(parameters.secure_parameters),
        "generated_parameters": _parameter_contract(parameters.generated_parameters),
    }
    return payload


def _validate_parameterized_conversion(
    conversion: AutomationConversionResult,
    task: Task,
) -> None:
    """Reject learned actions that cannot be rebound by the next task run."""

    available = task.input_parameters
    declared = conversion.automation.parameters.input_parameters
    unavailable = sorted(set(declared) - set(available))
    if unavailable:
        raise ValueError(
            "Converted memory introduced parameters unavailable to the workflow: "
            + ", ".join(unavailable)
        )

    value_actions = {"input_text", "select_option", "upload_file", "search"}
    for step in conversion.converted_steps:
        if step.optexity_action in value_actions and not step.parameter_references:
            raise ValueError(
                f"Converted {step.optexity_action} step {step.source_step_number} "
                "retained a runtime value instead of a parameter reference"
            )
        for reference in step.parameter_references:
            match = _PARAMETER_REFERENCE_PATTERN.fullmatch(reference)
            if match is None:
                raise ValueError(f"Invalid learned parameter reference: {reference!r}")
            name = match.group("name")
            index = int(match.group("index"))
            if name not in available or index >= len(available[name]):
                raise ValueError(
                    f"Learned parameter {reference!r} is unavailable in this workflow"
                )


def _parameter_value_type(value: Any) -> str:
    for secure_provider in ("onepassword", "amazon_secrets_manager", "totp"):
        if getattr(value, secure_provider, None) is not None:
            return f"secure:{secure_provider}"
    return type(value).__name__


def _stable_entry_context(
    starting_url: str,
    entry_url: str,
    *,
    input_parameters: dict[str, list[Any]] | None = None,
    generated_parameters: dict[str, list[Any]] | None = None,
) -> dict[str, Any]:
    """Build a value-agnostic but state-sensitive page-entry identity.

    Browser startup can expose either ``about:blank`` or Chromium's certificate
    error page before the same initial navigation settles. Treating those as
    different workflows prevents memory reuse.

    Runtime parameter values can also legitimately appear in a URL path or query.
    Exact URL matching would create a separate compatibility version for every
    value. Replace only complete URL components whose value maps unambiguously to
    one declared parameter. Static route, origin, and query-key differences remain
    part of the safety boundary.
    """

    references_by_value = _unique_runtime_parameter_references(
        input_parameters or {},
        generated_parameters or {},
    )

    if entry_url in {"about:blank", "chrome-error://chromewebdata/"}:
        return {
            "state": "initial_navigation_pending",
            "target_url": _parameter_agnostic_url(
                starting_url,
                references_by_value,
            ),
        }
    return {
        "state": "page",
        "url": _parameter_agnostic_url(entry_url, references_by_value),
    }


def _unique_runtime_parameter_references(
    input_parameters: dict[str, list[Any]],
    generated_parameters: dict[str, list[Any]],
) -> dict[str, str]:
    """Return only value-to-parameter mappings that are safe to canonicalize."""

    references: dict[str, list[str]] = {}
    for parameters in (input_parameters, generated_parameters):
        for name, values in sorted(parameters.items()):
            for index, value in enumerate(values):
                if not isinstance(value, (str, int, float, bool)):
                    continue
                text = str(value)
                if not text:
                    continue
                references.setdefault(text, []).append(f"{{{name}[{index}]}}")
    return {
        value: matches[0]
        for value, matches in references.items()
        if len(set(matches)) == 1
    }


def _parameter_agnostic_url(
    url: str,
    references_by_value: dict[str, str],
) -> dict[str, Any]:
    """Canonicalize exact parameter-valued URL segments without weakening routes."""

    parsed = urlparse(url)

    def canonicalize(component: str) -> str:
        decoded = unquote(component)
        return references_by_value.get(decoded, decoded)

    return {
        "scheme": parsed.scheme.casefold(),
        "netloc": parsed.netloc.casefold(),
        "path_segments": [canonicalize(segment) for segment in parsed.path.split("/")],
        "query": sorted(
            (
                canonicalize(key),
                canonicalize(value),
            )
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        ),
        "fragment": canonicalize(parsed.fragment),
    }
