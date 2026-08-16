from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from browser_use.agent.history_compiler import BrowserUseActionCache

from optexity.inference.core.automation_cache.converter import convert_action_cache
from optexity.inference.core.learning_memory.capabilities import (
    LocatorResolutionError,
    locator_action,
    prepare_action_node,
    verify_action_effect,
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
from optexity.inference.infra.browser_health import is_driver_closed_error
from optexity.schema.automation import ActionNode, Automation
from optexity.schema.memory import Memory
from optexity.schema.task import Task
from optexity.schema.token_usage import TokenUsage
from optexity.utils.settings import settings

logger = logging.getLogger(__name__)

LEARNING_SESSION_STATE_KEY = "learning_memory_session"
ACTION_CACHE_FILENAME = "browser_use_action_cache.json"


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
    run_kind: Literal["draft_replay", "active_replay"]
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
            task.automation.model_dump(mode="json")
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
            input_binding_fingerprint=_fingerprint(
                {
                    "input_parameters": self.task.input_parameters,
                    "secure_parameters": self.task.model_dump(
                        mode="json", include={"secure_parameters"}
                    ).get("secure_parameters", {}),
                    "generated_parameters": memory.variables.generated_variables,
                }
            ),
            starting_origin=origin,
            entry_url_fingerprint=_fingerprint(entry_url),
        )
        return workflow, compatibility

    async def replay_if_available(
        self,
        *,
        node_path: str,
        workflow: WorkflowIdentity,
        compatibility: SourceCompatibility,
        memory: Memory,
        browser: Browser,
        full_automation: list,
        execute_node: Callable[[ActionNode, int], Awaitable[None]],
    ) -> bool:
        try:
            version = self.store.select_replay_version(workflow, compatibility)
        except LearningMemoryStoreError:
            # A stale/corrupt local memory must never block the original agent.
            logger.exception(
                "Ignoring unreadable learning memory for %s %s",
                workflow.recording_id,
                node_path,
            )
            version = None
        if version is None:
            self.memory_misses.append((workflow, utc_now()))
            return False

        run_kind: Literal["draft_replay", "active_replay"] = (
            "draft_replay"
            if version.status == WorkflowVersionStatus.DRAFT
            else "active_replay"
        )
        started_at = utc_now()
        started_monotonic = time.monotonic()
        token_usage_before = memory.token_usage.model_copy(deep=True)
        events: list[LocatorValidationEvent] = []
        selected_indexes: dict[int, int] = {}
        selected_commands: dict[int, str] = {}
        executed_nodes = 0

        try:
            for node_index, cached_node in enumerate(version.automation.nodes):
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
                await execute_node(prepared.node, 1)
                executed_nodes += 1
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
            if not infrastructure_failure:
                self.store.record_failure(
                    workflow,
                    version.generation,
                    task_id=str(self.task.task_id),
                    reason=reason,
                )
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
                        else ReplayOutcome.ACTION_FAILED
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
            if isinstance(exc, LocatorResolutionError) and executed_nodes == 0:
                # No cached side effect has occurred, so the original Browser
                # Use node can safely repair the workflow in this same run.
                logger.info(
                    "Cached locator validation failed before execution; "
                    "falling back to fresh agentic discovery"
                )
                return False
            raise LearningReplayError(
                "Cached replay failed; the next run will use fresh agentic discovery"
            ) from exc

        pending = PendingReplay(
            workflow=workflow,
            version=version,
            run_kind=run_kind,
            started_at=started_at,
            started_monotonic=started_monotonic,
            replay_token_usage=memory.token_usage - token_usage_before,
            locator_events=events,
            selected_candidate_indexes=selected_indexes,
            selected_commands=selected_commands,
        )
        self.pending_replays.append(pending)
        return True

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
        try:
            signature = await capture_page_signature(browser)
        except Exception:
            if self.pending_replays:
                raise
            # Capturing first-run evidence is best-effort. The user-visible
            # Browser Use task already succeeded, so a missing page/body must
            # only skip draft creation, never fail the workflow.
            logger.exception(
                "Could not capture the learning-memory discovery signature"
            )
            self._append_memory_misses()
            self._finalized = True
            return

        for pending in self.pending_replays:
            if not pending.version.source_final_signature.matches(signature):
                reason = "Replay final page signature does not match discovery"
                self.store.record_failure(
                    pending.workflow,
                    pending.version.generation,
                    task_id=str(self.task.task_id),
                    reason=reason,
                    signature_mismatch=True,
                )
                self.failed_generations.add(
                    (pending.workflow.node_path, pending.version.generation)
                )
                self._append_observation(
                    _replay_observation(
                        pending,
                        task_id=str(self.task.task_id),
                        outcome=ReplayOutcome.SIGNATURE_MISMATCH,
                        signature_matches=False,
                        failure_reason=reason,
                    )
                )
                self._finalized = True
                raise LearningReplayError(reason)

            updated = _apply_successful_locator_choices(pending)
            zero_llm_replay = (
                pending.replay_token_usage.total_tokens == 0
                and pending.replay_token_usage.calculated_total_tokens == 0
            )
            store_failure: str | None = None
            try:
                self.store.record_success(
                    pending.workflow,
                    updated,
                    task_id=str(self.task.task_id),
                    promote=zero_llm_replay,
                )
            except LearningMemoryStoreError as exc:
                # The replay and final signature already passed. A concurrent
                # memory update must not change the user-visible task outcome.
                store_failure = f"Learning-memory update skipped: {exc}"
                logger.warning(store_failure)
            self._append_observation(
                _replay_observation(
                    pending,
                    task_id=str(self.task.task_id),
                    outcome=ReplayOutcome.PASSED,
                    signature_matches=True,
                    failure_reason=store_failure,
                )
            )

        for discovery in self.pending_discoveries:
            try:
                version = self._compile_discovery(discovery, signature)
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
        infrastructure_failure = is_driver_closed_error(error)
        for pending in self.pending_replays:
            key = (pending.workflow.node_path, pending.version.generation)
            if key in self.failed_generations:
                continue
            if not infrastructure_failure:
                self.store.record_failure(
                    pending.workflow,
                    pending.version.generation,
                    task_id=str(self.task.task_id),
                    reason=reason,
                )
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

    def _compile_discovery(
        self,
        discovery: PendingDiscovery,
        signature: PageSignature,
    ) -> WorkflowVersion:
        cache = BrowserUseActionCache.model_validate_json(
            discovery.cache_path.read_text(encoding="utf-8")
        )
        conversion = convert_action_cache(
            cache,
            allow_unvalidated_locators=True,
            allow_unresolved_select_options=(
                settings.LEARNING_MEMORY_ALLOW_UNRESOLVED_SELECT_OPTIONS
            ),
            allow_literal_password_inputs=(
                settings.LEARNING_MEMORY_ALLOW_LITERAL_PASSWORD_INPUTS
            ),
        )
        strict_automation = _strict_automation(conversion.automation)
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
                        strategy=LearnedStepStrategy.DIRECT,
                    )
                )
                continue
            if candidate is None:
                raise ValueError(
                    f"Locator node {node_index} has no cache candidate provenance"
                )
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
            raise ValueError("A strict learned locator action requires a command")
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
        failure_reason=failure_reason,
    )


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
