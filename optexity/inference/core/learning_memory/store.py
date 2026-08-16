from __future__ import annotations

import fcntl
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote

from pydantic import ValidationError

from optexity.inference.core.learning_memory.models import (
    LocatorCandidateState,
    LocatorValidationEvent,
    LocatorValidationOutcome,
    RunObservation,
    RunObservationFile,
    SourceCompatibility,
    WorkflowIdentity,
    WorkflowMemoryDocument,
    WorkflowVersion,
    WorkflowVersionStatus,
    utc_now,
)

MEMORY_FILENAME = "memory.json"
OBSERVATION_FILENAME = "learning_memory_observation.json"


class LearningMemoryStoreError(RuntimeError):
    """Raised when persisted learning memory is missing or corrupt."""


def _safe_component(value: str | None) -> str:
    return quote(value if value is not None else "none", safe="")


class LocalLearningMemoryStore:
    """Atomic, process-safe local store used by the assignment learning loop.

    Production deployments can implement the same model contract in the control
    plane. Task-local logs are deliberately not the durable store because worker
    cleanup removes them outside development mode.
    """

    def __init__(self, root: Path, *, max_versions: int = 5):
        self.root = Path(root)
        self.max_versions = max_versions

    def load(self, workflow: WorkflowIdentity) -> WorkflowMemoryDocument | None:
        memory_path = self._memory_path(workflow)
        if not memory_path.exists():
            return None
        try:
            document = WorkflowMemoryDocument.model_validate_json(
                memory_path.read_text(encoding="utf-8")
            )
            if document.workflow != workflow:
                raise LearningMemoryStoreError(
                    "Stored learning-memory identity does not match lookup identity"
                )
            return document
        except (OSError, ValidationError) as exc:
            raise LearningMemoryStoreError(
                f"Could not load learning memory for {workflow.recording_id!r}"
            ) from exc

    def select_replay_version(
        self,
        workflow: WorkflowIdentity,
        compatibility: SourceCompatibility,
    ) -> WorkflowVersion | None:
        versions = self.select_replay_versions(workflow, compatibility)
        return versions[0] if versions else None

    def select_replay_versions(
        self,
        workflow: WorkflowIdentity,
        compatibility: SourceCompatibility,
    ) -> list[WorkflowVersion]:
        """Return canary, active, then rollback versions for one real run."""

        document = self.load(workflow)
        if document is None:
            return []

        compatible = [
            version
            for version in document.versions
            if version.compatibility == compatibility
        ]
        ordered: list[WorkflowVersion] = []
        for status in (
            WorkflowVersionStatus.DRAFT,
            WorkflowVersionStatus.ACTIVE,
            WorkflowVersionStatus.SUPERSEDED,
        ):
            ordered.extend(
                sorted(
                    (version for version in compatible if version.status == status),
                    key=lambda version: version.generation,
                    reverse=True,
                )
            )
        return [version.model_copy(deep=True) for version in ordered]

    def create_draft(
        self,
        workflow: WorkflowIdentity,
        version: WorkflowVersion,
    ) -> WorkflowVersion:
        directory = self._workflow_directory(workflow)
        with self._locked(directory):
            document = self._load_unlocked(workflow) or WorkflowMemoryDocument(
                workflow=workflow
            )
            if document.workflow != workflow:
                raise LearningMemoryStoreError(
                    "Stored workflow identity does not match the requested workflow"
                )

            generation = document.next_generation
            created = version.model_copy(
                deep=True,
                update={
                    "generation": generation,
                    "status": WorkflowVersionStatus.DRAFT,
                    "created_at": utc_now(),
                    "updated_at": utc_now(),
                },
            )
            self._inherit_candidate_evidence(document, created)
            document.versions.append(created)
            document.next_generation = generation + 1
            self._prune_versions(document)
            self._persist_document(document)
            return created.model_copy(deep=True)

    def record_success(
        self,
        workflow: WorkflowIdentity,
        updated_version: WorkflowVersion,
        *,
        task_id: str,
        promote: bool,
    ) -> WorkflowVersion:
        directory = self._workflow_directory(workflow)
        with self._locked(directory):
            document = self._required_document(workflow)
            stored = self._required_version(document, updated_version.generation)
            if stored.compatibility != updated_version.compatibility:
                raise LearningMemoryStoreError(
                    "Replay compatibility changed while recording success"
                )
            if (
                stored.status != updated_version.status
                or stored.updated_at != updated_version.updated_at
            ):
                raise LearningMemoryStoreError(
                    "Learning-memory version changed during replay; refusing "
                    "to overwrite the newer outcome"
                )
            if stored.status in {
                WorkflowVersionStatus.REJECTED,
                WorkflowVersionStatus.QUARANTINED,
                WorkflowVersionStatus.DEGRADED,
            }:
                raise LearningMemoryStoreError(
                    f"Cannot record success for {stored.status.value} memory"
                )

            now = utc_now()
            updated = updated_version.model_copy(deep=True)
            # Candidate choices come from this replay, while aggregate stats
            # must always start from the latest copy protected by the lock.
            updated.stats = stored.stats.model_copy(deep=True)
            updated.stats.successful_full_runs += 1
            updated.stats.consecutive_failures = 0
            updated.stats.last_task_id = task_id
            updated.stats.last_run_at = now
            updated.updated_at = now
            updated.last_failure_reason = None

            if promote:
                for version in document.versions:
                    if (
                        version.status
                        in {
                            WorkflowVersionStatus.ACTIVE,
                            WorkflowVersionStatus.DRAFT,
                        }
                        and version.generation != updated.generation
                        and version.compatibility == updated.compatibility
                    ):
                        version.status = WorkflowVersionStatus.SUPERSEDED
                        version.updated_at = now
                updated.status = WorkflowVersionStatus.ACTIVE
                updated.promoted_at = updated.promoted_at or now
                document.active_generation = updated.generation

            self._replace_version(document, updated)
            self._persist_document(document)
            return updated.model_copy(deep=True)

    def record_failure(
        self,
        workflow: WorkflowIdentity,
        generation: int,
        *,
        task_id: str,
        reason: str,
        signature_mismatch: bool = False,
        locator_events: list[LocatorValidationEvent] | None = None,
        selected_candidate_indexes: dict[int, int] | None = None,
        failed_node_index: int | None = None,
        expected_version: WorkflowVersion | None = None,
    ) -> WorkflowVersion:
        directory = self._workflow_directory(workflow)
        with self._locked(directory):
            document = self._required_document(workflow)
            stored = self._required_version(document, generation)
            if expected_version is not None and (
                stored.status != expected_version.status
                or stored.updated_at != expected_version.updated_at
            ):
                raise LearningMemoryStoreError(
                    "Learning-memory version changed during replay; refusing "
                    "to overwrite the newer outcome"
                )
            version = stored.model_copy(deep=True)
            now = utc_now()
            version.stats.failed_full_runs += 1
            version.stats.consecutive_failures += 1
            version.stats.last_task_id = task_id
            version.stats.last_run_at = now
            version.updated_at = now
            version.last_failure_reason = reason
            self._apply_locator_failure_evidence(
                version,
                locator_events or [],
                selected_candidate_indexes or {},
                failed_node_index=failed_node_index,
                reason=reason,
            )

            if signature_mismatch:
                version.status = WorkflowVersionStatus.QUARANTINED
            elif version.status == WorkflowVersionStatus.DRAFT:
                version.status = WorkflowVersionStatus.REJECTED
            elif version.status in {
                WorkflowVersionStatus.ACTIVE,
                WorkflowVersionStatus.DEGRADED,
            }:
                version.status = WorkflowVersionStatus.DEGRADED

            if document.active_generation == generation:
                document.active_generation = None
            self._replace_version(document, version)
            self._persist_document(document)
            return version.model_copy(deep=True)

    @staticmethod
    def _inherit_candidate_evidence(
        document: WorkflowMemoryDocument,
        created: WorkflowVersion,
    ) -> None:
        """Carry objective locator outcomes into a newly discovered generation."""

        prior_versions = sorted(
            (
                version
                for version in document.versions
                if version.compatibility == created.compatibility
            ),
            key=lambda version: version.generation,
            reverse=True,
        )
        for step in created.steps:
            for candidate in step.candidates:
                inherited = next(
                    (
                        old_candidate
                        for old_version in prior_versions
                        for old_step in old_version.steps
                        if old_step.source_step_number == step.source_step_number
                        and old_step.optexity_action == step.optexity_action
                        for old_candidate in old_step.candidates
                        if old_candidate.command == candidate.command
                    ),
                    None,
                )
                if inherited is not None:
                    candidate.state = inherited.state
                    candidate.validation_successes = inherited.validation_successes
                    candidate.validation_failures = inherited.validation_failures
                    candidate.full_run_successes = inherited.full_run_successes
                    candidate.last_latency_ms = inherited.last_latency_ms
                    candidate.last_failure_reason = inherited.last_failure_reason
                    candidate.last_validated_at = inherited.last_validated_at

            if step.candidates:
                step.chosen_candidate_index = min(
                    range(len(step.candidates)),
                    key=lambda index: (
                        step.candidates[index].state == LocatorCandidateState.DEGRADED,
                        -step.candidates[index].full_run_successes,
                        step.candidates[index].validation_failures
                        - step.candidates[index].validation_successes,
                        step.candidates[index].original_rank,
                    ),
                )

    @staticmethod
    def _apply_locator_failure_evidence(
        version: WorkflowVersion,
        events: list[LocatorValidationEvent],
        selected_candidate_indexes: dict[int, int],
        *,
        failed_node_index: int | None,
        reason: str,
    ) -> None:
        now = utc_now()
        uncertain = {
            LocatorValidationOutcome.PAGE_NOT_READY,
            LocatorValidationOutcome.TIMED_OUT,
        }
        for event in events:
            if event.node_index >= len(version.steps):
                raise LearningMemoryStoreError(
                    "Locator event references an unknown step"
                )
            step = version.steps[event.node_index]
            if event.candidate_index >= len(step.candidates):
                raise LearningMemoryStoreError(
                    "Locator event references an unknown candidate"
                )
            candidate = step.candidates[event.candidate_index]
            if candidate.command != event.command:
                raise LearningMemoryStoreError(
                    "Locator event command does not match persisted evidence"
                )
            candidate.last_latency_ms = event.elapsed_ms
            candidate.last_validated_at = now
            if event.outcome == LocatorValidationOutcome.PASSED:
                candidate.validation_successes += 1
            elif event.outcome not in uncertain:
                candidate.validation_failures += 1
                candidate.last_failure_reason = event.explanation or event.outcome.value
                if candidate.state == LocatorCandidateState.ACTIVE:
                    candidate.state = LocatorCandidateState.DEGRADED

        if failed_node_index is None:
            return
        candidate_index = selected_candidate_indexes.get(failed_node_index)
        if candidate_index is None:
            return
        step = version.steps[failed_node_index]
        candidate = step.candidates[candidate_index]
        candidate.validation_failures += 1
        candidate.last_failure_reason = reason
        candidate.state = LocatorCandidateState.DEGRADED

    def append_observation(
        self,
        logs_directory: Path,
        observation: RunObservation,
    ) -> Path:
        logs_directory.mkdir(parents=True, exist_ok=True)
        path = logs_directory / OBSERVATION_FILENAME
        existing = RunObservationFile()
        if path.exists():
            try:
                existing = RunObservationFile.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
            except (OSError, ValidationError) as exc:
                raise LearningMemoryStoreError(
                    f"Could not append learning observation to {path}"
                ) from exc
        existing.observations.append(observation)
        self._write_json_atomically(existing.model_dump(mode="json"), path)
        return path

    def _workflow_directory(self, workflow: WorkflowIdentity) -> Path:
        return (
            self.root
            / f"company={_safe_component(workflow.company_id)}"
            / f"workspace={_safe_component(workflow.workspace_id)}"
            / f"user={_safe_component(workflow.user_id)}"
            / f"recording={_safe_component(workflow.recording_id)}"
            / f"endpoint={_safe_component(workflow.endpoint_name)}"
            / f"source-version={_safe_component(workflow.source_automation_version)}"
            / f"node={_safe_component(workflow.node_path)}"
        )

    def _memory_path(self, workflow: WorkflowIdentity) -> Path:
        return self._workflow_directory(workflow) / MEMORY_FILENAME

    @contextmanager
    def _locked(self, directory: Path) -> Iterator[None]:
        directory.mkdir(parents=True, exist_ok=True)
        os.chmod(directory, 0o700)
        lock_path = directory / ".lock"
        with lock_path.open("a+", encoding="utf-8") as lock_file:
            os.chmod(lock_path, 0o600)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _load_unlocked(
        self, workflow: WorkflowIdentity
    ) -> WorkflowMemoryDocument | None:
        path = self._memory_path(workflow)
        if not path.exists():
            return None
        try:
            document = WorkflowMemoryDocument.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            if document.workflow != workflow:
                raise LearningMemoryStoreError(
                    "Stored learning-memory identity does not match lookup identity"
                )
            return document
        except (OSError, ValidationError) as exc:
            raise LearningMemoryStoreError(
                f"Invalid learning memory at {path}"
            ) from exc

    def _required_document(self, workflow: WorkflowIdentity) -> WorkflowMemoryDocument:
        document = self._load_unlocked(workflow)
        if document is None:
            raise LearningMemoryStoreError("Learning-memory document does not exist")
        return document

    @staticmethod
    def _required_version(
        document: WorkflowMemoryDocument, generation: int
    ) -> WorkflowVersion:
        version = next(
            (
                candidate
                for candidate in document.versions
                if candidate.generation == generation
            ),
            None,
        )
        if version is None:
            raise LearningMemoryStoreError(
                f"Learning-memory generation {generation} does not exist"
            )
        return version

    @staticmethod
    def _replace_version(
        document: WorkflowMemoryDocument, updated: WorkflowVersion
    ) -> None:
        for index, version in enumerate(document.versions):
            if version.generation == updated.generation:
                document.versions[index] = updated
                return
        raise LearningMemoryStoreError(
            f"Learning-memory generation {updated.generation} does not exist"
        )

    def _prune_versions(self, document: WorkflowMemoryDocument) -> None:
        while len(document.versions) > self.max_versions:
            removable = sorted(
                (
                    version
                    for version in document.versions
                    if version.status != WorkflowVersionStatus.ACTIVE
                ),
                key=lambda version: version.generation,
            )
            if not removable:
                break
            document.versions.remove(removable[0])

    def _persist_document(self, document: WorkflowMemoryDocument) -> None:
        document.revision += 1
        document.updated_at = utc_now()
        validated = WorkflowMemoryDocument.model_validate(
            document.model_dump(mode="json")
        )
        self._write_json_atomically(
            validated.model_dump(mode="json"), self._memory_path(document.workflow)
        )

    @staticmethod
    def _write_json_atomically(payload: object, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        serialized = (
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                os.chmod(temporary_path, 0o600)
                temporary_file.write(serialized)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, destination)
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except Exception:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise
