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
        document = self.load(workflow)
        if document is None:
            return None

        compatible_active = [
            version
            for version in document.versions
            if version.status == WorkflowVersionStatus.ACTIVE
            and version.compatibility == compatibility
        ]
        if compatible_active:
            return max(
                compatible_active, key=lambda version: version.generation
            ).model_copy(deep=True)

        compatible_drafts = [
            version
            for version in document.versions
            if version.status == WorkflowVersionStatus.DRAFT
            and version.compatibility == compatibility
        ]
        if compatible_drafts:
            return max(
                compatible_drafts, key=lambda version: version.generation
            ).model_copy(deep=True)

        return None

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
                WorkflowVersionStatus.SUPERSEDED,
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
    ) -> WorkflowVersion:
        directory = self._workflow_directory(workflow)
        with self._locked(directory):
            document = self._required_document(workflow)
            version = self._required_version(document, generation).model_copy(deep=True)
            now = utc_now()
            version.stats.failed_full_runs += 1
            version.stats.consecutive_failures += 1
            version.stats.last_task_id = task_id
            version.stats.last_run_at = now
            version.updated_at = now
            version.last_failure_reason = reason

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
