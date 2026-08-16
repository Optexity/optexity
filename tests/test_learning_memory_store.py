from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from optexity.inference.core.learning_memory.models import (
    LearnedStep,
    LearnedStepStrategy,
    PageSignature,
    SourceCompatibility,
    WorkflowIdentity,
    WorkflowVersion,
    WorkflowVersionStatus,
)
from optexity.inference.core.learning_memory.store import (
    LearningMemoryStoreError,
    LocalLearningMemoryStore,
)
from optexity.schema.actions.misc_action import SleepAction
from optexity.schema.automation import ActionNode, Automation, Parameters


def _workflow() -> WorkflowIdentity:
    return WorkflowIdentity(
        company_id="company",
        workspace_id="workspace",
        user_id="user",
        recording_id="recording",
        endpoint_name="endpoint",
        node_path="nodes[0]",
    )


def _version() -> WorkflowVersion:
    compatibility = SourceCompatibility(
        source_node_fingerprint="node",
        source_automation_fingerprint="automation",
        parameter_schema_fingerprint="parameters",
        starting_origin="https://example.test",
        entry_url_fingerprint="entry",
    )
    automation = Automation(
        url="https://example.test",
        parameters=Parameters(input_parameters={}, generated_parameters={}),
        nodes=[
            ActionNode(
                type="action_node",
                sleep_action=SleepAction(sleep_time=0.01),
                end_sleep_time=0,
            )
        ],
    )
    return WorkflowVersion(
        generation=1,
        status=WorkflowVersionStatus.DRAFT,
        source_task_id="source-task",
        compatibility=compatibility,
        cache_format_version="1.3",
        conversion_status="draft_requires_replay_validation",
        automation=automation,
        steps=[
            LearnedStep(
                node_index=0,
                source_step_number=1,
                browser_use_action="wait",
                optexity_action="sleep_action",
                strategy=LearnedStepStrategy.DIRECT,
            )
        ],
        source_final_signature=PageSignature(
            url="https://example.test",
            title="Example",
            body_text_sha256="0" * 64,
            body_character_count=7,
        ),
    )


class LearningMemoryStoreTests(unittest.TestCase):
    def test_draft_promotes_then_degrades_without_stale_resurrection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = LocalLearningMemoryStore(Path(temporary_directory))
            workflow = _workflow()
            draft = store.create_draft(workflow, _version())
            active = store.record_success(
                workflow,
                draft,
                task_id="replay-success",
                promote=True,
            )
            self.assertEqual(active.status, WorkflowVersionStatus.ACTIVE)

            degraded = store.record_failure(
                workflow,
                active.generation,
                task_id="replay-failure",
                reason="locator failed",
                expected_version=active,
            )
            self.assertEqual(degraded.status, WorkflowVersionStatus.DEGRADED)
            self.assertEqual(
                store.select_replay_versions(workflow, active.compatibility), []
            )

            with self.assertRaises(LearningMemoryStoreError):
                store.record_success(
                    workflow,
                    active,
                    task_id="stale-success",
                    promote=True,
                )


if __name__ == "__main__":
    unittest.main()
