from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from browser_use.agent.history_compiler import BrowserUseActionCache
from pydantic import BaseModel, ConfigDict

from optexity.inference.core.automation_cache.converter import (
    convert_action_cache,
    plan_action_cache_conversion,
)
from optexity.inference.core.automation_cache.llm_resolver import (
    resolve_unresolved_steps,
)
from optexity.inference.core.automation_cache.models import (
    AutomationConversionPlan,
    AutomationConversionResult,
    ConversionMode,
    StepResolution,
)
from optexity.inference.core.automation_cache.parameters import RuntimeParameterBinding
from optexity.inference.core.automation_cache.resolution_models import (
    LLMResolutionResult,
    LLMResolverConfig,
)
from optexity.inference.models.llm_model import LLMModel
from optexity.schema.automation import SecureParameter


class AutomaticConversionOutcome(BaseModel):
    """Complete audit record for deterministic and optional LLM conversion."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    initial_plan: AutomationConversionPlan
    final_plan: AutomationConversionPlan
    resolution: LLMResolutionResult
    conversion: AutomationConversionResult | None = None

    @property
    def complete(self) -> bool:
        return self.conversion is not None


def automatically_convert_action_cache(
    cache: BrowserUseActionCache,
    *,
    resolver_config: LLMResolverConfig | None = None,
    inherited_provider: str | None = None,
    inherited_model_name: str | None = None,
    model: LLMModel | None = None,
    source_input_parameters: Mapping[str, list[str | int | float | bool]] | None = None,
    runtime_parameter_bindings: Sequence[RuntimeParameterBinding] | None = None,
    source_secure_parameters: Mapping[str, list[SecureParameter]] | None = None,
    source_generated_parameters: (
        Mapping[str, list[str | int | float | bool | None]] | None
    ) = None,
    preserve_unmatched_literals: bool = False,
    allow_unvalidated_locators: bool = False,
    allow_unresolved_select_options: bool = False,
    allow_literal_password_inputs: bool = False,
    allow_literal_upload_paths: bool = False,
) -> AutomaticConversionOutcome:
    """Convert cache evidence without allowing an LLM to invent executable data.

    Deterministic adapters run first.  Only remaining eligible source steps are
    sent to the schema-constrained resolver.  Its proposals are rebuilt through
    trusted Optexity models before the complete Automation is validated.
    """

    policy = resolver_config or LLMResolverConfig()
    initial_plan = plan_action_cache_conversion(
        cache,
        source_input_parameters=source_input_parameters,
        runtime_parameter_bindings=runtime_parameter_bindings,
        source_secure_parameters=source_secure_parameters,
        source_generated_parameters=source_generated_parameters,
        preserve_unmatched_literals=preserve_unmatched_literals,
        allow_unvalidated_locators=allow_unvalidated_locators,
        allow_unresolved_select_options=allow_unresolved_select_options,
        allow_literal_password_inputs=allow_literal_password_inputs,
        allow_literal_upload_paths=allow_literal_upload_paths,
    )
    resolution = LLMResolutionResult()
    final_plan = initial_plan

    if not initial_plan.complete:
        resolution = resolve_unresolved_steps(
            cache,
            initial_plan,
            config=policy,
            inherited_provider=inherited_provider,
            inherited_model_name=inherited_model_name,
            model=model,
        )
        step_resolutions: dict[int, StepResolution] = {}
        for resolved_step in resolution.resolved_steps:
            mode = (
                ConversionMode.LLM_LOCATOR_ASSISTED
                if resolved_step.resolution_type == "locator_assisted"
                else ConversionMode.LLM_AGENTIC_FALLBACK
            )
            step_number = resolved_step.source_step_number
            step_resolutions[step_number] = StepResolution(
                source_step_number=step_number,
                node=resolved_step.node,
                optexity_action=resolved_step.optexity_action,
                explanation=resolved_step.explanation,
                conversion_mode=mode,
            )
        final_plan = plan_action_cache_conversion(
            cache,
            step_resolutions=step_resolutions,
            source_input_parameters=source_input_parameters,
            runtime_parameter_bindings=runtime_parameter_bindings,
            source_secure_parameters=source_secure_parameters,
            source_generated_parameters=source_generated_parameters,
            preserve_unmatched_literals=preserve_unmatched_literals,
            allow_unvalidated_locators=allow_unvalidated_locators,
            allow_unresolved_select_options=allow_unresolved_select_options,
            allow_literal_password_inputs=allow_literal_password_inputs,
            allow_literal_upload_paths=allow_literal_upload_paths,
        )

    conversion = None
    if final_plan.complete:
        conversion = convert_action_cache(
            cache,
            source_input_parameters=source_input_parameters,
            runtime_parameter_bindings=runtime_parameter_bindings,
            source_secure_parameters=source_secure_parameters,
            source_generated_parameters=source_generated_parameters,
            preserve_unmatched_literals=preserve_unmatched_literals,
            step_resolutions={
                resolved_step.source_step_number: StepResolution(
                    source_step_number=resolved_step.source_step_number,
                    node=resolved_step.node,
                    optexity_action=resolved_step.optexity_action,
                    explanation=resolved_step.explanation,
                    conversion_mode=(
                        ConversionMode.LLM_LOCATOR_ASSISTED
                        if resolved_step.resolution_type == "locator_assisted"
                        else ConversionMode.LLM_AGENTIC_FALLBACK
                    ),
                )
                for resolved_step in resolution.resolved_steps
            },
            allow_unvalidated_locators=allow_unvalidated_locators,
            allow_unresolved_select_options=allow_unresolved_select_options,
            allow_literal_password_inputs=allow_literal_password_inputs,
            allow_literal_upload_paths=allow_literal_upload_paths,
        )

    return AutomaticConversionOutcome(
        initial_plan=initial_plan,
        final_plan=final_plan,
        resolution=resolution,
        conversion=conversion,
    )


def write_automatic_conversion_artifacts(
    outcome: AutomaticConversionOutcome,
    directory: Path,
) -> None:
    """Atomically persist the audit trail and complete Automation, when present."""

    directory.mkdir(parents=True, exist_ok=True)
    _write_json(
        directory / "automation_conversion_plan.json",
        outcome.final_plan.model_dump(mode="json"),
    )
    _write_json(
        directory / "automation_conversion_report.json",
        outcome.model_dump(mode="json", exclude={"conversion": {"automation"}}),
    )
    if outcome.conversion is not None:
        _write_json(
            directory / "test_automation_cached.json",
            outcome.conversion.automation.model_dump(
                mode="json", exclude_none=True, exclude_defaults=True
            ),
        )


def _write_json(destination: Path, payload: object) -> None:
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
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(serialized)
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.chmod(0o600)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
