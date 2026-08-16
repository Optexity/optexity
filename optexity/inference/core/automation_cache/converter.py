from __future__ import annotations

import json
import os
import re
import tempfile
import urllib.parse
from collections.abc import Mapping
from pathlib import Path

from browser_use.agent.history_compiler import (
    BrowserActionExecutionStatus,
    BrowserUseActionCache,
    CachedAutomationDecisionStatus,
    DeterministicStepCandidate,
    DirectActionCandidate,
    DirectFindTextAction,
    DirectGoBackAction,
    DirectNavigateAction,
    DirectScrollAction,
    DirectSearchAction,
    DirectSendKeysAction,
    DirectSleepAction,
    ObservedStep,
    PlaywrightAction,
    PlaywrightClickAction,
    PlaywrightScrollAction,
    PlaywrightSelectAction,
    PlaywrightUploadAction,
    build_locator_options,
)
from pydantic import ValidationError

from optexity.inference.core.automation_cache.action_adapters import (
    ActionAdapterContext,
    DirectActionAdapterContext,
    build_action_node,
    build_direct_action_node,
)
from optexity.inference.core.automation_cache.locator_commands import (
    validate_playwright_locator_command,
)
from optexity.inference.core.automation_cache.models import (
    ActionCacheConversionError,
    AutomationConversionPlan,
    AutomationConversionPlanStatus,
    AutomationConversionResult,
    AutomationConversionStatus,
    ConversionMode,
    ConvertedStep,
    ExcludedStep,
    LocatorSource,
    PlannedStep,
    PlannedStepStatus,
    StepResolution,
    UnconvertedStep,
)
from optexity.schema.automation import Automation, Parameters

SUPPORTED_CACHE_FORMAT_VERSIONS = frozenset({"1.1", "1.2", "1.3"})
DEFAULT_ACTION_MAX_TRIES = 10
DEFAULT_ACTION_TIMEOUT_SECONDS = 1.0
DEFAULT_PAGE_TRANSITION_WAIT_SECONDS = 0.5
_PARAMETER_REFERENCE_PATTERN = re.compile(
    r"\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)\[(?P<index>[0-9]+)\]\}"
)


def plan_action_cache_conversion(
    cache: BrowserUseActionCache,
    *,
    step_resolutions: Mapping[int, StepResolution] | None = None,
    source_input_parameters: Mapping[str, list[str | int | float | bool]] | None = None,
    allow_unvalidated_locators: bool = False,
    allow_unresolved_select_options: bool = False,
    allow_literal_password_inputs: bool = False,
    allow_literal_upload_paths: bool = False,
    action_max_tries: int = DEFAULT_ACTION_MAX_TRIES,
    action_timeout_seconds: float = DEFAULT_ACTION_TIMEOUT_SECONDS,
    page_transition_wait_seconds: float = DEFAULT_PAGE_TRANSITION_WAIT_SECONDS,
) -> AutomationConversionPlan:
    """Plan every cached action without discarding supported partial work.

    The cache is the sole workflow source. Dashboard automations are not used as
    templates. Locator candidates remain untested until replay, while cache format
    1.2 direct candidates can be translated without a locator. Unsupported actions
    remain ordered, explicit unresolved steps; this function never emits a partial
    runnable Automation.
    """

    _validate_cache_header(cache)
    _validate_conversion_policy(
        action_max_tries=action_max_tries,
        action_timeout_seconds=action_timeout_seconds,
        page_transition_wait_seconds=page_transition_wait_seconds,
    )
    candidates = {
        candidate.candidate_number: candidate
        for candidate in cache.deterministic_step_candidates
    }
    planned_steps: list[PlannedStep] = []
    unresolved_select_steps: list[int] = []
    literal_password_steps: list[int] = []
    global_problems: list[UnconvertedStep] = []
    used_candidate_numbers: set[int] = set()
    resolutions = dict(step_resolutions or {})
    used_resolution_steps: set[int] = set()

    for position, observed_step in enumerate(cache.all_observed_steps):
        decision = observed_step.cached_automation_decision
        if decision.decision in {
            CachedAutomationDecisionStatus.EXCLUDE_TERMINAL_ACTION,
            CachedAutomationDecisionStatus.EXCLUDE_OBSERVATION_ACTION,
            CachedAutomationDecisionStatus.EXCLUDE_FAILED_ACTION,
            CachedAutomationDecisionStatus.EXCLUDE_NOT_EXECUTED_ACTION,
        }:
            excluded = ExcludedStep(
                source_step_number=observed_step.step_number,
                browser_use_action=observed_step.browser_action.original_action_name,
                cache_decision=decision.decision.value,
                explanation=decision.explanation,
            )
            planned_steps.append(
                PlannedStep(
                    source_step_number=observed_step.step_number,
                    browser_use_action=observed_step.browser_action.original_action_name,
                    status=PlannedStepStatus.EXCLUDED,
                    excluded_step=excluded,
                )
            )
            continue
        if (
            decision.decision
            == CachedAutomationDecisionStatus.READY_FOR_REPLAY_VALIDATION
        ):
            candidate_number = decision.deterministic_candidate_number
            candidate = candidates.get(candidate_number) if candidate_number else None
            if (
                not isinstance(candidate, DirectActionCandidate)
                or candidate.source_step_number != observed_step.step_number
            ):
                planned_steps.append(
                    _planned_unresolved_step(
                        observed_step,
                        explanation=(
                            "A replay-ready source step must reference a matching "
                            "direct action candidate."
                        ),
                    )
                )
                continue
            used_candidate_numbers.add(candidate.candidate_number)
            if (
                observed_step.browser_action_result.status
                != BrowserActionExecutionStatus.EXECUTED_WITHOUT_REPORTED_ERROR
            ):
                planned_steps.append(
                    _planned_unresolved_step(
                        observed_step,
                        explanation=(
                            "A direct candidate must come from an action that "
                            "executed without a reported error."
                        ),
                    )
                )
                continue
            if observed_step.browser_action.unhandled_action_arguments:
                planned_steps.append(
                    _planned_unresolved_step(
                        observed_step,
                        explanation=(
                            "The source action contains arguments that its direct "
                            "adapter did not handle."
                        ),
                    )
                )
                continue
            provenance_problem = _direct_candidate_provenance_problem(
                observed_step, candidate
            )
            if provenance_problem is not None:
                planned_steps.append(
                    _planned_unresolved_step(
                        observed_step,
                        problem=provenance_problem,
                    )
                )
                continue
            recorded_transition = _recorded_page_transition_after_step(
                observed_step,
                _next_step(cache.all_observed_steps, position),
            )
            try:
                adapted = build_direct_action_node(
                    DirectActionAdapterContext(
                        observed_step=observed_step,
                        candidate=candidate,
                        end_sleep_time=(
                            page_transition_wait_seconds if recorded_transition else 0.0
                        ),
                    )
                )
            except ActionCacheConversionError as exc:
                planned_steps.append(
                    _planned_unresolved_step(
                        observed_step,
                        problem=_problem_from_conversion_error(exc, observed_step),
                    )
                )
                continue
            converted = ConvertedStep(
                source_step_number=observed_step.step_number,
                deterministic_candidate_number=candidate.candidate_number,
                browser_use_action=observed_step.browser_action.original_action_name,
                optexity_action=adapted.optexity_action_name,
                conversion_mode=ConversionMode.NATIVE_DETERMINISTIC,
                prompt_fallback_enabled=False,
                recorded_page_transition_after_step=recorded_transition,
            )
            planned_steps.append(
                PlannedStep(
                    source_step_number=observed_step.step_number,
                    browser_use_action=(
                        observed_step.browser_action.original_action_name
                    ),
                    status=PlannedStepStatus.CONVERTED,
                    converted_step=converted,
                    node=adapted.node,
                )
            )
            continue
        if (
            decision.decision
            != CachedAutomationDecisionStatus.WAITING_FOR_LOCATOR_VALIDATION
        ):
            resolution = resolutions.get(observed_step.step_number)
            if (
                resolution is not None
                and decision.decision
                in {
                    CachedAutomationDecisionStatus.REQUIRES_AGENTIC_HANDLING,
                    CachedAutomationDecisionStatus.UNSUPPORTED_ACTION,
                }
                and observed_step.browser_action_result.status
                == BrowserActionExecutionStatus.EXECUTED_WITHOUT_REPORTED_ERROR
            ):
                used_resolution_steps.add(observed_step.step_number)
                converted = ConvertedStep(
                    source_step_number=observed_step.step_number,
                    browser_use_action=(
                        observed_step.browser_action.original_action_name
                    ),
                    optexity_action=resolution.optexity_action,
                    conversion_mode=resolution.conversion_mode,
                    prompt_fallback_enabled=False,
                    recorded_page_transition_after_step=(
                        _recorded_page_transition_after_step(
                            observed_step,
                            _next_step(cache.all_observed_steps, position),
                        )
                    ),
                )
                planned_steps.append(
                    PlannedStep(
                        source_step_number=observed_step.step_number,
                        browser_use_action=(
                            observed_step.browser_action.original_action_name
                        ),
                        status=PlannedStepStatus.CONVERTED,
                        converted_step=converted,
                        node=resolution.node,
                    )
                )
                continue
            planned_steps.append(_planned_unresolved_step(observed_step))
            continue

        candidate_number = decision.deterministic_candidate_number
        candidate = candidates.get(candidate_number) if candidate_number else None
        if (
            not isinstance(candidate, DeterministicStepCandidate)
            or candidate.source_step_number != observed_step.step_number
        ):
            planned_steps.append(
                _planned_unresolved_step(
                    observed_step,
                    explanation="The source step does not reference a matching deterministic candidate.",
                )
            )
            continue
        used_candidate_numbers.add(candidate.candidate_number)
        if (
            observed_step.browser_action_result.status
            != BrowserActionExecutionStatus.EXECUTED_WITHOUT_REPORTED_ERROR
        ):
            planned_steps.append(
                _planned_unresolved_step(
                    observed_step,
                    explanation=(
                        "A deterministic candidate must come from an action that "
                        "executed without a reported error."
                    ),
                )
            )
            continue
        if observed_step.browser_action.unhandled_action_arguments:
            planned_steps.append(
                _planned_unresolved_step(
                    observed_step,
                    explanation=(
                        "The source action contains arguments that its deterministic "
                        "adapter did not handle."
                    ),
                )
            )
            continue
        expected_browser_use_action = {
            "fill": "input",
            "type": "input",
            "click": "click",
            "select_option": "select_dropdown",
            "upload_file": "upload_file",
            "scroll_element": "scroll",
        }.get(candidate.playwright_action.action_type)
        if (
            expected_browser_use_action
            != observed_step.browser_action.original_action_name
        ):
            planned_steps.append(
                _planned_unresolved_step(
                    observed_step,
                    explanation=(
                        "The cached Playwright action does not match the recorded "
                        "Browser Use action."
                    ),
                )
            )
            continue
        provenance_problem = _candidate_provenance_problem(observed_step, candidate)
        if provenance_problem is not None:
            planned_steps.append(
                _planned_unresolved_step(
                    observed_step,
                    problem=provenance_problem,
                )
            )
            continue
        if (
            isinstance(candidate.playwright_action, PlaywrightAction)
            and observed_step.element_used is not None
            and observed_step.element_used.locator_relevant_attributes.get("type")
            == "password"
            and not allow_literal_password_inputs
        ):
            planned_steps.append(
                _planned_unresolved_step(
                    observed_step,
                    explanation=(
                        "A password-field value requires explicit secure-parameter "
                        "mapping. Pass allow_literal_password_inputs=True only for "
                        "a non-sensitive test credential."
                    ),
                )
            )
            continue
        if (
            isinstance(candidate.playwright_action, PlaywrightUploadAction)
            and _PARAMETER_REFERENCE_PATTERN.fullmatch(
                candidate.playwright_action.file_path
            )
            is None
            and not allow_literal_upload_paths
        ):
            planned_steps.append(
                _planned_unresolved_step(
                    observed_step,
                    explanation=(
                        "A reusable upload requires an explicit Optexity runtime "
                        "parameter or managed-file binding. Pass "
                        "allow_literal_upload_paths=True only for a trusted local "
                        "replay whose file path is stable."
                    ),
                )
            )
            continue
        try:
            locator, locator_source = _select_locator(
                candidate,
                observed_step,
                allow_unvalidated_locators=allow_unvalidated_locators,
            )
            validate_playwright_locator_command(
                locator,
                source_step_number=observed_step.step_number,
                browser_use_action=observed_step.browser_action.original_action_name,
            )
        except ActionCacheConversionError as exc:
            planned_steps.append(
                _planned_unresolved_step(
                    observed_step,
                    problem=_problem_from_conversion_error(exc, observed_step),
                )
            )
            continue

        if (
            isinstance(candidate.playwright_action, PlaywrightSelectAction)
            and candidate.playwright_action.option_match == "unresolved"
            and not allow_unresolved_select_options
        ):
            planned_steps.append(
                _planned_unresolved_step(
                    observed_step,
                    explanation=(
                        "The cache does not yet know whether this select option "
                        "matches by label or value. Pass "
                        "allow_unresolved_select_options=True only for a draft."
                    ),
                )
            )
            continue
        recorded_transition = _recorded_page_transition_after_step(
            observed_step,
            _next_step(cache.all_observed_steps, position),
        )
        try:
            adapted = build_action_node(
                ActionAdapterContext(
                    observed_step=observed_step,
                    candidate=candidate,
                    playwright_command=locator,
                    # Cache format 1.1 cannot yet prove live locator validation,
                    # so use a prompt fallback when target evidence can describe
                    # one safely. XPath-only candidates remain strict commands.
                    enable_prompt_fallback=True,
                    end_sleep_time=(
                        page_transition_wait_seconds if recorded_transition else 0.0
                    ),
                    max_tries=action_max_tries,
                    max_timeout_seconds_per_try=action_timeout_seconds,
                )
            )
        except ActionCacheConversionError as exc:
            planned_steps.append(
                _planned_unresolved_step(
                    observed_step,
                    problem=_problem_from_conversion_error(exc, observed_step),
                )
            )
            continue

        converted = ConvertedStep(
            source_step_number=observed_step.step_number,
            deterministic_candidate_number=candidate.candidate_number,
            browser_use_action=observed_step.browser_action.original_action_name,
            optexity_action=adapted.optexity_action_name,
            conversion_mode=ConversionMode.CACHED_LOCATOR,
            playwright_command=locator,
            locator_source=locator_source,
            prompt_fallback_enabled=adapted.prompt_fallback_enabled,
            recorded_page_transition_after_step=recorded_transition,
        )
        planned_steps.append(
            PlannedStep(
                source_step_number=observed_step.step_number,
                browser_use_action=observed_step.browser_action.original_action_name,
                status=PlannedStepStatus.CONVERTED,
                converted_step=converted,
                node=adapted.node,
            )
        )
        if (
            isinstance(candidate.playwright_action, PlaywrightSelectAction)
            and candidate.playwright_action.option_match == "unresolved"
        ):
            unresolved_select_steps.append(observed_step.step_number)
        if (
            isinstance(candidate.playwright_action, PlaywrightAction)
            and observed_step.element_used is not None
            and observed_step.element_used.locator_relevant_attributes.get("type")
            == "password"
        ):
            literal_password_steps.append(observed_step.step_number)

    unused_candidates = sorted(set(candidates) - used_candidate_numbers)
    if unused_candidates:
        global_problems.append(
            UnconvertedStep(
                explanation=(
                    "The cache contains deterministic candidates that are not "
                    f"referenced by an observed step: {unused_candidates}."
                )
            )
        )
    unused_resolutions = sorted(set(resolutions) - used_resolution_steps)
    if unused_resolutions:
        global_problems.append(
            UnconvertedStep(
                explanation=(
                    "Step resolutions do not match unresolved source steps: "
                    f"{unused_resolutions}."
                )
            )
        )
    terminal_steps = [
        step
        for step in planned_steps
        if step.excluded_step is not None
        and step.excluded_step.cache_decision
        == CachedAutomationDecisionStatus.EXCLUDE_TERMINAL_ACTION.value
    ]
    if (
        len(terminal_steps) != 1
        or not cache.all_observed_steps
        or planned_steps[-1].status != PlannedStepStatus.EXCLUDED
        or cache.all_observed_steps[-1].browser_action.original_action_name != "done"
        or cache.all_observed_steps[-1].browser_action_result.status
        != BrowserActionExecutionStatus.TASK_COMPLETED_SUCCESSFULLY
    ):
        global_problems.append(
            UnconvertedStep(
                explanation=(
                    "A convertible source run must end with exactly one successful "
                    "terminal done action."
                )
            )
        )
    if not any(step.status == PlannedStepStatus.CONVERTED for step in planned_steps):
        global_problems.append(
            UnconvertedStep(
                explanation=(
                    "The action cache contains no executable deterministic candidates."
                )
            )
        )

    planned_steps, input_parameters = _parameterize_planned_steps(
        planned_steps,
        source_input_parameters=source_input_parameters,
    )

    if global_problems or any(
        step.status == PlannedStepStatus.UNRESOLVED for step in planned_steps
    ):
        status = AutomationConversionPlanStatus.PARTIAL_REQUIRES_RESOLUTION
    elif any(
        step.converted_step is not None
        and step.converted_step.conversion_mode
        in {
            ConversionMode.LLM_LOCATOR_ASSISTED,
            ConversionMode.LLM_AGENTIC_FALLBACK,
        }
        for step in planned_steps
    ):
        status = AutomationConversionPlanStatus.COMPLETE_HYBRID_DRAFT
    else:
        status = AutomationConversionPlanStatus.COMPLETE_DRAFT
    return AutomationConversionPlan(
        status=status,
        starting_url=cache.original_run.starting_url or "",
        input_parameters=input_parameters,
        ordered_steps=planned_steps,
        global_problems=global_problems,
        uses_unvalidated_locators=any(
            step.converted_step is not None
            and step.converted_step.conversion_mode == ConversionMode.CACHED_LOCATOR
            for step in planned_steps
        ),
        unresolved_select_option_steps=unresolved_select_steps,
        literal_password_input_steps=literal_password_steps,
    )


def convert_action_cache(
    cache: BrowserUseActionCache,
    *,
    step_resolutions: Mapping[int, StepResolution] | None = None,
    source_input_parameters: Mapping[str, list[str | int | float | bool]] | None = None,
    allow_unvalidated_locators: bool = False,
    allow_unresolved_select_options: bool = False,
    allow_literal_password_inputs: bool = False,
    allow_literal_upload_paths: bool = False,
    action_max_tries: int = DEFAULT_ACTION_MAX_TRIES,
    action_timeout_seconds: float = DEFAULT_ACTION_TIMEOUT_SECONDS,
    page_transition_wait_seconds: float = DEFAULT_PAGE_TRANSITION_WAIT_SECONDS,
) -> AutomationConversionResult:
    """Materialize an Automation only when every required step has a safe node."""

    plan = plan_action_cache_conversion(
        cache,
        step_resolutions=step_resolutions,
        source_input_parameters=source_input_parameters,
        allow_unvalidated_locators=allow_unvalidated_locators,
        allow_unresolved_select_options=allow_unresolved_select_options,
        allow_literal_password_inputs=allow_literal_password_inputs,
        allow_literal_upload_paths=allow_literal_upload_paths,
        action_max_tries=action_max_tries,
        action_timeout_seconds=action_timeout_seconds,
        page_transition_wait_seconds=page_transition_wait_seconds,
    )
    if not plan.complete:
        raise ActionCacheConversionError(
            "The action cache cannot produce a complete Automation",
            plan.problems,
            plan=plan,
        )

    automation = Automation.model_validate(
        {
            "url": plan.starting_url,
            "parameters": Parameters(
                input_parameters=plan.input_parameters,
                generated_parameters={},
            ).model_dump(mode="json"),
            "nodes": [node.model_dump(mode="json") for node in plan.nodes],
            "max_retries": 0,
        }
    )
    return AutomationConversionResult(
        status=(
            AutomationConversionStatus.HYBRID_DRAFT_REQUIRES_REPLAY_VALIDATION
            if plan.status == AutomationConversionPlanStatus.COMPLETE_HYBRID_DRAFT
            else AutomationConversionStatus.DRAFT_REQUIRES_REPLAY_VALIDATION
        ),
        automation=automation,
        converted_steps=plan.converted_steps,
        excluded_steps=plan.excluded_steps,
        uses_unvalidated_locators=plan.uses_unvalidated_locators,
        unresolved_select_option_steps=plan.unresolved_select_option_steps,
        literal_password_input_steps=plan.literal_password_input_steps,
    )


def convert_action_cache_file(
    cache_path: str | Path,
    automation_path: str | Path,
    *,
    source_input_parameters: Mapping[str, list[str | int | float | bool]] | None = None,
    allow_unvalidated_locators: bool = False,
    allow_unresolved_select_options: bool = False,
    allow_literal_password_inputs: bool = False,
    allow_literal_upload_paths: bool = False,
    action_max_tries: int = DEFAULT_ACTION_MAX_TRIES,
    action_timeout_seconds: float = DEFAULT_ACTION_TIMEOUT_SECONDS,
    page_transition_wait_seconds: float = DEFAULT_PAGE_TRANSITION_WAIT_SECONDS,
    overwrite: bool = False,
) -> AutomationConversionResult:
    """Load a cache, convert it, and atomically write a compact Automation JSON."""

    source_path = Path(cache_path)
    destination_path = Path(automation_path)
    if source_path.resolve() == destination_path.resolve():
        raise ActionCacheConversionError(
            "The cache and Automation paths must be different files"
        )
    if destination_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {destination_path}")

    try:
        cache = BrowserUseActionCache.model_validate_json(
            source_path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as exc:
        raise ActionCacheConversionError(
            f"Could not load action cache {source_path.name!r}"
        ) from exc

    result = convert_action_cache(
        cache,
        source_input_parameters=source_input_parameters,
        allow_unvalidated_locators=allow_unvalidated_locators,
        allow_unresolved_select_options=allow_unresolved_select_options,
        allow_literal_password_inputs=allow_literal_password_inputs,
        allow_literal_upload_paths=allow_literal_upload_paths,
        action_max_tries=action_max_tries,
        action_timeout_seconds=action_timeout_seconds,
        page_transition_wait_seconds=page_transition_wait_seconds,
    )
    payload = _compact_automation_payload(result.automation)
    serialized = (
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    Automation.model_validate_json(serialized)
    _write_atomically(serialized, destination_path, overwrite=overwrite)
    return result


class _InputParameterAllocator:
    """Create stable parameter references without persisting recorded values."""

    def __init__(
        self,
        source_input_parameters: Mapping[str, list[str | int | float | bool]] | None,
    ) -> None:
        self._source = source_input_parameters or {}
        self.parameters: dict[str, list[str | int | float | bool]] = {}

    def bind(self, value: str, *, source_step_number: int, suffix: str) -> str:
        existing = _PARAMETER_REFERENCE_PATTERN.fullmatch(value)
        if existing is not None:
            name = existing.group("name")
            self.parameters.setdefault(name, [])
            return value

        matches = [
            (name, index)
            for name, values in self._source.items()
            for index, candidate in enumerate(values)
            if str(candidate) == value
        ]
        if len(matches) == 1:
            name, index = matches[0]
            self.parameters.setdefault(name, [])
            return f"{{{name}[{index}]}}"

        base_name = f"step_{source_step_number}_{suffix}"
        name = base_name
        collision_index = 2
        while name in self.parameters or name in self._source:
            name = f"{base_name}_{collision_index}"
            collision_index += 1
        self.parameters[name] = []
        return f"{{{name}[0]}}"


def _parameterize_planned_steps(
    planned_steps: list[PlannedStep],
    *,
    source_input_parameters: Mapping[str, list[str | int | float | bool]] | None,
) -> tuple[list[PlannedStep], dict[str, list[str | int | float | bool]]]:
    """Replace value-bearing action data with typed Optexity parameters."""

    allocator = _InputParameterAllocator(source_input_parameters)
    parameterized_steps: list[PlannedStep] = []
    for planned_step in planned_steps:
        if planned_step.node is None or planned_step.converted_step is None:
            parameterized_steps.append(planned_step)
            continue

        node = planned_step.node.model_copy(deep=True)
        interaction = node.interaction_action
        references: list[str] = []
        if interaction is not None and interaction.input_text is not None:
            value = interaction.input_text.input_text
            if value is not None:
                reference = allocator.bind(
                    value,
                    source_step_number=planned_step.source_step_number,
                    suffix="input_text",
                )
                interaction.input_text.input_text = reference
                references.append(reference)
        elif interaction is not None and interaction.select_option is not None:
            values = interaction.select_option.select_values or []
            parameterized_values: list[str] = []
            for value_index, value in enumerate(values):
                reference = allocator.bind(
                    value,
                    source_step_number=planned_step.source_step_number,
                    suffix=(
                        "select_option"
                        if len(values) == 1
                        else f"select_option_{value_index + 1}"
                    ),
                )
                parameterized_values.append(reference)
                references.append(reference)
                prompt = interaction.select_option.prompt_instructions
                if prompt:
                    interaction.select_option.prompt_instructions = prompt.replace(
                        json.dumps(value, ensure_ascii=False),
                        json.dumps(reference, ensure_ascii=False),
                    )
            interaction.select_option.select_values = parameterized_values
        elif interaction is not None and interaction.upload_file is not None:
            value = interaction.upload_file.file_path
            if value is not None:
                reference = allocator.bind(
                    value,
                    source_step_number=planned_step.source_step_number,
                    suffix="upload_file",
                )
                interaction.upload_file.file_path = reference
                references.append(reference)
        elif interaction is not None and interaction.search is not None:
            reference = allocator.bind(
                interaction.search.query,
                source_step_number=planned_step.source_step_number,
                suffix="search_query",
            )
            interaction.search.query = reference
            references.append(reference)

        parameterized_steps.append(
            planned_step.model_copy(
                update={
                    "node": node,
                    "converted_step": planned_step.converted_step.model_copy(
                        update={"parameter_references": references}
                    ),
                }
            )
        )

    return parameterized_steps, allocator.parameters


def _validate_cache_header(cache: BrowserUseActionCache) -> None:
    problems: list[UnconvertedStep] = []
    if cache.cache_format_version not in SUPPORTED_CACHE_FORMAT_VERSIONS:
        problems.append(
            UnconvertedStep(
                explanation=(
                    f"Cache format {cache.cache_format_version!r} is not supported; "
                    "recompile raw_history.json with a supported cache compiler."
                )
            )
        )
    if not cache.original_run.starting_url:
        problems.append(UnconvertedStep(explanation="The cache has no starting URL."))
    if (
        not cache.original_run.task_completed
        or cache.original_run.task_succeeded is not True
    ):
        problems.append(
            UnconvertedStep(
                explanation="Only a Browser Use run that completed successfully can be converted."
            )
        )
    if cache.issues:
        problems.append(
            UnconvertedStep(
                explanation=(
                    f"The cache contains {len(cache.issues)} compilation issue(s) "
                    "that require review."
                )
            )
        )
    if problems:
        raise ActionCacheConversionError("Invalid action cache", problems)


def _validate_conversion_policy(
    *,
    action_max_tries: int,
    action_timeout_seconds: float,
    page_transition_wait_seconds: float,
) -> None:
    if isinstance(action_max_tries, bool) or action_max_tries < 1:
        raise ActionCacheConversionError("action_max_tries must be a positive integer")
    if action_timeout_seconds <= 0:
        raise ActionCacheConversionError("action_timeout_seconds must be positive")
    if not 0 <= page_transition_wait_seconds <= 30:
        raise ActionCacheConversionError(
            "page_transition_wait_seconds must be between 0 and 30"
        )


def _select_locator(
    candidate: DeterministicStepCandidate,
    observed_step: ObservedStep,
    *,
    allow_unvalidated_locators: bool,
) -> tuple[str, LocatorSource]:
    if candidate.chosen_playwright_locator:
        # Cache format 1.1 cannot yet store a positive live-validation result.
        if not allow_unvalidated_locators:
            raise ActionCacheConversionError(
                "Locator validation is required",
                [
                    _unconverted_step(
                        observed_step,
                        explanation=(
                            "The cache cannot prove that its chosen locator passed "
                            "a fresh-browser replay."
                        ),
                    )
                ],
            )
        return (
            candidate.chosen_playwright_locator,
            LocatorSource.CHOSEN_CACHE_LOCATOR,
        )
    if not allow_unvalidated_locators:
        raise ActionCacheConversionError(
            "Locator validation is required",
            [
                _unconverted_step(
                    observed_step,
                    explanation=(
                        "Only an unvalidated locator candidate is available. Pass "
                        "allow_unvalidated_locators=True to generate a draft."
                    ),
                )
            ],
        )
    return (
        candidate.highest_ranked_unvalidated_locator,
        LocatorSource.HIGHEST_RANKED_UNVALIDATED_LOCATOR,
    )


def _direct_candidate_provenance_problem(
    observed_step: ObservedStep,
    candidate: DirectActionCandidate,
) -> UnconvertedStep | None:
    """Prove that a typed direct candidate matches its recorded source action."""

    action = candidate.direct_action
    source_action = observed_step.browser_action.original_action_name
    details = observed_step.browser_action.action_details

    if isinstance(action, DirectSleepAction):
        if source_action != "wait":
            return _unconverted_step(
                observed_step,
                explanation="A direct sleep candidate must come from a wait action.",
            )
        recorded_seconds = details.get("seconds", 3)
        if (
            set(details) - {"seconds"}
            or isinstance(recorded_seconds, bool)
            or not isinstance(recorded_seconds, int)
            or recorded_seconds != action.requested_seconds
            or action.replay_seconds != min(max(action.requested_seconds - 1, 0), 30)
        ):
            return _unconverted_step(
                observed_step,
                explanation=(
                    "The direct sleep duration does not match the recorded wait action."
                ),
            )
        return None

    if isinstance(action, DirectGoBackAction):
        description = details.get("description")
        if source_action != "go_back":
            return _unconverted_step(
                observed_step,
                explanation="A direct go-back candidate must come from go_back.",
            )
        if set(details) - {"description"} or (
            description is not None and not isinstance(description, str)
        ):
            return _unconverted_step(
                observed_step,
                explanation=(
                    "The recorded go-back details contain behavior that the direct "
                    "adapter cannot reproduce."
                ),
            )
        return None

    if isinstance(action, DirectNavigateAction):
        if source_action != "navigate":
            return _unconverted_step(
                observed_step,
                explanation=(
                    "A direct navigation candidate must come from a navigate action."
                ),
            )
        if (
            set(details) != {"url", "open_in_new_tab"}
            or details.get("url") != action.url
            or details.get("open_in_new_tab") != action.new_tab
        ):
            return _unconverted_step(
                observed_step,
                explanation=(
                    "The direct navigation URL or tab behavior does not match the "
                    "recorded navigate action."
                ),
            )
        return None

    if isinstance(action, DirectSearchAction):
        if source_action != "search":
            return _unconverted_step(
                observed_step,
                explanation="A direct search candidate must come from search.",
            )
        query = details.get("query")
        engine = details.get("engine")
        if not isinstance(query, str) or not isinstance(engine, str):
            return _unconverted_step(
                observed_step,
                explanation="The recorded search query or engine is malformed.",
            )
        encoded_query = urllib.parse.quote_plus(query)
        expected_urls = {
            "duckduckgo": f"https://duckduckgo.com/?q={encoded_query}",
            "google": f"https://www.google.com/search?q={encoded_query}&udm=14",
            "bing": f"https://www.bing.com/search?q={encoded_query}",
        }
        normalised_engine = engine.lower()
        if (
            set(details) != {"query", "engine"}
            or action.query != query
            or action.engine != normalised_engine
            or expected_urls.get(normalised_engine) != action.url
        ):
            return _unconverted_step(
                observed_step,
                explanation="The direct search candidate does not match its recorded action.",
            )
        return None

    if isinstance(action, DirectScrollAction):
        if source_action != "scroll":
            return _unconverted_step(
                observed_step,
                explanation="A direct scroll candidate must come from scroll.",
            )
        if (
            set(details) != {"down", "pages"}
            or details.get("down") != action.down
            or details.get("pages") != action.pages
            or observed_step.browser_action.temporary_browser_use_element_index
            not in {None, 0}
        ):
            return _unconverted_step(
                observed_step,
                explanation="The direct scroll candidate does not match its page-level source action.",
            )
        return None

    if isinstance(action, DirectSendKeysAction):
        if (
            source_action != "send_keys"
            or set(details) != {"keys"}
            or details.get("keys") != action.keys
        ):
            return _unconverted_step(
                observed_step,
                explanation="The direct key sequence does not match the recorded send_keys action.",
            )
        return None

    if isinstance(action, DirectFindTextAction):
        if (
            source_action != "find_text"
            or set(details) != {"text"}
            or details.get("text") != action.text
        ):
            return _unconverted_step(
                observed_step,
                explanation="The direct text target does not match the recorded find_text action.",
            )
        return None

    return _unconverted_step(
        observed_step,
        explanation="The direct candidate has no supported provenance check.",
    )


def _candidate_provenance_problem(
    observed_step: ObservedStep,
    candidate: DeterministicStepCandidate,
) -> UnconvertedStep | None:
    """Prove that candidate behavior is derived from this observed source step."""

    action = candidate.playwright_action
    recorded_details = observed_step.browser_action.action_details
    if isinstance(action, PlaywrightAction):
        recorded_text = recorded_details.get("text")
        clear_existing_text = recorded_details.get("clear_existing_text")
        expected_action_type = (
            "fill"
            if clear_existing_text is True
            else "type" if clear_existing_text is False else None
        )
        if (
            not isinstance(recorded_text, str)
            or action.input_text != recorded_text
            or action.action_type != expected_action_type
        ):
            return _unconverted_step(
                observed_step,
                explanation=(
                    "The cached input value or fill/type behavior does not match "
                    "the observed Browser Use action."
                ),
            )
    elif isinstance(action, PlaywrightSelectAction):
        recorded_option = recorded_details.get("text")
        if (
            not isinstance(recorded_option, str)
            or action.option_text != recorded_option
        ):
            return _unconverted_step(
                observed_step,
                explanation=(
                    "The cached select option does not match the observed Browser "
                    "Use action."
                ),
            )
    elif isinstance(action, PlaywrightUploadAction):
        recorded_path = recorded_details.get("path")
        if not isinstance(recorded_path, str) or action.file_path != recorded_path:
            return _unconverted_step(
                observed_step,
                explanation=(
                    "The cached upload path does not match the observed Browser "
                    "Use action."
                ),
            )
        if (
            observed_step.element_used is None
            or observed_step.element_used.html_tag != "input"
            or observed_step.element_used.locator_relevant_attributes.get("type")
            != "file"
        ):
            return _unconverted_step(
                observed_step,
                explanation="The upload source lacks input[type=file] target evidence.",
            )
    elif isinstance(action, PlaywrightScrollAction):
        if (
            recorded_details.get("down") != action.down
            or recorded_details.get("pages") != action.pages
            or set(recorded_details) != {"down", "pages"}
            or observed_step.browser_action.temporary_browser_use_element_index
            in {None, 0}
        ):
            return _unconverted_step(
                observed_step,
                explanation=(
                    "The cached element scroll does not match the recorded Browser "
                    "Use action."
                ),
            )
    elif not isinstance(action, PlaywrightClickAction):
        return _unconverted_step(
            observed_step,
            explanation="The cached action has no supported evidence check.",
        )

    if observed_step.element_used is None:
        return _unconverted_step(
            observed_step,
            explanation="The source step has no element evidence for its locator.",
        )
    expected_locator_options = build_locator_options(observed_step.element_used)
    if candidate.playwright_locator_options != expected_locator_options:
        return _unconverted_step(
            observed_step,
            explanation=(
                "The cached locator options do not match locators regenerated from "
                "the observed element evidence."
            ),
        )
    return None


def _recorded_page_transition_after_step(
    current_step: ObservedStep,
    next_step: ObservedStep | None,
) -> bool:
    if next_step is None:
        return False
    current_page = current_step.page_before_action_batch
    next_page = next_step.page_before_action_batch
    return bool(
        current_page.url
        and next_page.url
        and current_page.url != next_page.url
        and current_step.history_location.browser_use_history_item
        < next_step.history_location.browser_use_history_item
    )


def _next_step(
    observed_steps: list[ObservedStep],
    position: int,
) -> ObservedStep | None:
    next_position = position + 1
    return (
        observed_steps[next_position] if next_position < len(observed_steps) else None
    )


def _planned_unresolved_step(
    observed_step: ObservedStep,
    *,
    explanation: str | None = None,
    problem: UnconvertedStep | None = None,
) -> PlannedStep:
    unresolved = problem or _unconverted_step(
        observed_step,
        explanation=explanation,
    )
    return PlannedStep(
        source_step_number=observed_step.step_number,
        browser_use_action=observed_step.browser_action.original_action_name,
        status=PlannedStepStatus.UNRESOLVED,
        unconverted_step=unresolved,
    )


def _problem_from_conversion_error(
    error: ActionCacheConversionError,
    observed_step: ObservedStep,
) -> UnconvertedStep:
    for problem in error.problems:
        if problem.source_step_number == observed_step.step_number:
            return problem
    if error.problems:
        source_problem = error.problems[0]
        return _unconverted_step(
            observed_step,
            explanation=source_problem.explanation,
        )
    return _unconverted_step(observed_step, explanation=str(error))


def _unconverted_step(
    observed_step: ObservedStep,
    *,
    explanation: str | None = None,
) -> UnconvertedStep:
    decision = observed_step.cached_automation_decision
    return UnconvertedStep(
        source_step_number=observed_step.step_number,
        browser_use_action=observed_step.browser_action.original_action_name,
        cache_decision=decision.decision.value,
        explanation=explanation or decision.explanation,
    )


def _compact_automation_payload(automation: Automation) -> dict:
    payload = automation.model_dump(
        mode="json",
        exclude_none=True,
        exclude_defaults=True,
    )
    # The target schema, rather than a hand-maintained action-name switch, owns
    # the complete Automation JSON shape. This keeps new typed adapters safe as
    # long as they construct and validate real Optexity models first.
    Automation.model_validate(payload)
    return payload


def _write_atomically(
    serialized: str,
    destination_path: Path,
    *,
    overwrite: bool,
) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination_path.parent,
            prefix=f".{destination_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            os.chmod(temporary_path, 0o600)
            temporary_file.write(serialized)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        if overwrite:
            os.replace(temporary_path, destination_path)
        else:
            # A hard link creates the destination atomically and fails if another
            # process won the race after the earlier existence check.
            os.link(temporary_path, destination_path)
            temporary_path.unlink()
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
