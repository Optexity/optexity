from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from browser_use.agent.history_compiler import (
    BrowserActionExecutionStatus,
    BrowserUseActionCache,
    CachedAutomationDecisionStatus,
    DeterministicStepCandidate,
    ObservedStep,
    PlaywrightAction,
    PlaywrightClickAction,
    PlaywrightSelectAction,
    build_locator_options,
)
from pydantic import ValidationError

from optexity.inference.core.automation_cache.action_adapters import (
    ActionAdapterContext,
    build_action_node,
)
from optexity.inference.core.automation_cache.locator_commands import (
    validate_playwright_locator_command,
)
from optexity.inference.core.automation_cache.models import (
    ActionCacheConversionError,
    AutomationConversionResult,
    AutomationConversionStatus,
    ConvertedStep,
    ExcludedStep,
    LocatorSource,
    UnconvertedStep,
)
from optexity.schema.automation import ActionNode, Automation, Parameters

SUPPORTED_CACHE_FORMAT_VERSION = "1.1"
DEFAULT_ACTION_MAX_TRIES = 10
DEFAULT_ACTION_TIMEOUT_SECONDS = 1.0
DEFAULT_PAGE_TRANSITION_WAIT_SECONDS = 0.5


def convert_action_cache(
    cache: BrowserUseActionCache,
    *,
    allow_unvalidated_locators: bool = False,
    allow_unresolved_select_options: bool = False,
    allow_literal_password_inputs: bool = False,
    action_max_tries: int = DEFAULT_ACTION_MAX_TRIES,
    action_timeout_seconds: float = DEFAULT_ACTION_TIMEOUT_SECONDS,
    page_transition_wait_seconds: float = DEFAULT_PAGE_TRANSITION_WAIT_SECONDS,
) -> AutomationConversionResult:
    """Convert a complete action cache into a new, ordered Optexity Automation.

    The cache is the sole workflow source. Dashboard automations are not used as
    templates. Current cache format 1.1 carries only untested locator candidates,
    so callers must opt in to a draft that uses command-first locator fallbacks.
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
    converted_steps: list[ConvertedStep] = []
    excluded_steps: list[ExcludedStep] = []
    unresolved_select_steps: list[int] = []
    literal_password_steps: list[int] = []
    nodes = []
    problems: list[UnconvertedStep] = []
    used_candidate_numbers: set[int] = set()

    for position, observed_step in enumerate(cache.all_observed_steps):
        decision = observed_step.cached_automation_decision
        if decision.decision == CachedAutomationDecisionStatus.EXCLUDE_TERMINAL_ACTION:
            excluded_steps.append(
                ExcludedStep(
                    source_step_number=observed_step.step_number,
                    browser_use_action=observed_step.browser_action.original_action_name,
                    cache_decision=decision.decision.value,
                    explanation=decision.explanation,
                )
            )
            continue
        if (
            decision.decision
            != CachedAutomationDecisionStatus.WAITING_FOR_LOCATOR_VALIDATION
        ):
            problems.append(_unconverted_step(observed_step))
            continue

        candidate_number = decision.deterministic_candidate_number
        candidate = candidates.get(candidate_number) if candidate_number else None
        if (
            candidate is None
            or candidate.source_step_number != observed_step.step_number
        ):
            problems.append(
                _unconverted_step(
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
            problems.append(
                _unconverted_step(
                    observed_step,
                    explanation=(
                        "A deterministic candidate must come from an action that "
                        "executed without a reported error."
                    ),
                )
            )
            continue
        if observed_step.browser_action.unhandled_action_arguments:
            problems.append(
                _unconverted_step(
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
        }.get(candidate.playwright_action.action_type)
        if (
            expected_browser_use_action
            != observed_step.browser_action.original_action_name
        ):
            problems.append(
                _unconverted_step(
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
            problems.append(provenance_problem)
            continue
        if (
            isinstance(candidate.playwright_action, PlaywrightAction)
            and observed_step.element_used is not None
            and observed_step.element_used.locator_relevant_attributes.get("type")
            == "password"
            and not allow_literal_password_inputs
        ):
            problems.append(
                _unconverted_step(
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
            isinstance(candidate.playwright_action, PlaywrightAction)
            and observed_step.element_used is not None
            and observed_step.element_used.locator_relevant_attributes.get("type")
            == "password"
        ):
            literal_password_steps.append(observed_step.step_number)
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

        if (
            isinstance(candidate.playwright_action, PlaywrightSelectAction)
            and candidate.playwright_action.option_match == "unresolved"
        ):
            if not allow_unresolved_select_options:
                problems.append(
                    _unconverted_step(
                        observed_step,
                        explanation=(
                            "The cache does not yet know whether this select option "
                            "matches by label or value. Pass "
                            "allow_unresolved_select_options=True only for a draft."
                        ),
                    )
                )
                continue
            unresolved_select_steps.append(observed_step.step_number)

        recorded_transition = _recorded_page_transition_after_step(
            observed_step,
            _next_step(cache.all_observed_steps, position),
        )
        adapted = build_action_node(
            ActionAdapterContext(
                observed_step=observed_step,
                candidate=candidate,
                playwright_command=locator,
                # Cache format 1.1 cannot yet prove live locator validation, even
                # when a locator was manually marked chosen. Keep command-first
                # fallback enabled until a replay certificate can be persisted.
                enable_prompt_fallback=True,
                end_sleep_time=(
                    page_transition_wait_seconds if recorded_transition else 0.0
                ),
                max_tries=action_max_tries,
                max_timeout_seconds_per_try=action_timeout_seconds,
            )
        )
        nodes.append(adapted.node)
        converted_steps.append(
            ConvertedStep(
                source_step_number=observed_step.step_number,
                deterministic_candidate_number=candidate.candidate_number,
                browser_use_action=observed_step.browser_action.original_action_name,
                optexity_action=adapted.optexity_action_name,
                playwright_command=locator,
                locator_source=locator_source,
                prompt_fallback_enabled=True,
                recorded_page_transition_after_step=recorded_transition,
            )
        )

    unused_candidates = sorted(set(candidates) - used_candidate_numbers)
    if unused_candidates:
        problems.append(
            UnconvertedStep(
                explanation=(
                    "The cache contains deterministic candidates that are not "
                    f"referenced by an observed step: {unused_candidates}."
                )
            )
        )
    if (
        len(excluded_steps) != 1
        or not cache.all_observed_steps
        or excluded_steps[0].source_step_number
        != cache.all_observed_steps[-1].step_number
        or cache.all_observed_steps[-1].browser_action.original_action_name != "done"
        or cache.all_observed_steps[-1].browser_action_result.status
        != BrowserActionExecutionStatus.TASK_COMPLETED_SUCCESSFULLY
    ):
        problems.append(
            UnconvertedStep(
                explanation=(
                    "A convertible source run must end with exactly one successful "
                    "terminal done action."
                )
            )
        )
    if problems:
        raise ActionCacheConversionError(
            "The action cache cannot produce a complete Automation",
            problems,
        )
    if not nodes:
        raise ActionCacheConversionError(
            "The action cache contains no executable deterministic candidates"
        )

    automation = Automation(
        url=cache.original_run.starting_url or "",
        parameters=Parameters(input_parameters={}, generated_parameters={}),
        nodes=nodes,
        max_retries=0,
    )
    automation = Automation.model_validate(automation.model_dump(mode="json"))
    return AutomationConversionResult(
        status=AutomationConversionStatus.DRAFT_REQUIRES_REPLAY_VALIDATION,
        automation=automation,
        converted_steps=converted_steps,
        excluded_steps=excluded_steps,
        uses_unvalidated_locators=True,
        unresolved_select_option_steps=unresolved_select_steps,
        literal_password_input_steps=literal_password_steps,
    )


def convert_action_cache_file(
    cache_path: str | Path,
    automation_path: str | Path,
    *,
    allow_unvalidated_locators: bool = False,
    allow_unresolved_select_options: bool = False,
    allow_literal_password_inputs: bool = False,
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
        allow_unvalidated_locators=allow_unvalidated_locators,
        allow_unresolved_select_options=allow_unresolved_select_options,
        allow_literal_password_inputs=allow_literal_password_inputs,
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


def _validate_cache_header(cache: BrowserUseActionCache) -> None:
    problems: list[UnconvertedStep] = []
    if cache.cache_format_version != SUPPORTED_CACHE_FORMAT_VERSION:
        problems.append(
            UnconvertedStep(
                explanation=(
                    f"Cache format {cache.cache_format_version!r} is not supported; "
                    "recompile raw_history.json with cache format 1.1."
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
    payload = {
        "url": automation.url,
        "max_retries": automation.max_retries,
        "parameters": {
            "input_parameters": automation.parameters.input_parameters,
            "generated_parameters": automation.parameters.generated_parameters,
        },
        "nodes": [],
    }
    for node in automation.nodes:
        if not isinstance(node, ActionNode) or node.interaction_action is None:
            raise ActionCacheConversionError(
                "The generated Automation contains a non-interaction node"
            )
        interaction = node.interaction_action
        interaction_payload = {
            "max_tries": interaction.max_tries,
            "max_timeout_seconds_per_try": interaction.max_timeout_seconds_per_try,
        }
        if interaction.input_text:
            action = interaction.input_text
            interaction_payload["input_text"] = {
                "command": action.command,
                "prompt_instructions": action.prompt_instructions,
                "input_text": action.input_text,
                "fill_or_type": action.fill_or_type,
                "skip_prompt": action.skip_prompt,
                "assert_locator_presence": action.assert_locator_presence,
            }
        elif interaction.click_element:
            action = interaction.click_element
            interaction_payload["click_element"] = {
                "command": action.command,
                "prompt_instructions": action.prompt_instructions,
                "skip_prompt": action.skip_prompt,
                "assert_locator_presence": action.assert_locator_presence,
            }
        elif interaction.select_option:
            action = interaction.select_option
            interaction_payload["select_option"] = {
                "command": action.command,
                "prompt_instructions": action.prompt_instructions,
                "select_values": action.select_values,
                "skip_prompt": action.skip_prompt,
                "assert_locator_presence": action.assert_locator_presence,
            }
        else:
            raise ActionCacheConversionError(
                "The generated Automation contains an unsupported interaction"
            )
        payload["nodes"].append(
            {
                "type": "action_node",
                "interaction_action": interaction_payload,
                "end_sleep_time": node.end_sleep_time,
            }
        )
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
