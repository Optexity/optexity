from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from optexity.inference.core.automation_cache.locator_commands import (
    validate_playwright_locator_command,
)
from optexity.inference.core.automation_cache.models import (
    ActionCacheConversionError,
)
from optexity.inference.core.learning_memory.models import (
    LearnedStep,
    LearnedStepStrategy,
    LearningPolicy,
    LocatorCapability,
    LocatorValidationEvent,
    LocatorValidationOutcome,
)
from optexity.inference.infra.browser import Browser
from optexity.schema.actions.interaction_action import BaseAction
from optexity.schema.automation import ActionNode


class LocatorResolutionError(RuntimeError):
    """Raised when no bounded, evidence-linked locator candidate is usable."""

    def __init__(self, message: str, events: list[LocatorValidationEvent]):
        self.events = tuple(events)
        super().__init__(message)


class LearnedActionEffectError(RuntimeError):
    """Raised when a command ran but its deterministic state change is absent."""


@dataclass(frozen=True, slots=True)
class LocatorRequirements:
    capability: LocatorCapability
    visible_required: bool
    enabled_required: bool
    editable_required: bool = False
    checkable_required: bool = False
    file_input_required: bool = False
    trial_action: str | None = None


@dataclass(frozen=True, slots=True)
class PreparedNode:
    node: ActionNode
    events: list[LocatorValidationEvent]
    selected_candidate_index: int | None
    selected_command: str | None


LOCATOR_REQUIREMENTS: dict[str, LocatorRequirements] = {
    "click_element": LocatorRequirements(
        capability=LocatorCapability.CLICK,
        visible_required=True,
        enabled_required=True,
        trial_action="click",
    ),
    "input_text": LocatorRequirements(
        capability=LocatorCapability.INPUT,
        visible_required=True,
        enabled_required=True,
        editable_required=True,
    ),
    "select_option": LocatorRequirements(
        capability=LocatorCapability.SELECT,
        visible_required=True,
        enabled_required=True,
    ),
    "check": LocatorRequirements(
        capability=LocatorCapability.CHECK,
        visible_required=True,
        enabled_required=True,
        checkable_required=True,
        trial_action="check",
    ),
    "uncheck": LocatorRequirements(
        capability=LocatorCapability.UNCHECK,
        visible_required=True,
        enabled_required=True,
        checkable_required=True,
        trial_action="uncheck",
    ),
    "hover": LocatorRequirements(
        capability=LocatorCapability.HOVER,
        visible_required=True,
        enabled_required=False,
        trial_action="hover",
    ),
    "upload_file": LocatorRequirements(
        capability=LocatorCapability.UPLOAD,
        visible_required=False,
        enabled_required=True,
        file_input_required=True,
    ),
}


_PROBE_SCRIPT = """
(elements, args) => {
  const count = elements.length;
  if (count !== 1) {
    return {count, capabilityPassed: false};
  }

  const element = elements[0];
  const style = window.getComputedStyle(element);
  const rect = element.getBoundingClientRect();
  const visible = style.visibility !== 'hidden'
    && style.display !== 'none'
    && rect.width > 0
    && rect.height > 0;
  const tag = element.tagName.toLowerCase();
  const type = (element.getAttribute('type') || '').toLowerCase();
  const enabled = !element.matches(':disabled');
  const textInputTypes = new Set([
    '', 'text', 'email', 'password', 'search', 'tel', 'url', 'number',
    'date', 'datetime-local', 'month', 'time', 'week',
  ]);
  const inputCompatible = element.isContentEditable
    || tag === 'textarea'
    || (tag === 'input' && textInputTypes.has(type));
  const editable = inputCompatible
    && !(('readOnly' in element) && element.readOnly);
  const checkable = tag === 'input' && (type === 'checkbox' || type === 'radio');
  const fileInput = tag === 'input' && type === 'file';
  const selectCompatible = tag === 'select';
  const options = tag === 'select'
    ? Array.from(element.options).map((option) => ({
        label: option.text,
        value: option.value,
      }))
    : [];
  const selectValuesPresent = args.selectValues.every((wanted) =>
    options.some((option) => option.label === wanted || option.value === wanted)
  );

  const capabilityPassed =
    (!args.visibleRequired || visible)
    && (!args.enabledRequired || enabled)
    && (!args.editableRequired || editable)
    && (!args.checkableRequired || checkable)
    && (!args.fileInputRequired || fileInput)
    && (args.capability !== 'input' || inputCompatible)
    && (args.capability !== 'select' || (selectCompatible && selectValuesPresent));

  return {
    count,
    capabilityPassed,
    visible,
    enabled,
    editable,
    inputCompatible,
    selectCompatible,
    checkable,
    fileInput,
    selectValuesPresent,
    fingerprint: {
      tag,
      type,
      id: element.getAttribute('id'),
      name: element.getAttribute('name'),
    },
  };
}
"""

_EFFECT_SCRIPT = """
(element, args) => {
  if (args.capability === 'input') {
    const actual = String(element.value ?? element.textContent ?? '');
    return args.inputMode === 'type'
      ? actual.endsWith(args.expectedText)
      : actual === args.expectedText;
  }
  if (args.capability === 'select') {
    const selected = Array.from(element.selectedOptions).map((option) => ({
      label: option.text,
      value: option.value,
    }));
    return args.selectValues.every((wanted) =>
      selected.some((option) => option.label === wanted || option.value === wanted)
    );
  }
  if (args.capability === 'check') {
    return element.checked === true;
  }
  if (args.capability === 'uncheck') {
    return element.checked === false;
  }
  if (args.capability === 'upload') {
    return Boolean(element.files && element.files.length > 0);
  }
  return true;
}
"""


def locator_action(
    node: ActionNode,
) -> tuple[str, BaseAction, LocatorRequirements] | None:
    interaction = node.interaction_action
    if interaction is None:
        return None
    for field_name, requirements in LOCATOR_REQUIREMENTS.items():
        action = getattr(interaction, field_name)
        if action is not None:
            return field_name, action, requirements
    return None


async def prepare_action_node(
    node: ActionNode,
    learned_step: LearnedStep,
    browser: Browser,
    policy: LearningPolicy,
) -> PreparedNode:
    """Choose one validated locator without performing the real action."""

    if learned_step.strategy == LearnedStepStrategy.DIRECT:
        return PreparedNode(
            node=node.model_copy(deep=True),
            events=[],
            selected_candidate_index=None,
            selected_command=None,
        )
    if learned_step.strategy != LearnedStepStrategy.LOCATOR:
        raise LocatorResolutionError(
            f"Learned step {learned_step.node_index} is not replayable",
            [],
        )

    action_data = locator_action(node)
    if action_data is None:
        raise LocatorResolutionError(
            f"Learned step {learned_step.node_index} has no locator action",
            [],
        )
    field_name, _action, requirements = action_data
    if learned_step.capability != requirements.capability:
        raise LocatorResolutionError(
            f"Learned step {learned_step.node_index} changed locator capability",
            [],
        )

    primary_index = learned_step.chosen_candidate_index or 0
    candidate_indexes = [primary_index]
    candidate_indexes.extend(
        index for index in range(len(learned_step.candidates)) if index != primary_index
    )
    candidate_indexes = candidate_indexes[: 1 + policy.max_alternatives]

    events: list[LocatorValidationEvent] = []
    repair_started = time.monotonic()
    select_values: list[str] = []
    if field_name == "select_option" and node.interaction_action is not None:
        select_action = node.interaction_action.select_option
        if select_action is not None:
            select_values = list(select_action.select_values or [])

    for candidate_index in candidate_indexes:
        elapsed_repair_ms = (time.monotonic() - repair_started) * 1000
        remaining_ms = policy.repair_budget_ms - elapsed_repair_ms
        if remaining_ms <= 0:
            break
        candidate = learned_step.candidates[candidate_index]
        event = await _validate_candidate(
            command=candidate.command,
            node_index=learned_step.node_index,
            candidate_index=candidate_index,
            requirements=requirements,
            select_values=select_values,
            browser=browser,
            timeout_ms=min(policy.candidate_timeout_ms, remaining_ms),
            soft_target_ms=policy.soft_validation_target_ms,
        )
        events.append(event)
        if event.outcome != LocatorValidationOutcome.PASSED:
            continue

        prepared = node.model_copy(deep=True)
        prepared_data = locator_action(prepared)
        assert prepared_data is not None
        _, prepared_action, _ = prepared_data
        prepared_action.command = candidate.command
        prepared_action.skip_command = False
        prepared_action.skip_prompt = True
        prepared_action.assert_locator_presence = True
        return PreparedNode(
            node=prepared,
            events=events,
            selected_candidate_index=candidate_index,
            selected_command=candidate.command,
        )

    raise LocatorResolutionError(
        f"No locator candidate passed for learned step {learned_step.node_index}",
        events,
    )


async def verify_action_effect(
    node: ActionNode,
    command: str | None,
    browser: Browser,
    *,
    timeout_ms: float,
) -> None:
    """Verify stateful element actions after the existing handler executes."""

    if command is None:
        return
    action_data = locator_action(node)
    if action_data is None:
        return
    field_name, action, requirements = action_data
    if requirements.capability not in {
        LocatorCapability.INPUT,
        LocatorCapability.SELECT,
        LocatorCapability.CHECK,
        LocatorCapability.UNCHECK,
        LocatorCapability.UPLOAD,
    }:
        return

    expected_text = ""
    input_mode = "fill"
    select_values: list[str] = []
    if field_name == "input_text":
        expected_text = str(getattr(action, "input_text", "") or "")
        input_mode = str(getattr(action, "fill_or_type", "fill"))
    elif field_name == "select_option":
        select_values = list(getattr(action, "select_values", None) or [])

    async def _verify() -> bool:
        locator = await browser.get_locator_from_command(command)
        if locator is None:
            return False
        return bool(
            await locator.evaluate(
                _EFFECT_SCRIPT,
                {
                    "capability": requirements.capability.value,
                    "expectedText": expected_text,
                    "inputMode": input_mode,
                    "selectValues": select_values,
                },
            )
        )

    try:
        verified = await asyncio.wait_for(
            _verify(), timeout=max(timeout_ms, 1.0) / 1000
        )
    except TimeoutError as exc:
        raise LearnedActionEffectError(
            f"Post-action verification timed out for {field_name}"
        ) from exc
    if not verified:
        raise LearnedActionEffectError(
            f"Post-action state did not match the learned {field_name} action"
        )


async def _validate_candidate(
    *,
    command: str,
    node_index: int,
    candidate_index: int,
    requirements: LocatorRequirements,
    select_values: list[str],
    browser: Browser,
    timeout_ms: float,
    soft_target_ms: float,
) -> LocatorValidationEvent:
    started = time.monotonic()
    deadline = started + max(timeout_ms, 1.0) / 1000
    outcome = LocatorValidationOutcome.ERROR
    matched_count: int | None = None
    explanation: str | None = None

    try:
        validate_playwright_locator_command(
            command,
            source_step_number=node_index + 1,
            browser_use_action=requirements.capability.value,
        )
    except (ActionCacheConversionError, ValueError) as exc:
        outcome = LocatorValidationOutcome.INVALID_COMMAND
        explanation = f"{type(exc).__name__}: {exc}"
    else:
        try:

            async def _run_validation() -> dict:
                locator = await browser.get_locator_from_command(command)
                if locator is None:
                    raise RuntimeError("No active page could resolve the locator")
                probe_result = await locator.evaluate_all(
                    _PROBE_SCRIPT,
                    {
                        "capability": requirements.capability.value,
                        "visibleRequired": requirements.visible_required,
                        "enabledRequired": requirements.enabled_required,
                        "editableRequired": requirements.editable_required,
                        "checkableRequired": requirements.checkable_required,
                        "fileInputRequired": requirements.file_input_required,
                        "selectValues": select_values,
                    },
                )
                if probe_result.get("count") != 1 or not probe_result.get(
                    "capabilityPassed", False
                ):
                    return probe_result

                remaining_ms = max((deadline - time.monotonic()) * 1000, 1.0)
                if requirements.trial_action == "click":
                    await locator.click(trial=True, timeout=remaining_ms)
                elif requirements.trial_action == "hover":
                    await locator.hover(trial=True, timeout=remaining_ms)
                elif requirements.trial_action == "check":
                    await locator.check(trial=True, timeout=remaining_ms)
                elif requirements.trial_action == "uncheck":
                    await locator.uncheck(trial=True, timeout=remaining_ms)
                return probe_result

            probe = await asyncio.wait_for(
                _run_validation(),
                timeout=max(deadline - time.monotonic(), 0.001),
            )
            matched_count = int(probe.get("count", 0))
            if matched_count == 0:
                outcome = LocatorValidationOutcome.NO_MATCH
            elif matched_count > 1:
                outcome = LocatorValidationOutcome.MULTIPLE_MATCHES
            elif not probe.get("capabilityPassed", False):
                outcome = LocatorValidationOutcome.CAPABILITY_MISMATCH
                explanation = json_safe_probe_explanation(probe)
            else:
                outcome = LocatorValidationOutcome.PASSED
        except TimeoutError:
            outcome = LocatorValidationOutcome.TIMED_OUT
        # Playwright and Patchright expose several backend-specific transport
        # and actionability exception classes. Every failure is converted into
        # a non-promotable validation event at this boundary.
        except Exception as exc:  # noqa: BLE001
            outcome = LocatorValidationOutcome.ERROR
            explanation = f"{type(exc).__name__}: {exc}"

    elapsed_ms = (time.monotonic() - started) * 1000
    return LocatorValidationEvent(
        node_index=node_index,
        candidate_index=candidate_index,
        command=command,
        capability=requirements.capability,
        outcome=outcome,
        elapsed_ms=elapsed_ms,
        exceeded_soft_target=elapsed_ms > soft_target_ms,
        matched_count=matched_count,
        explanation=explanation,
    )


def json_safe_probe_explanation(probe: dict) -> str:
    keys = (
        "visible",
        "enabled",
        "editable",
        "inputCompatible",
        "selectCompatible",
        "checkable",
        "fileInput",
        "selectValuesPresent",
        "fingerprint",
    )
    return ", ".join(f"{key}={probe.get(key)!r}" for key in keys)
