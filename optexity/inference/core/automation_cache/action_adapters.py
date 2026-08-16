from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from browser_use.agent.history_compiler import (
    DeterministicStepCandidate,
    ObservedStep,
    PlaywrightAction,
    PlaywrightClickAction,
    PlaywrightSelectAction,
)

from optexity.inference.core.automation_cache.models import (
    ActionCacheConversionError,
    UnconvertedStep,
)
from optexity.schema.actions.interaction_action import (
    ClickElementAction,
    InputTextAction,
    InteractionAction,
    SelectOptionAction,
)
from optexity.schema.automation import ActionNode


@dataclass(frozen=True, slots=True)
class ActionAdapterContext:
    observed_step: ObservedStep
    candidate: DeterministicStepCandidate
    playwright_command: str
    enable_prompt_fallback: bool
    end_sleep_time: float
    max_tries: int
    max_timeout_seconds_per_try: float


@dataclass(frozen=True, slots=True)
class AdaptedAction:
    node: ActionNode
    optexity_action_name: str


ActionAdapter = Callable[[ActionAdapterContext], AdaptedAction]


def build_action_node(context: ActionAdapterContext) -> AdaptedAction:
    """Dispatch a typed cached action to its Optexity action adapter."""

    action_type = context.candidate.playwright_action.action_type
    adapter = ACTION_ADAPTERS.get(action_type)
    if adapter is None:
        raise ActionCacheConversionError(
            "Unsupported deterministic candidate",
            [
                UnconvertedStep(
                    source_step_number=context.observed_step.step_number,
                    browser_use_action=(
                        context.observed_step.browser_action.original_action_name
                    ),
                    explanation=(
                        f"No Optexity adapter is registered for cached action "
                        f"{action_type!r}."
                    ),
                )
            ],
        )
    return adapter(context)


def _build_input_action(context: ActionAdapterContext) -> AdaptedAction:
    action = context.candidate.playwright_action
    if not isinstance(action, PlaywrightAction):
        return _raise_action_type_mismatch(context)
    normalised_input_text = action.input_text.lower()
    if "<secret" in normalised_input_text or "</secret" in normalised_input_text:
        raise ActionCacheConversionError(
            "A redacted secret cannot be emitted as literal input",
            [
                UnconvertedStep(
                    source_step_number=context.observed_step.step_number,
                    browser_use_action=(
                        context.observed_step.browser_action.original_action_name
                    ),
                    explanation=(
                        "Map Browser Use secret placeholders to an explicit Optexity "
                        "secure parameter before conversion."
                    ),
                )
            ],
        )

    prompt = (
        _build_prompt(context.observed_step, action.action_type)
        if context.enable_prompt_fallback
        else ""
    )
    input_action = InputTextAction(
        command=context.playwright_command,
        prompt_instructions=prompt,
        input_text=action.input_text,
        fill_or_type=action.action_type,
        skip_prompt=not context.enable_prompt_fallback,
        assert_locator_presence=not context.enable_prompt_fallback,
    )
    return AdaptedAction(
        node=_wrap_interaction(
            input_text=input_action,
            end_sleep_time=context.end_sleep_time,
            max_tries=context.max_tries,
            max_timeout_seconds_per_try=context.max_timeout_seconds_per_try,
        ),
        optexity_action_name="input_text",
    )


def _build_click_action(context: ActionAdapterContext) -> AdaptedAction:
    action = context.candidate.playwright_action
    if not isinstance(action, PlaywrightClickAction):
        return _raise_action_type_mismatch(context)

    prompt = (
        _build_prompt(context.observed_step, action.action_type)
        if context.enable_prompt_fallback
        else ""
    )
    click_action = ClickElementAction(
        command=context.playwright_command,
        prompt_instructions=prompt,
        skip_prompt=not context.enable_prompt_fallback,
        assert_locator_presence=not context.enable_prompt_fallback,
    )
    return AdaptedAction(
        node=_wrap_interaction(
            click_element=click_action,
            end_sleep_time=context.end_sleep_time,
            max_tries=context.max_tries,
            max_timeout_seconds_per_try=context.max_timeout_seconds_per_try,
        ),
        optexity_action_name="click_element",
    )


def _build_select_action(context: ActionAdapterContext) -> AdaptedAction:
    action = context.candidate.playwright_action
    if not isinstance(action, PlaywrightSelectAction):
        return _raise_action_type_mismatch(context)

    prompt = (
        _build_prompt(
            context.observed_step,
            action.action_type,
            option_text=action.option_text,
        )
        if context.enable_prompt_fallback
        else ""
    )
    select_action = SelectOptionAction(
        command=context.playwright_command,
        prompt_instructions=prompt,
        select_values=[action.option_text],
        skip_prompt=not context.enable_prompt_fallback,
        assert_locator_presence=not context.enable_prompt_fallback,
    )
    return AdaptedAction(
        node=_wrap_interaction(
            select_option=select_action,
            end_sleep_time=context.end_sleep_time,
            max_tries=context.max_tries,
            max_timeout_seconds_per_try=context.max_timeout_seconds_per_try,
        ),
        optexity_action_name="select_option",
    )


def _wrap_interaction(
    *,
    end_sleep_time: float,
    max_tries: int,
    max_timeout_seconds_per_try: float,
    input_text: InputTextAction | None = None,
    click_element: ClickElementAction | None = None,
    select_option: SelectOptionAction | None = None,
) -> ActionNode:
    interaction_action = InteractionAction(
        max_tries=max_tries,
        max_timeout_seconds_per_try=max_timeout_seconds_per_try,
        input_text=input_text,
        click_element=click_element,
        select_option=select_option,
    )
    return ActionNode.model_validate(
        {
            "type": "action_node",
            "interaction_action": interaction_action,
            "end_sleep_time": end_sleep_time,
        }
    )


def _build_prompt(
    observed_step: ObservedStep,
    action_type: str,
    *,
    option_text: str | None = None,
) -> str:
    """Build an atomic fallback prompt using recorded element evidence only."""

    element = observed_step.element_used
    if element is None or element.html_tag is None:
        raise _missing_prompt_evidence(observed_step)

    target_evidence: list[str] = []
    if element.accessibility_name:
        target_evidence.append(
            f"with accessible name {json.dumps(element.accessibility_name)}"
        )
    attributes = element.locator_relevant_attributes
    for attribute_name in (
        "aria-label",
        "placeholder",
        "data-testid",
        "data-test-id",
        "data-test",
        "data-cy",
        "data-qa",
        "id",
        "name",
    ):
        value = attributes.get(attribute_name)
        if value:
            target_evidence.append(
                f"with {attribute_name}={json.dumps(value, ensure_ascii=False)}"
            )
            break
    if not target_evidence:
        raise _missing_prompt_evidence(observed_step)

    readable_tag = "link" if element.html_tag == "a" else element.html_tag
    target = f"recorded {readable_tag} element {' and '.join(target_evidence)}"
    if action_type in {"fill", "type"}:
        return f"Enter the supplied value in the {target}."
    if action_type == "click":
        return f"Click the {target}."
    if action_type == "select_option" and option_text is not None:
        return f"Select {json.dumps(option_text, ensure_ascii=False)} in the {target}."
    raise _missing_prompt_evidence(observed_step)


def _missing_prompt_evidence(observed_step: ObservedStep) -> ActionCacheConversionError:
    return ActionCacheConversionError(
        "Cannot build cache-derived locator fallback",
        [
            UnconvertedStep(
                source_step_number=observed_step.step_number,
                browser_use_action=observed_step.browser_action.original_action_name,
                explanation=(
                    "The unvalidated locator has no accessibility name or stable "
                    "attribute from which to build an atomic fallback prompt."
                ),
            )
        ],
    )


def _raise_action_type_mismatch(
    context: ActionAdapterContext,
) -> AdaptedAction:
    raise ActionCacheConversionError(
        "Cached action type does not match its adapter",
        [
            UnconvertedStep(
                source_step_number=context.observed_step.step_number,
                browser_use_action=(
                    context.observed_step.browser_action.original_action_name
                ),
                explanation="The typed cached action does not match its adapter.",
            )
        ],
    )


ACTION_ADAPTERS: dict[str, ActionAdapter] = {
    "fill": _build_input_action,
    "type": _build_input_action,
    "click": _build_click_action,
    "select_option": _build_select_action,
}
