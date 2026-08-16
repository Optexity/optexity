from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from browser_use.agent.history_compiler import (
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
)

from optexity.inference.core.automation_cache.models import (
    ActionCacheConversionError,
    UnconvertedStep,
)
from optexity.schema.actions.interaction_action import (
    ClickElementAction,
    GoBackAction,
    GoToUrlAction,
    InputTextAction,
    InteractionAction,
    KeyPressAction,
    ScrollAction,
    ScrollToTextAction,
    SearchAction,
    SelectOptionAction,
    UploadFileAction,
)
from optexity.schema.actions.misc_action import SleepAction
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
    prompt_fallback_enabled: bool


ActionAdapter = Callable[[ActionAdapterContext], AdaptedAction]


@dataclass(frozen=True, slots=True)
class DirectActionAdapterContext:
    observed_step: ObservedStep
    candidate: DirectActionCandidate
    end_sleep_time: float


DirectActionAdapter = Callable[[DirectActionAdapterContext], AdaptedAction]


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


def build_direct_action_node(context: DirectActionAdapterContext) -> AdaptedAction:
    """Dispatch a typed direct replay candidate to an Optexity action node."""

    action_type = context.candidate.direct_action.action_type
    adapter = DIRECT_ACTION_ADAPTERS.get(action_type)
    if adapter is None:
        raise ActionCacheConversionError(
            "Unsupported direct replay candidate",
            [
                UnconvertedStep(
                    source_step_number=context.observed_step.step_number,
                    browser_use_action=(
                        context.observed_step.browser_action.original_action_name
                    ),
                    explanation=(
                        f"No Optexity adapter is registered for direct action "
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
        build_evidence_prompt(context.observed_step, action.action_type)
        if context.enable_prompt_fallback
        else None
    )
    prompt_fallback_enabled = prompt is not None
    input_action = InputTextAction(
        command=context.playwright_command,
        prompt_instructions=prompt or "",
        input_text=action.input_text,
        fill_or_type=action.action_type,
        skip_prompt=not prompt_fallback_enabled,
        assert_locator_presence=not prompt_fallback_enabled,
    )
    return AdaptedAction(
        node=_wrap_interaction(
            input_text=input_action,
            end_sleep_time=context.end_sleep_time,
            max_tries=context.max_tries,
            max_timeout_seconds_per_try=context.max_timeout_seconds_per_try,
        ),
        optexity_action_name="input_text",
        prompt_fallback_enabled=prompt_fallback_enabled,
    )


def _build_click_action(context: ActionAdapterContext) -> AdaptedAction:
    action = context.candidate.playwright_action
    if not isinstance(action, PlaywrightClickAction):
        return _raise_action_type_mismatch(context)

    prompt = (
        build_evidence_prompt(context.observed_step, action.action_type)
        if context.enable_prompt_fallback
        else None
    )
    prompt_fallback_enabled = prompt is not None
    click_action = ClickElementAction(
        command=context.playwright_command,
        prompt_instructions=prompt or "",
        skip_prompt=not prompt_fallback_enabled,
        assert_locator_presence=not prompt_fallback_enabled,
    )
    return AdaptedAction(
        node=_wrap_interaction(
            click_element=click_action,
            end_sleep_time=context.end_sleep_time,
            max_tries=context.max_tries,
            max_timeout_seconds_per_try=context.max_timeout_seconds_per_try,
        ),
        optexity_action_name="click_element",
        prompt_fallback_enabled=prompt_fallback_enabled,
    )


def _build_select_action(context: ActionAdapterContext) -> AdaptedAction:
    action = context.candidate.playwright_action
    if not isinstance(action, PlaywrightSelectAction):
        return _raise_action_type_mismatch(context)

    prompt = (
        build_evidence_prompt(
            context.observed_step,
            action.action_type,
            option_text=action.option_text,
        )
        if context.enable_prompt_fallback
        else None
    )
    prompt_fallback_enabled = prompt is not None
    select_action = SelectOptionAction(
        command=context.playwright_command,
        prompt_instructions=prompt or "",
        select_values=[action.option_text],
        skip_prompt=not prompt_fallback_enabled,
        assert_locator_presence=not prompt_fallback_enabled,
    )
    return AdaptedAction(
        node=_wrap_interaction(
            select_option=select_action,
            end_sleep_time=context.end_sleep_time,
            max_tries=context.max_tries,
            max_timeout_seconds_per_try=context.max_timeout_seconds_per_try,
        ),
        optexity_action_name="select_option",
        prompt_fallback_enabled=prompt_fallback_enabled,
    )


def _build_upload_action(context: ActionAdapterContext) -> AdaptedAction:
    action = context.candidate.playwright_action
    if not isinstance(action, PlaywrightUploadAction):
        return _raise_action_type_mismatch(context)
    prompt = (
        build_evidence_prompt(context.observed_step, action.action_type)
        if context.enable_prompt_fallback
        else None
    )
    prompt_fallback_enabled = prompt is not None
    return AdaptedAction(
        node=_wrap_interaction(
            upload_file=UploadFileAction(
                command=context.playwright_command,
                prompt_instructions=prompt or "",
                file_path=action.file_path,
                skip_prompt=not prompt_fallback_enabled,
                assert_locator_presence=not prompt_fallback_enabled,
            ),
            end_sleep_time=context.end_sleep_time,
            max_tries=context.max_tries,
            max_timeout_seconds_per_try=context.max_timeout_seconds_per_try,
        ),
        optexity_action_name="upload_file",
        prompt_fallback_enabled=prompt_fallback_enabled,
    )


def _build_element_scroll_action(context: ActionAdapterContext) -> AdaptedAction:
    action = context.candidate.playwright_action
    if not isinstance(action, PlaywrightScrollAction):
        return _raise_action_type_mismatch(context)
    return AdaptedAction(
        node=_wrap_interaction(
            scroll=ScrollAction(
                down=action.down,
                pages=action.pages,
                command=context.playwright_command,
            ),
            end_sleep_time=context.end_sleep_time,
            max_tries=1,
            max_timeout_seconds_per_try=context.max_timeout_seconds_per_try,
        ),
        optexity_action_name="scroll",
        prompt_fallback_enabled=False,
    )


def _build_sleep_action(context: DirectActionAdapterContext) -> AdaptedAction:
    action = context.candidate.direct_action
    if not isinstance(action, DirectSleepAction):
        return _raise_direct_action_type_mismatch(context)
    node = ActionNode.model_validate(
        {
            "type": "action_node",
            "sleep_action": SleepAction(sleep_time=action.replay_seconds),
            "end_sleep_time": context.end_sleep_time,
        }
    )
    return AdaptedAction(
        node=node,
        optexity_action_name="sleep_action",
        prompt_fallback_enabled=False,
    )


def _build_go_back_action(context: DirectActionAdapterContext) -> AdaptedAction:
    if not isinstance(context.candidate.direct_action, DirectGoBackAction):
        return _raise_direct_action_type_mismatch(context)
    return AdaptedAction(
        node=_wrap_interaction(
            go_back=GoBackAction(),
            end_sleep_time=context.end_sleep_time,
            max_tries=1,
            max_timeout_seconds_per_try=1.0,
        ),
        optexity_action_name="go_back",
        prompt_fallback_enabled=False,
    )


def _build_navigate_action(context: DirectActionAdapterContext) -> AdaptedAction:
    action = context.candidate.direct_action
    if not isinstance(action, DirectNavigateAction):
        return _raise_direct_action_type_mismatch(context)
    return AdaptedAction(
        node=_wrap_interaction(
            go_to_url=GoToUrlAction(url=action.url, new_tab=action.new_tab),
            end_sleep_time=context.end_sleep_time,
            max_tries=1,
            max_timeout_seconds_per_try=1.0,
        ),
        optexity_action_name="go_to_url",
        prompt_fallback_enabled=False,
    )


def _build_search_action(context: DirectActionAdapterContext) -> AdaptedAction:
    action = context.candidate.direct_action
    if not isinstance(action, DirectSearchAction):
        return _raise_direct_action_type_mismatch(context)
    return AdaptedAction(
        node=_wrap_interaction(
            search=SearchAction(query=action.query, engine=action.engine),
            end_sleep_time=context.end_sleep_time,
            max_tries=1,
            max_timeout_seconds_per_try=1.0,
        ),
        optexity_action_name="search",
        prompt_fallback_enabled=False,
    )


def _build_scroll_action(context: DirectActionAdapterContext) -> AdaptedAction:
    action = context.candidate.direct_action
    if not isinstance(action, DirectScrollAction):
        return _raise_direct_action_type_mismatch(context)
    return AdaptedAction(
        node=_wrap_interaction(
            scroll=ScrollAction(down=action.down, pages=action.pages),
            end_sleep_time=context.end_sleep_time,
            max_tries=1,
            max_timeout_seconds_per_try=1.0,
        ),
        optexity_action_name="scroll",
        prompt_fallback_enabled=False,
    )


def _build_send_keys_action(context: DirectActionAdapterContext) -> AdaptedAction:
    action = context.candidate.direct_action
    if not isinstance(action, DirectSendKeysAction):
        return _raise_direct_action_type_mismatch(context)
    return AdaptedAction(
        node=_wrap_interaction(
            key_press=KeyPressAction(keys=action.keys),
            end_sleep_time=context.end_sleep_time,
            max_tries=1,
            max_timeout_seconds_per_try=1.0,
        ),
        optexity_action_name="key_press",
        prompt_fallback_enabled=False,
    )


def _build_find_text_action(context: DirectActionAdapterContext) -> AdaptedAction:
    action = context.candidate.direct_action
    if not isinstance(action, DirectFindTextAction):
        return _raise_direct_action_type_mismatch(context)
    return AdaptedAction(
        node=_wrap_interaction(
            scroll_to_text=ScrollToTextAction(text=action.text),
            end_sleep_time=context.end_sleep_time,
            max_tries=1,
            max_timeout_seconds_per_try=1.0,
        ),
        optexity_action_name="scroll_to_text",
        prompt_fallback_enabled=False,
    )


def _wrap_interaction(
    *,
    end_sleep_time: float,
    max_tries: int,
    max_timeout_seconds_per_try: float,
    input_text: InputTextAction | None = None,
    click_element: ClickElementAction | None = None,
    select_option: SelectOptionAction | None = None,
    upload_file: UploadFileAction | None = None,
    scroll: ScrollAction | None = None,
    scroll_to_text: ScrollToTextAction | None = None,
    search: SearchAction | None = None,
    key_press: KeyPressAction | None = None,
    go_back: GoBackAction | None = None,
    go_to_url: GoToUrlAction | None = None,
) -> ActionNode:
    interaction_action = InteractionAction(
        max_tries=max_tries,
        max_timeout_seconds_per_try=max_timeout_seconds_per_try,
        input_text=input_text,
        click_element=click_element,
        select_option=select_option,
        upload_file=upload_file,
        scroll=scroll,
        scroll_to_text=scroll_to_text,
        search=search,
        key_press=key_press,
        go_back=go_back,
        go_to_url=go_to_url,
    )
    return ActionNode.model_validate(
        {
            "type": "action_node",
            "interaction_action": interaction_action,
            "end_sleep_time": end_sleep_time,
        }
    )


def build_evidence_prompt(
    observed_step: ObservedStep,
    action_type: str,
    *,
    option_text: str | None = None,
) -> str | None:
    """Build an atomic fallback prompt using recorded element evidence only."""

    element = observed_step.element_used
    if element is None or element.html_tag is None:
        return None

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
        "title",
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
        return None

    readable_tag = "link" if element.html_tag == "a" else element.html_tag
    target = f"recorded {readable_tag} element {' and '.join(target_evidence)}"
    if action_type in {"fill", "type", "input_text"}:
        return f"Enter the supplied value in the {target}."
    if action_type in {"click", "click_element"}:
        return f"Click the {target}."
    if action_type == "select_option" and option_text is not None:
        return f"Select {json.dumps(option_text, ensure_ascii=False)} in the {target}."
    if action_type == "check":
        return f"Check the {target}."
    if action_type == "uncheck":
        return f"Uncheck the {target}."
    if action_type == "hover":
        return f"Hover over the {target}."
    if action_type == "upload_file":
        return f"Upload the supplied file into the {target}."
    return None


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


def _raise_direct_action_type_mismatch(
    context: DirectActionAdapterContext,
) -> AdaptedAction:
    raise ActionCacheConversionError(
        "Cached direct action type does not match its adapter",
        [
            UnconvertedStep(
                source_step_number=context.observed_step.step_number,
                browser_use_action=(
                    context.observed_step.browser_action.original_action_name
                ),
                explanation="The typed direct action does not match its adapter.",
            )
        ],
    )


ACTION_ADAPTERS: dict[str, ActionAdapter] = {
    "fill": _build_input_action,
    "type": _build_input_action,
    "click": _build_click_action,
    "select_option": _build_select_action,
    "upload_file": _build_upload_action,
    "scroll_element": _build_element_scroll_action,
}


DIRECT_ACTION_ADAPTERS: dict[str, DirectActionAdapter] = {
    "sleep": _build_sleep_action,
    "go_back": _build_go_back_action,
    "navigate": _build_navigate_action,
    "search": _build_search_action,
    "scroll": _build_scroll_action,
    "send_keys": _build_send_keys_action,
    "find_text": _build_find_text_action,
}
