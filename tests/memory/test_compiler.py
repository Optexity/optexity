"""Tests for the compiler. No browser and no LLM are needed.

Each test builds a tiny action cache by hand and checks what the compiler makes
of it. Run with:  pytest tests/memory
"""

import json
from pathlib import Path

import pytest

from optexity.memory import ActionCache, compile_cache, compile_file
from optexity.schema.automation import Automation

FIXTURES = Path(__file__).parent / "fixtures"
URL = "https://example.com/form"


def _note(step, action, params=None, *, locators=None, element=None, success=True, **extra):
    """Make one note as a plain dict (the same shape as a line in actions.json)."""
    note = {
        "step": step,
        "position": 0,
        "actions_in_step": 1,
        "action": action,
        "params": params or {},
        "locators": locators or [],
        "url_before": URL,
        "success": success,
    }
    if element is not None:
        note["element"] = element
    note.update(extra)
    return note


def _unique(locator, kind="name", score=84):
    return {"locator": locator, "kind": kind, "score": score, "match_count": 1, "unique": True}


def _not_unique(locator, kind="name", score=84):
    return {"locator": locator, "kind": kind, "score": score, "match_count": 2, "unique": False}


def _cache(*notes, start_url=URL):
    return ActionCache.model_validate(
        {"goal": "test goal", "start_url": start_url, "notes": list(notes)}
    )


def _commands(result):
    """The command of every node that has one, in order."""
    out = []
    for node in result.automation["nodes"]:
        body = node["interaction_action"]
        kind = next(iter(body))
        out.append((kind, body[kind].get("command")))
    return out


def test_roboform_fixture_compiles_to_four_cached_steps():
    result = compile_file(
        FIXTURES / "roboform_actions.json", FIXTURES / "_out_roboform.cached.json"
    )
    try:
        assert result.cached_steps == 4
        assert result.agentic_steps == 0
        assert _commands(result) == [
            ("input_text", "locator(\"input[name='02frstname']\")"),
            ("input_text", "locator(\"input[name='10address1']\")"),
            ("input_text", "locator(\"input[name='11address2']\")"),
            ("input_text", "locator(\"input[name='13city']\")"),
        ]
        # The output must be a valid Optexity workflow.
        Automation.model_validate(
            json.loads((FIXTURES / "_out_roboform.cached.json").read_text())
        )
    finally:
        (FIXTURES / "_out_roboform.cached.json").unlink(missing_ok=True)


def test_failed_scroll_and_done_steps_are_dropped():
    result = compile_cache(
        _cache(
            _note(1, "scroll"),
            _note(2, "click", success=False, error="boom"),
            _note(3, "done"),
        )
    )
    assert result.automation["nodes"] == []


def test_earlier_typing_is_dropped_when_same_box_typed_again():
    box = _unique("locator(\"input[name='a']\")")
    result = compile_cache(
        _cache(
            _note(1, "input", {"text": "wrong"}, locators=[box]),
            _note(2, "input", {"text": "right"}, locators=[box]),
        )
    )
    nodes = result.automation["nodes"]
    assert len(nodes) == 1
    assert nodes[0]["interaction_action"]["input_text"]["input_text"] == "right"


def test_earlier_typing_is_kept_when_a_click_happened_in_between():
    box = _unique("locator(\"input[name='q']\")")
    button = _unique("get_by_role(\"button\", name=\"Search\")", "role+name", 72)
    result = compile_cache(
        _cache(
            _note(1, "input", {"text": "first"}, locators=[box]),
            _note(2, "click", locators=[button]),
            _note(3, "input", {"text": "second"}, locators=[box]),
        )
    )
    assert len(result.automation["nodes"]) == 3


def test_best_unique_locator_wins_over_a_better_scored_non_unique_one():
    result = compile_cache(
        _cache(
            _note(
                1,
                "input",
                {"text": "x"},
                locators=[
                    _not_unique("locator(\"#dup\")", "id", 92),
                    _unique("get_by_placeholder(\"Name\")", "placeholder", 64),
                ],
            )
        )
    )
    assert _commands(result) == [("input_text", "get_by_placeholder(\"Name\")")]


def test_step_without_a_unique_locator_stays_agentic():
    result = compile_cache(
        _cache(
            _note(
                1,
                "click",
                locators=[_not_unique("locator(\"button\")", "css", 50)],
                next_goal="Click the big button",
            )
        )
    )
    assert result.cached_steps == 0
    assert result.agentic_steps == 1
    task = result.automation["nodes"][0]["interaction_action"]["agentic_task"]["task"]
    assert task == "Click the big button"


def test_unsupported_action_stays_agentic():
    result = compile_cache(_cache(_note(1, "extract", {"query": "price"})))
    assert result.agentic_steps == 1


def test_secret_value_is_never_written_to_the_workflow():
    result = compile_cache(
        _cache(
            _note(
                1,
                "input",
                {"text": "<redacted>"},
                locators=[_unique("locator(\"input[type='password']\")", "css", 60)],
                sensitive=True,
            )
        )
    )
    dumped = json.dumps(result.automation)
    assert "<redacted>" not in dumped
    assert "{secret_1[0]}" in dumped
    assert "secret_1" in result.automation["parameters"]["input_parameters"]


def test_first_navigate_to_the_start_url_is_dropped_but_later_ones_are_kept():
    result = compile_cache(
        _cache(
            _note(1, "navigate", {"url": URL + "/"}),
            _note(2, "navigate", {"url": "https://example.com/other"}),
        )
    )
    nodes = result.automation["nodes"]
    assert len(nodes) == 1
    assert nodes[0]["interaction_action"]["go_to_url"]["url"] == "https://example.com/other"


def test_select_dropdown_becomes_select_option():
    result = compile_cache(
        _cache(
            _note(
                1,
                "select_dropdown",
                {"index": 3, "text": "Option 2"},
                locators=[_unique("locator(\"select[name='dd']\")")],
            )
        )
    )
    body = result.automation["nodes"][0]["interaction_action"]["select_option"]
    assert body["select_values"] == ["Option 2"]


def test_send_keys_enter_is_kept_and_other_keys_stay_agentic():
    result = compile_cache(
        _cache(
            _note(1, "send_keys", {"keys": "Enter"}),
            _note(2, "send_keys", {"keys": "Control+a"}),
        )
    )
    assert result.cached_steps == 1
    assert result.agentic_steps == 1


def test_every_step_has_a_short_pause_not_the_5_second_default():
    result = compile_cache(
        _cache(_note(1, "click", locators=[_unique("get_by_text(\"Go\")", "text", 38)]))
    )
    assert result.automation["nodes"][0]["end_sleep_time"] <= 2.0


def test_unsupported_cache_version_is_rejected():
    cache = ActionCache.model_validate(
        {"version": 99, "goal": "g", "start_url": URL, "notes": []}
    )
    with pytest.raises(ValueError, match="version"):
        compile_cache(cache)


def test_missing_start_url_is_an_error():
    cache = ActionCache.model_validate({"goal": "g", "notes": []})
    with pytest.raises(ValueError, match="started"):
        compile_cache(cache)


def test_step_the_llm_judged_as_failure_is_dropped_even_if_it_technically_worked():
    box_a = _unique("locator(\"input[name='a']\")")
    box_b = _unique("locator(\"input[name='b']\")")
    result = compile_cache(
        _cache(
            _note(1, "input", {"text": "v"}, locators=[box_a]),
            _note(
                2,
                "input",
                {"text": "v"},
                locators=[box_b],
                evaluation="Typed into the wrong box. Verdict: Failure",
            ),
        )
    )
    assert _commands(result) == [("input_text", "locator(\"input[name='b']\")")]


def test_uncertain_verdict_does_not_drop_anything():
    box_a = _unique("locator(\"input[name='a']\")")
    box_b = _unique("locator(\"input[name='b']\")")
    result = compile_cache(
        _cache(
            _note(1, "input", {"text": "v"}, locators=[box_a]),
            _note(2, "input", {"text": "w"}, locators=[box_b], evaluation="Verdict: Uncertain"),
        )
    )
    assert len(result.automation["nodes"]) == 2


def test_a_run_where_the_agent_gave_up_is_never_cached():
    """If the agent's own `done` says success=False, we refuse to cache the mess."""
    cache = _cache(
        _note(1, "click", locators=[_unique("get_by_role('button')")]),
        _note(2, "done", {"text": "ran out of steps", "success": False}),
    )
    with pytest.raises(ValueError, match="did NOT finish"):
        compile_cache(cache)
