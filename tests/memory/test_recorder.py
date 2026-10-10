"""Tests for the recorder. No real browser and no LLM: the page is a tiny fake.

The fake page answers "how many elements match this locator?" so we can check
the recorder keeps only locators that match exactly one element.
"""

import asyncio
import json
from types import SimpleNamespace

from optexity.memory.recorder import ActionRecorder, facts_from_element, is_secret_field
from optexity.memory.schema import ActionCache


class _FakeLocator:
    """Answers `count()` with a number we chose."""

    def __init__(self, count: int):
        self._count = count

    async def count(self) -> int:
        return self._count


class _FakePage:
    """`#name` matches 2 elements; every other locator matches exactly 1."""

    url = "https://example.com/form"

    def locator(self, selector, **_kwargs):
        return _FakeLocator(2 if "#name" in selector else 1)

    def __getattr__(self, _name):  # get_by_role, get_by_label, ...
        return lambda *a, **k: _FakeLocator(1)


class _FakeBrowser:
    async def get_current_page(self):
        return _FakePage()


def _element(**attrs):
    return SimpleNamespace(
        tag_name="input",
        attributes=attrs,
        ax_node=SimpleNamespace(role="textbox", name="Full name"),
        xpath="/html/body/form/input[1]",
        get_meaningful_text_for_llm=lambda: "",
    )


def _event(phase="before", element=None, name="input", params=None, **extra):
    base = dict(
        phase=phase,
        step=2,
        position=0,
        actions_in_step=1,
        action_name=name,
        params=params or {"index": 12, "text": "xyz"},
        element=element,
        next_goal="Fill the name",
        evaluation="Verdict: Success",
        url_before="https://example.com/form",
        url_after=None,
        success=None,
        error=None,
    )
    base.update(extra)
    return SimpleNamespace(**base)


def test_facts_keep_stable_things_only():
    facts = facts_from_element(_element(id="name", placeholder="Full name"))
    assert facts.tag == "input"
    assert facts.attributes == {"id": "name", "placeholder": "Full name"}
    assert facts.role == "textbox"
    assert facts.xpath == "/html/body/form/input[1]"


def test_password_boxes_are_detected():
    assert is_secret_field(facts_from_element(_element(type="password")))
    assert not is_secret_field(facts_from_element(_element(type="text")))
    assert not is_secret_field(None)


def test_note_has_checked_locators_and_result():
    recorder = ActionRecorder(_FakeBrowser(), goal="Fill the form", start_url="https://example.com/form")
    event = _event(element=_element(id="name", placeholder="Full name"))

    asyncio.run(recorder.on_action(event))
    event.phase, event.success, event.url_after = "after", True, "https://example.com/form"
    asyncio.run(recorder.on_action(event))

    note = recorder.cache.notes[0]
    assert note.action == "input" and note.success is True
    assert note.next_goal == "Fill the name" and note.evaluation == "Verdict: Success"
    by_kind = {c.kind: c for c in note.locators}
    assert by_kind["id"].match_count == 2 and not by_kind["id"].unique  # not safe
    assert by_kind["placeholder"].unique
    # best_locator skips the non-unique id and picks a unique one.
    assert note.best_locator().kind != "id"


def test_typed_secret_is_never_stored():
    recorder = ActionRecorder(_FakeBrowser(), goal="Log in", start_url="https://example.com/login")
    event = _event(element=_element(type="password", name="pw"), params={"index": 5, "text": "hunter2"})
    asyncio.run(recorder.on_action(event))

    note = recorder.cache.notes[0]
    assert note.sensitive is True
    assert note.params["text"] == ""
    assert "hunter2" not in recorder.cache.model_dump_json()


def test_actions_without_an_element_still_get_a_note():
    recorder = ActionRecorder(_FakeBrowser(), goal="Go", start_url="https://example.com")
    asyncio.run(recorder.on_action(_event(name="navigate", params={"url": "https://a.test"})))
    note = recorder.cache.notes[0]
    assert note.element is None and note.locators == []


def test_save_writes_both_files_and_cached_is_valid(tmp_path):
    recorder = ActionRecorder(_FakeBrowser(), goal="Fill the form", start_url="https://example.com/form")
    event = _event(element=_element(id="name", placeholder="Full name"))
    asyncio.run(recorder.on_action(event))
    event.phase, event.success = "after", True
    asyncio.run(recorder.on_action(event))

    actions_path, cached_path = recorder.save(tmp_path / "run")

    assert ActionCache.model_validate_json(actions_path.read_text()).goal == "Fill the form"
    cached = json.loads(cached_path.read_text())
    node = cached["nodes"][0]["interaction_action"]["input_text"]
    # The id locator matched 2 elements, so the next best unique one (role+name) wins.
    assert node["command"] == 'get_by_role("textbox", name="Full name")'
    assert node["input_text"] == "xyz"


def test_save_still_writes_actions_when_compile_fails(tmp_path):
    recorder = ActionRecorder(_FakeBrowser(), goal="Nothing useful", start_url=None)
    actions_path, cached_path = recorder.save(tmp_path)
    assert actions_path.exists()
    assert cached_path is None
