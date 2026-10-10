"""Tests for the action cache shape (schema.py). No browser, no LLM."""

from pathlib import Path

from optexity.memory.schema import ActionCache, ActionNote, LocatorCandidate

FIXTURES = Path(__file__).parent / "fixtures"


def test_fixture_file_loads():
    cache = ActionCache.model_validate_json(
        (FIXTURES / "roboform_actions.json").read_text()
    )
    assert cache.version == 1
    assert len(cache.notes) == 8


def test_best_locator_skips_non_unique_candidates():
    note = ActionNote(
        step=1,
        action="input",
        locators=[
            LocatorCandidate(locator='locator("#a")', kind="id", score=92, match_count=2, unique=False),
            LocatorCandidate(locator='locator("[name=b]")', kind="name", score=84, match_count=1, unique=True),
        ],
    )
    best = note.best_locator()
    assert best is not None
    assert best.kind == "name"


def test_best_locator_is_none_when_nothing_was_verified():
    note = ActionNote(
        step=1,
        action="click",
        locators=[LocatorCandidate(locator='locator("#a")', kind="id", score=92)],
    )
    assert note.best_locator() is None


def test_round_trip_keeps_the_data():
    cache = ActionCache(goal="g", start_url="https://x.test", notes=[ActionNote(step=1, action="done")])
    again = ActionCache.model_validate_json(cache.model_dump_json())
    assert again.goal == "g"
    assert again.notes[0].action == "done"
