"""Tests for self-healing (write-back). No browser and no LLM.

We build a tiny workflow file, break one command on purpose, pretend the
fallback fixed it, and check what ends up in the file.
"""

import asyncio
import json
from types import SimpleNamespace

from optexity.memory.writeback import (
    RepairWatcher,
    apply_repair,
    command_of_action_node,
    repairs_log_path,
)

OLD = 'get_by_label("Old label")'
NEW = "locator(\"input[name='email']\")"


def _workflow(tmp_path):
    """A valid two-step workflow file; step 1 is the one we will repair."""
    data = {
        "url": "https://example.com",
        "parameters": {"input_parameters": {}, "generated_parameters": {}},
        "nodes": [
            {
                "type": "action_node",
                "interaction_action": {"click_element": {"command": 'get_by_role("button")'}},
            },
            {
                "type": "action_node",
                "interaction_action": {"input_text": {"command": OLD, "input_text": "hi"}},
            },
        ],
    }
    path = tmp_path / "cached.json"
    path.write_text(json.dumps(data))
    return path


def test_apply_repair_replaces_only_that_command_and_logs_the_old_one(tmp_path):
    path = _workflow(tmp_path)

    assert apply_repair(path, 1, OLD, NEW, kind="name") is True

    data = json.loads(path.read_text())
    assert data["nodes"][1]["interaction_action"]["input_text"]["command"] == NEW
    assert data["nodes"][0]["interaction_action"]["click_element"]["command"] == 'get_by_role("button")'
    log = [json.loads(line) for line in repairs_log_path(path).read_text().splitlines()]
    assert log[0]["old_command"] == OLD and log[0]["new_command"] == NEW
    assert not list(tmp_path.glob("*.tmp"))  # the temp file was cleaned up by the rename


def test_apply_repair_refuses_when_the_node_holds_a_different_command(tmp_path):
    path = _workflow(tmp_path)
    before = path.read_text()
    assert apply_repair(path, 1, "some other command", NEW) is False
    assert apply_repair(path, 99, OLD, NEW) is False  # no such node
    assert path.read_text() == before


def test_apply_repair_refuses_to_write_an_invalid_workflow(tmp_path):
    path = _workflow(tmp_path)
    before = path.read_text()
    # A command that is not a string makes the workflow invalid.
    assert apply_repair(path, 1, OLD, 12345) is False  # type: ignore[arg-type]
    assert path.read_text() == before


# --- the watcher, with fake pieces -----------------------------------------


class _FakeAgent:
    register_action_callback = None


class _FakePage:
    def locator(self, selector, **_):
        return SimpleNamespace(count=_count(1))

    def __getattr__(self, _name):
        return lambda *a, **k: SimpleNamespace(count=_count(3))  # not unique


def _count(n):
    async def count():
        return n

    return count


class _FakeBrowser:
    def __init__(self):
        self.backend_agent = _FakeAgent()

    async def get_current_page(self):
        return _FakePage()


def _event(phase, success=None):
    element = SimpleNamespace(
        tag_name="input",
        attributes={"name": "email"},
        ax_node=SimpleNamespace(role="textbox", name="Email"),
        xpath="/html/body/input",
        get_meaningful_text_for_llm=lambda: "",
    )
    return SimpleNamespace(
        phase=phase, action_name="input", element=element if phase == "before" else None, success=success
    )


def test_watcher_saves_the_unique_locator_after_a_successful_fallback(tmp_path):
    path = _workflow(tmp_path)
    browser = _FakeBrowser()
    watcher = RepairWatcher(browser)
    watcher.start()
    assert browser.backend_agent.register_action_callback is not None

    asyncio.run(watcher.on_action(_event("before")))
    asyncio.run(watcher.on_action(_event("after", success=True)))

    assert watcher.finish(path, 1, OLD) is True
    new = json.loads(path.read_text())["nodes"][1]["interaction_action"]["input_text"]["command"]
    assert new == NEW  # the one locator the fake page said matches exactly once
    watcher.stop()
    assert browser.backend_agent.register_action_callback is None


def test_watcher_does_nothing_if_the_fallback_action_failed(tmp_path):
    path = _workflow(tmp_path)
    before = path.read_text()
    watcher = RepairWatcher(_FakeBrowser())
    asyncio.run(watcher.on_action(_event("before")))
    asyncio.run(watcher.on_action(_event("after", success=False)))
    assert watcher.finish(path, 1, OLD) is False
    assert path.read_text() == before


def test_watcher_does_nothing_if_no_fallback_happened(tmp_path):
    path = _workflow(tmp_path)
    before = path.read_text()
    assert RepairWatcher(_FakeBrowser()).finish(path, 1, OLD) is False
    assert path.read_text() == before


def test_command_of_action_node():
    click = SimpleNamespace(command='get_by_role("button")', skip_command=False)
    node = SimpleNamespace(interaction_action=SimpleNamespace(click_element=click))
    assert command_of_action_node(node) == 'get_by_role("button")'

    click.skip_command = True
    assert command_of_action_node(node) is None
    assert command_of_action_node(SimpleNamespace(interaction_action=None)) is None
