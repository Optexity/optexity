"""Self-healing: save a repaired locator back into the workflow file.

What this file is for
---------------------
A cached workflow step has an exact Playwright `command`. If the website
changes, that command stops matching. Optexity then falls back to asking the
LLM to find the element ("prompt fallback"), and that works, but it costs one
LLM call on EVERY future run because the broken command is still in the file.

Write-back fixes it once: when the fallback worked, we take the element it
acted on, build a new locator for it, check that locator matches exactly one
element, and replace the broken `command` in the file. The next run is back to
zero LLM calls.

Safety rules (decision D5)
--------------------------
We only write when ALL of these are true:
  1. the step really used the fallback (browser-use acted on a real element),
  2. that action itself worked,
  3. the whole step finished without raising an error,
  4. the new locator matched exactly ONE element on the live page,
  5. the file's node at that position still holds the old command we expect,
  6. the patched file is still a valid Optexity workflow.
The old command is never lost: every repair is appended to a small
`<file>.repairs.jsonl` log next to the workflow.

Where it fits
-------------
    run_action_node() -> RepairWatcher.start() ... step runs ... RepairWatcher.finish()
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from optexity.memory.recorder import checked_locators
from optexity.memory.schema import LocatorCandidate

if TYPE_CHECKING:
    from browser_use.agent.action_event import ActionEvent

    from optexity.inference.infra.browser import Browser

logger = logging.getLogger(__name__)

# browser-use actions that act on one element and map onto a node with a `command`.
_ELEMENT_ACTIONS = {"click", "input", "select_dropdown"}


def repairs_log_path(workflow_path: str | Path) -> Path:
    """Where the repair history for a workflow file is kept.

    Example: `cached.json` -> `cached.repairs.jsonl` (same folder).
    """
    path = Path(workflow_path)
    return path.with_name(path.stem + ".repairs.jsonl")


def apply_repair(
    workflow_path: str | Path,
    node_index: int,
    old_command: str,
    new_command: str,
    kind: str = "",
) -> bool:
    """Replace one broken `command` in a workflow file. True if the file changed.

    Example: node 3 holds `get_by_label("Email")` which broke. We call
    `apply_repair(file, 3, 'get_by_label("Email")', 'locator("#email")')`.

    It refuses (returns False, file untouched) when:
      * the file cannot be read,
      * node `node_index` does not exist or does not hold `old_command`
        (so we never patch the wrong step),
      * the patched workflow would not pass Optexity's own validation.
    The write is atomic: we write a temp file and rename it over the original,
    so a crash can never leave a half-written workflow.
    """
    from optexity.schema.automation import Automation  # local: keep import light

    path = Path(workflow_path)
    try:
        data = json.loads(path.read_text())
        node = data["nodes"][node_index]
    except Exception as e:
        logger.warning(f"Write-back skipped: cannot read node {node_index} of {path}: {e}")
        return False

    body = (node.get("interaction_action") or {}) if isinstance(node, dict) else {}
    target = next(
        (
            action
            for action in body.values()
            if isinstance(action, dict) and action.get("command") == old_command
        ),
        None,
    )
    if target is None:
        logger.warning(
            f"Write-back skipped: node {node_index} in {path} does not hold the expected command."
        )
        return False

    target["command"] = new_command
    try:
        Automation.model_validate(data)
    except Exception as e:
        logger.warning(f"Write-back skipped: patched workflow is not valid: {e}")
        return False

    # Atomic write: temp file in the same folder, then rename over the original.
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as tmp:
            tmp.write(json.dumps(data, indent=2) + "\n")
        os.replace(tmp_name, path)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise

    with open(repairs_log_path(path), "a") as log:
        log.write(
            json.dumps(
                {
                    "time": datetime.now(timezone.utc).isoformat(),
                    "node_index": node_index,
                    "old_command": old_command,
                    "new_command": new_command,
                    "kind": kind,
                }
            )
            + "\n"
        )
    return True


class RepairWatcher:
    """Watches ONE workflow step and, if the fallback repaired it, saves the fix.

    Use it around a single node:

        watcher = RepairWatcher(browser)
        watcher.start()
        ...run the node...          # raises if the step failed
        watcher.finish(path, node_index, old_command)
        # always: watcher.stop()    (or use try/finally)

    How it knows a fallback happened: the command path never uses browser-use,
    so if browser-use acts on an element during this node, the command must
    have failed and the fallback took over.
    """

    def __init__(self, browser: Browser):
        """`browser` is Optexity's browser wrapper (it owns the fallback agent)."""
        self.browser = browser
        self._locators: list[LocatorCandidate] = []
        self._action_ok: bool | None = None

    def start(self) -> None:
        """Begin listening to the fallback agent (if there is one)."""
        self._locators, self._action_ok = [], None
        agent = getattr(self.browser, "backend_agent", None)
        if agent is not None:
            agent.register_action_callback = self.on_action

    def stop(self) -> None:
        """Stop listening. Safe to call even if `start` never ran."""
        agent = getattr(self.browser, "backend_agent", None)
        if agent is not None and agent.register_action_callback == self.on_action:
            agent.register_action_callback = None

    async def on_action(self, event: ActionEvent) -> None:
        """Callback for the fallback agent's actions.

        "before": remember checked locators for the element it is about to use.
        "after": remember whether the action worked. Never raises.
        """
        try:
            if event.action_name not in _ELEMENT_ACTIONS:
                return
            if event.phase == "before" and event.element is not None:
                self._locators = await checked_locators(self.browser, event.element)
                self._action_ok = None
            elif event.phase == "after":
                self._action_ok = bool(event.success)
        except Exception as e:
            logger.debug(f"Repair watcher ignored an action: {type(e).__name__}: {e}")

    def repaired_command(self) -> tuple[str, str] | None:
        """The (new_command, kind) the fallback proved works, or None.

        None unless the fallback acted on an element, that action worked, and
        at least one locator for it matched exactly one element on the page.
        """
        if not self._locators or self._action_ok is not True:
            return None
        best = next((c for c in self._locators if c.unique), None)
        return (best.locator, best.kind) if best else None

    def finish(
        self, workflow_path: str | Path, node_index: int, old_command: str | None
    ) -> bool:
        """Save the repair if there is one. Call ONLY after the step succeeded.

        Returns True if the workflow file was changed. Never raises: a failed
        write-back must not fail a task that otherwise worked.
        """
        if not old_command:
            return False
        repair = self.repaired_command()
        if repair is None:
            return False
        new_command, kind = repair
        if new_command == old_command:
            return False
        try:
            changed = apply_repair(workflow_path, node_index, old_command, new_command, kind)
        except Exception as e:
            logger.warning(f"Write-back failed: {type(e).__name__}: {e}")
            return False
        if changed:
            logger.info(
                f"Self-healed step {node_index}: {old_command!r} -> {new_command!r} ({kind})"
            )
        return changed


def command_of_action_node(action_node: Any) -> str | None:
    """Return the `command` of an interaction node, or None if it has none.

    Example: a node whose `click_element.command` is `get_by_role("button")`
    returns that string. Nodes that skip their command return None, because
    there is nothing to repair.
    """
    interaction = getattr(action_node, "interaction_action", None)
    if interaction is None:
        return None
    for name in ("click_element", "input_text", "select_option", "check", "uncheck", "hover"):
        action = getattr(interaction, name, None)
        if action is not None and getattr(action, "command", None) and not action.skip_command:
            return action.command
    return None
