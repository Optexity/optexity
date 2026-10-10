"""Turns the raw action notes into a normal Optexity workflow.

What this file is for
---------------------
After an `agentic_task` finishes, we have a list of notes ("browser-use typed xyz
into the Address 1 box, then clicked Submit"). The notes are messy: browser-use
makes wrong guesses, scrolls around, and says "done" at the end.

This file cleans the notes up and writes a normal Optexity workflow where every
step has an exact Playwright `command`. When Optexity runs that workflow, it uses
Playwright directly and does not need the LLM.

Where it fits
-------------
    recorder.py --> actions.json --> [ compiler.py ] --> cached.json --> normal Optexity run

How it works, in plain steps
----------------------------
1. Throw away notes that are not real work: failed actions, "done", scrolls, waits.
2. Throw away actions that browser-use itself later judged as a mistake
   ("Verdict: Failure"), and typing that was later typed over in the same box.
3. For each remaining note, pick the best locator that was CHECKED to match exactly
   one element on the live page.
4. Translate the browser-use action into the matching Optexity action:
       input -> input_text, click -> click_element, select_dropdown -> select_option,
       navigate -> go_to_url, go_back -> go_back, send_keys(Enter/Tab) -> key_press
5. If a note has no trustworthy locator (or is an action we cannot cache safely),
   keep that one step as a small `agentic_task` instead of guessing. The workflow
   still works; only that step stays slow.

This file never opens a browser and never calls an LLM, so it is fast, free, and
easy to test with plain Python.

Command line use (re-compile an old cache without running the agent again):
    python -m optexity.memory.compiler cache/roboform.actions.json cache/roboform.cached.json
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from optexity.memory.schema import CACHE_FORMAT_VERSION, ActionCache, ActionNote
from optexity.schema.automation import Automation

logger = logging.getLogger(__name__)

# Actions that are never part of the final workflow: they either do not change
# the page in a way we need to repeat, or they just report information.
_IGNORED_ACTIONS = {
    "done",  # "I am finished": not an action on the page
    "wait",  # Playwright already waits for elements by itself
    "scroll",  # Playwright scrolls an element into view before using it
    "screenshot",
    "find_text",
    "dropdown_options",  # only reads the options, changes nothing
    "read_file",
    "write_file",
    "replace_file",
}

# Pause (seconds) after each kind of step. Optexity's default is 5 seconds after
# EVERY step, which would throw away the speed we are trying to win.
_PAUSE_AFTER_INPUT = 0.2
_PAUSE_AFTER_CLICK = 1.0
_PAUSE_AFTER_NAVIGATION = 2.0

# How many browser-use steps a flagged (not cacheable) step may use.
_FALLBACK_AGENTIC_MAX_STEPS = 5


@dataclass
class CompileResult:
    """What `compile_cache` gives back."""

    automation: dict[str, Any]  # the workflow, ready to save as JSON
    cached_steps: int = 0  # steps now using an exact Playwright command
    agentic_steps: int = 0  # steps that had to stay as an agentic_task
    report: list[str] = field(default_factory=list)  # plain-English log of choices


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _element_label(note: ActionNote) -> str:
    """A short human-readable name for the element, used in text we write.

    Tries, in order: the accessible name, aria-label, placeholder, name, id,
    then the visible text. Returns "element" if nothing is known.

    Example: an input with placeholder "Address 1" -> "Address 1".
    """
    if note.element is None:
        return "element"
    attrs = note.element.attributes
    for candidate in (
        note.element.name,
        attrs.get("aria-label"),
        attrs.get("placeholder"),
        attrs.get("name"),
        attrs.get("id"),
        note.element.text,
    ):
        if candidate and candidate.strip():
            return candidate.strip()
    return "element"


def _prompt_for(note: ActionNote, fallback_text: str) -> str:
    """The plain-English sentence kept as the backup plan for a step.

    If the website changes and the exact `command` stops working, Optexity uses
    this sentence to ask the LLM which element to use.

    We reuse the LLM's own words (`next_goal`) when that step had exactly one
    action, because then the sentence describes this action alone. Otherwise we
    write our own sentence (`fallback_text`).
    """
    if note.actions_in_step == 1 and note.next_goal and note.next_goal.strip():
        return note.next_goal.strip()
    return fallback_text


def _field_key(note: ActionNote) -> str | None:
    """A name that identifies "which element" a note acted on.

    Used to spot two notes that touched the same box. Prefers the verified
    locator, falls back to the xpath, returns None if neither is known.
    """
    best = note.best_locator()
    if best is not None:
        return best.locator
    if note.element is not None and note.element.xpath:
        return f"xpath:{note.element.xpath}"
    return None


def _same_url(a: str | None, b: str | None) -> bool:
    """True when two URLs are the same page (ignores a trailing slash)."""
    return (a or "").rstrip("/") == (b or "").rstrip("/")


def _agentic_node(note: ActionNote, why: str) -> dict[str, Any]:
    """Build a small `agentic_task` node for a step we cannot cache safely.

    We prefer a slow-but-correct step over a guessed locator. `why` is only used
    in the report (and is easy to read in the demo).
    """
    if note.next_goal and note.next_goal.strip():
        task = note.next_goal.strip()
    elif note.sensitive:
        task = f"Perform the '{note.action}' action (value hidden because it is secret)."
    else:
        task = f"Perform the '{note.action}' action with {note.params}."
    logger.debug(f"Keeping step {note.step}.{note.position} as agentic_task: {why}")
    return {
        "type": "action_node",
        "interaction_action": {
            "agentic_task": {
                "task": task,
                "max_steps": _FALLBACK_AGENTIC_MAX_STEPS,
                "backend": "browser_use",
            }
        },
    }


# ---------------------------------------------------------------------------
# Step 1 and 2: clean the notes
# ---------------------------------------------------------------------------


def _keep_real_work(notes: list[ActionNote], report: list[str]) -> list[ActionNote]:
    """Drop notes that should never be replayed (failed, unfinished, ignorable).

    Example: [scroll, input(ok), click(failed), done] -> [input(ok)]
    """
    kept: list[ActionNote] = []
    for note in notes:
        where = f"step {note.step}.{note.position} ({note.action})"
        if note.action in _IGNORED_ACTIONS:
            report.append(f"dropped {where}: not needed for replay")
        elif note.success is None:
            report.append(f"dropped {where}: never saw it finish")
        elif note.success is False:
            report.append(f"dropped {where}: it failed ({note.error})")
        else:
            kept.append(note)
    return kept


def _drop_steps_judged_as_failures(
    notes: list[ActionNote], report: list[str]
) -> list[ActionNote]:
    """Drop actions whose OWN step was later called a failure by browser-use.

    After every step, browser-use's LLM looks at the page and writes a verdict on
    the previous step, for example "Typed into the Title box, not Full Name.
    Verdict: Failure". An action can "work" technically (the typing happened) and
    still be wrong. That verdict is the clearest sign we have, so we trust it.

    How it works: each note stores that judgement in `evaluation`, taken from the
    step AFTER the action. So if any note of step N+1 says "Verdict: Failure",
    every action of step N is dropped.

    Only a clear "Failure" counts. "Success" and "Uncertain" keep the actions.
    """
    steps = sorted({n.step for n in notes})
    failed_steps: set[int] = set()
    for previous, current in zip(steps, steps[1:]):
        for note in notes:
            if note.step == current and note.evaluation:
                if "verdict: failure" in note.evaluation.lower():
                    failed_steps.add(previous)
                    break

    kept: list[ActionNote] = []
    for note in notes:
        if note.step in failed_steps:
            report.append(
                f"dropped step {note.step}.{note.position} ({note.action}): "
                "the next step's verdict called it a failure"
            )
        else:
            kept.append(note)
    return kept


def _drop_overwritten_inputs(
    notes: list[ActionNote], report: list[str]
) -> list[ActionNote]:
    """Drop typing into a box when the same box is typed into again right after.

    browser-use sometimes types into the wrong place, or types, then corrects
    itself. Only the LAST value in a box matters, so earlier ones are noise.

    We are careful: we only drop an earlier typing if everything between the two
    typings is also just typing (into other boxes). If there was a click or page
    change in between, the earlier typing may have mattered (for example, typing
    a search and pressing Search, then typing again), so we keep it.
    """
    keep = [True] * len(notes)
    for i, note in enumerate(notes):
        if note.action != "input":
            continue
        key = _field_key(note)
        if key is None:
            continue
        for j in range(i + 1, len(notes)):
            later = notes[j]
            if later.action != "input":
                break  # a click, navigation... might have used the earlier typing
            if _field_key(later) == key:
                keep[i] = False
                report.append(
                    f"dropped step {note.step}.{note.position} (input): "
                    f"typed over again in step {later.step}.{later.position}"
                )
                break
    return [n for n, k in zip(notes, keep) if k]


# ---------------------------------------------------------------------------
# Step 3 and 4: translate one note into one workflow node
# ---------------------------------------------------------------------------


class _Translator:
    """Turns notes into workflow nodes, one at a time.

    It is a small class (not a function) because it remembers a little between
    notes: the start URL, a counter for secret placeholders, and the counts.
    """

    def __init__(self, start_url: str | None, report: list[str]):
        self.start_url = start_url
        self.report = report
        self.input_parameters: dict[str, list[str]] = {}
        self.cached_steps = 0
        self.agentic_steps = 0
        self._secret_counter = 0
        self._emitted_any_node = False

    # -- public ----------------------------------------------------------

    def translate(self, note: ActionNote) -> dict[str, Any] | None:
        """Return the workflow node for one note, or None to skip it."""
        handler = {
            "input": self._input,
            "click": self._click,
            "select_dropdown": self._select,
            "navigate": self._navigate,
            "go_back": self._go_back,
            "send_keys": self._send_keys,
        }.get(note.action)

        if handler is None:
            return self._flag(note, f"action '{note.action}' cannot be cached safely")

        node = handler(note)
        if node is not None:
            self._emitted_any_node = True
        return node

    # -- helpers ---------------------------------------------------------

    def _flag(self, note: ActionNote, why: str) -> dict[str, Any]:
        """Keep a step as an agentic_task, and remember we did so."""
        self.agentic_steps += 1
        self._emitted_any_node = True
        self.report.append(
            f"kept step {note.step}.{note.position} ({note.action}) as agentic_task: {why}"
        )
        return _agentic_node(note, why)

    def _cached(self, node_body: dict[str, Any], pause: float) -> dict[str, Any]:
        """Wrap an interaction body into a full workflow node."""
        self.cached_steps += 1
        return {
            "type": "action_node",
            "interaction_action": node_body,
            "end_sleep_time": pause,
        }

    def _command_for(self, note: ActionNote) -> str | None:
        """The verified-unique Playwright command for the note, or None."""
        best = note.best_locator()
        return best.locator if best else None

    # -- one handler per browser-use action -------------------------------

    def _input(self, note: ActionNote) -> dict[str, Any]:
        command = self._command_for(note)
        if command is None:
            return self._flag(note, "no locator was verified to match exactly one element")

        if note.sensitive:
            # Never write a secret into a file. Use a named placeholder that the
            # caller must provide as an input parameter when running the workflow.
            self._secret_counter += 1
            name = f"secret_{self._secret_counter}"
            self.input_parameters[name] = ["REPLACE_ME"]
            text = f"{{{name}[0]}}"
        else:
            text = str(note.params.get("text", ""))

        label = _element_label(note)
        prompt = _prompt_for(note, f"Type the value into the '{label}' field.")
        return self._cached(
            {
                "input_text": {
                    "command": command,
                    "prompt_instructions": prompt,
                    "input_text": text,
                }
            },
            _PAUSE_AFTER_INPUT,
        )

    def _click(self, note: ActionNote) -> dict[str, Any]:
        command = self._command_for(note)
        if command is None:
            return self._flag(note, "no locator was verified to match exactly one element")

        label = _element_label(note)
        prompt = _prompt_for(note, f"Click the '{label}' element.")
        return self._cached(
            {"click_element": {"command": command, "prompt_instructions": prompt}},
            _PAUSE_AFTER_CLICK,
        )

    def _select(self, note: ActionNote) -> dict[str, Any]:
        command = self._command_for(note)
        option = note.params.get("text")
        if command is None or not option:
            return self._flag(note, "no verified locator or no option text")

        label = _element_label(note)
        prompt = _prompt_for(note, f"Choose '{option}' in the '{label}' dropdown.")
        return self._cached(
            {
                "select_option": {
                    "command": command,
                    "prompt_instructions": prompt,
                    "select_values": [str(option)],
                }
            },
            _PAUSE_AFTER_CLICK,
        )

    def _navigate(self, note: ActionNote) -> dict[str, Any] | None:
        url = note.params.get("url")
        if not url:
            return self._flag(note, "navigate had no url")
        # The workflow already opens its start URL first. Repeating it as the
        # very first step would just reload the same page.
        if not self._emitted_any_node and _same_url(url, self.start_url):
            self.report.append(
                f"dropped step {note.step}.{note.position} (navigate): "
                "workflow already starts on this url"
            )
            return None
        return self._cached(
            {
                "go_to_url": {
                    "url": str(url),
                    "new_tab": bool(note.params.get("new_tab", False)),
                }
            },
            _PAUSE_AFTER_NAVIGATION,
        )

    def _go_back(self, note: ActionNote) -> dict[str, Any]:
        return self._cached({"go_back": {}}, _PAUSE_AFTER_NAVIGATION)

    def _send_keys(self, note: ActionNote) -> dict[str, Any]:
        keys = str(note.params.get("keys", "")).strip()
        # Optexity's key_press currently handles only a few keys. Anything else
        # stays as an agentic step rather than being silently ignored.
        # (Optexity's allowed key names are lower-case, e.g. "enter".)
        if keys.lower() in ("enter", "tab"):
            return self._cached({"key_press": {"type": keys.lower()}}, _PAUSE_AFTER_CLICK)
        return self._flag(note, f"key '{keys}' is not supported by key_press")


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def compile_cache(cache: ActionCache) -> CompileResult:
    """Turn an action cache into a validated Optexity workflow (as a dict).

    Input : the notes from one agentic run.
    Output: a CompileResult whose `.automation` is a workflow dict that follows
            Optexity's own `Automation` schema (we check this before returning).

    Raises:
        ValueError: if the cache file is from an unsupported format version,
                    if the agent's own final `done` says it failed, or
                    if the workflow we built does not pass Optexity's validation.
    """
    if cache.version != CACHE_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported action cache version {cache.version} "
            f"(this code reads version {CACHE_FORMAT_VERSION})."
        )

    # A run where the agent itself said "I failed" is not a recipe worth keeping:
    # replaying it would just repeat the mess (wrong clicks, retries, dead ends).
    for note in cache.notes:
        if note.action == "done" and note.params.get("success") is False:
            raise ValueError(
                "Refusing to cache: the agent reported that it did NOT finish the task "
                f"({note.params.get('text', 'no reason given')!r})."
            )

    report: list[str] = []
    notes = _keep_real_work(cache.notes, report)
    notes = _drop_steps_judged_as_failures(notes, report)
    notes = _drop_overwritten_inputs(notes, report)

    start_url = cache.start_url
    if not start_url:
        # Use the page the very first note started on.
        start_url = next((n.url_before for n in cache.notes if n.url_before), None)
    if not start_url:
        raise ValueError("Cannot compile: the cache does not say which page the run started on.")

    translator = _Translator(start_url, report)
    nodes = [node for note in notes if (node := translator.translate(note)) is not None]

    automation = {
        "url": start_url,
        "automation_description": f"Compiled from agentic task: {cache.goal}",
        "parameters": {
            "input_parameters": translator.input_parameters,
            "generated_parameters": {},
        },
        "nodes": nodes,
    }

    # Fail loudly if what we built is not a valid Optexity workflow.
    try:
        Automation.model_validate(automation)
    except Exception as e:
        raise ValueError(f"Compiled workflow is not valid: {e}") from e

    return CompileResult(
        automation=automation,
        cached_steps=translator.cached_steps,
        agentic_steps=translator.agentic_steps,
        report=report,
    )


def compile_file(actions_path: str | Path, output_path: str | Path) -> CompileResult:
    """Read an `actions.json`, compile it, and write the workflow JSON.

    Example:
        compile_file("cache/roboform.actions.json", "cache/roboform.cached.json")
    """
    cache = ActionCache.model_validate_json(Path(actions_path).read_text())
    result = compile_cache(cache)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.automation, indent=2) + "\n")
    return result


def main() -> None:
    """Command-line entry: compile an existing actions.json into a workflow."""
    parser = argparse.ArgumentParser(
        description="Compile an Optexity action cache into a workflow JSON."
    )
    parser.add_argument("actions", help="path to the actions.json written by a run")
    parser.add_argument("output", help="where to write the compiled workflow JSON")
    args = parser.parse_args()

    result = compile_file(args.actions, args.output)
    print(
        f"cached steps: {result.cached_steps}, "
        f"agentic steps kept: {result.agentic_steps}"
    )
    for line in result.report:
        print(f"  - {line}")


if __name__ == "__main__":
    main()
