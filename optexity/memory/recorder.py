"""The recorder: watches browser-use work and writes down every action.

What this file is for
---------------------
While an `agentic_task` runs, browser-use clicks and types by element NUMBER
("click [12]"). Numbers are useless later. The recorder listens to the agent
(through the `register_action_callback` hook in the browser-use fork) and, for
every action, writes a note with:

  * what the action was (click / input / ...),
  * an ID card of the real element (tag, id, name, placeholder, xpath...),
  * a ranked list of Playwright locators for it, each one CHECKED on the live
    page: "how many elements does this locator match?" (only 1 is safe),
  * what happened (worked or failed) and the page address before / after.

At the end the notes are saved as `actions.json`, and the compiler turns them
into `cached.json`, a normal Optexity workflow that needs no LLM.

Everything here is "best effort": if recording breaks, we log it and the agentic
run carries on exactly as it would without us.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from optexity.inference.core.interaction.utils import LocatorExtraction
from optexity.memory.compiler import CompileResult, compile_cache
from optexity.memory.schema import (
    ActionCache,
    ActionNote,
    ElementFacts,
    LocatorCandidate,
)

if TYPE_CHECKING:
    from browser_use.agent.action_event import ActionEvent

    from optexity.inference.infra.browser import Browser

logger = logging.getLogger(__name__)

# How long we wait for one "how many elements match?" question to the page.
_COUNT_TIMEOUT_SECONDS = 2.0

# Parameter names that can hold a typed value. Used to hide secrets.
_TEXT_PARAM_KEYS = ("text", "value")


def facts_from_element(element: Any) -> ElementFacts:
    """Turn a browser-use element object into our small, saveable ID card.

    Reads the tag, attributes, accessibility role/name, visible text and xpath.
    Missing pieces are simply left empty. It never raises, because a weird
    element must not stop the recording.
    """
    try:
        ax = getattr(element, "ax_node", None)
        text = None
        getter = getattr(element, "get_meaningful_text_for_llm", None)
        if callable(getter):
            try:
                text = " ".join((getter() or "").split())[:200] or None
            except Exception:
                text = None
        return ElementFacts(
            tag=(getattr(element, "tag_name", None) or None),
            attributes={
                str(k): str(v)
                for k, v in (getattr(element, "attributes", None) or {}).items()
            },
            role=(ax.role if ax and ax.role else None),
            name=(ax.name if ax and ax.name else None),
            text=text,
            xpath=(getattr(element, "xpath", None) or None),
        )
    except Exception as e:
        logger.debug(f"facts_from_element failed: {type(e).__name__}: {e}")
        return ElementFacts()


def is_secret_field(facts: ElementFacts | None) -> bool:
    """Is this element a place where a secret is typed (a password box)?

    Today this means `<input type="password">` or an `autocomplete` hint that
    mentions a password. If true, the recorder never saves the typed value.
    """
    if facts is None:
        return False
    attrs = {k.lower(): v.lower() for k, v in facts.attributes.items()}
    return attrs.get("type") == "password" or "password" in attrs.get("autocomplete", "")


class ActionRecorder:
    """Collects one `ActionNote` per browser-use action, then saves the lot.

    Use it like this:

        recorder = ActionRecorder(browser, goal="Fill the form", start_url=url)
        Agent(..., register_action_callback=recorder.on_action)
        ...run the agent...
        recorder.save(directory)

    It is a class (not a function) because it has to remember all notes between
    callbacks.
    """

    def __init__(self, browser: Browser, goal: str, start_url: str | None = None):
        """Start an empty recording.

        `browser` is Optexity's browser wrapper; we use it to ask the live page
        "how many elements match this locator?". `goal` is the sentence the
        human wrote in the agentic task. `start_url` is the page we began on.
        """
        self.browser = browser
        self.cache = ActionCache(goal=goal, start_url=start_url)
        # The "before" and "after" calls share one event object. We use it as a
        # key to find the note that belongs to it.
        self._notes_by_event: dict[int, ActionNote] = {}

    async def on_action(self, event: ActionEvent) -> None:
        """The callback given to browser-use. Called twice per action.

        "before": the element is still live, so we build the note now.
        "after": we fill in whether it worked. Errors here are logged, never
        raised, so recording can never break the agent.
        """
        try:
            if event.phase == "before":
                await self._start_note(event)
            else:
                self._finish_note(event)
        except Exception as e:
            logger.warning(f"Action recorder skipped an action: {type(e).__name__}: {e}")

    async def _start_note(self, event: ActionEvent) -> None:
        """Create the note for an action that is about to run.

        Describes the element, finds and checks its locators, hides secrets,
        and stores the note. Example: for "click [12]" on a Submit button it
        stores the button's ID card plus e.g. `get_by_role("button",
        name="Submit")` with `match_count=1, unique=True`.
        """
        facts = facts_from_element(event.element) if event.element is not None else None
        locators = (
            await self._checked_locators(event.element)
            if event.element is not None
            else []
        )

        params = dict(event.params)
        sensitive = is_secret_field(facts) and event.action_name == "input"
        if sensitive:
            for key in _TEXT_PARAM_KEYS:
                if key in params:
                    params[key] = ""  # the real secret is never written to disk

        note = ActionNote(
            step=event.step,
            position=event.position,
            actions_in_step=event.actions_in_step,
            action=event.action_name,
            params=params,
            element=facts,
            locators=locators,
            next_goal=event.next_goal,
            evaluation=event.evaluation,
            url_before=event.url_before,
            sensitive=sensitive,
        )
        self.cache.notes.append(note)
        self._notes_by_event[id(event)] = note

    def _finish_note(self, event: ActionEvent) -> None:
        """Fill in the result (worked / failed, new URL) on the matching note."""
        note = self._notes_by_event.pop(id(event), None)
        if note is None:
            return
        note.success = event.success
        note.error = event.error
        note.url_after = event.url_after

    async def _checked_locators(self, element: Any) -> list[LocatorCandidate]:
        """Build locators for the element and check each one on the live page.

        LocatorExtraction ranks candidates by how stable they LOOK, but it never
        checks they point at exactly one element. We do that here by counting
        matches. Example: `locator("#name")` matching 3 elements gets
        `unique=False` and will never be used as a command.
        """
        return await checked_locators(self.browser, element)

    def compile(self) -> CompileResult:
        """Turn the notes recorded so far into a cached workflow (no files written)."""
        return compile_cache(self.cache)

    def save(self, directory: str | Path) -> tuple[Path, Path | None]:
        """Write `actions.json` and `cached.json` into `directory`.

        Returns the two paths. `cached.json` is None when compiling failed (for
        example nothing useful was recorded); `actions.json` is always written
        so the compile can be retried by hand with
        `python -m optexity.memory.compiler`.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        actions_path = directory / "actions.json"
        actions_path.write_text(self.cache.model_dump_json(indent=2))

        cached_path: Path | None = directory / "cached.json"
        try:
            result = self.compile()
            cached_path.write_text(json.dumps(result.automation, indent=2) + "\n")  # type: ignore[union-attr]
            logger.info(
                f"Memory cache saved to {directory} "
                f"({result.cached_steps} cached step(s), {result.agentic_steps} agentic step(s))"
            )
        except Exception as e:
            logger.warning(f"Could not compile the recording: {type(e).__name__}: {e}")
            cached_path = None
        return actions_path, cached_path


async def checked_locators(browser: Browser, element: Any) -> list[LocatorCandidate]:
    """Build ranked locators for a live element and count the matches of each.

    Shared by the recorder (first run) and the write-back repair watcher (a
    repaired step), so both apply the same "must match exactly one element" rule.
    Must be called while the element is still on the page, i.e. before the
    action changes anything. Never raises; on trouble the match count is None.
    """
    raw = LocatorExtraction.locator_candidates(element, method="")
    try:
        page = await browser.get_current_page()
    except Exception:
        page = None

    out: list[LocatorCandidate] = []
    for item in raw:
        locator = item["locator"].removeprefix("page.")
        count = await _count_matches(page, locator) if page is not None else None
        out.append(
            LocatorCandidate(
                locator=locator,
                kind=item["kind"],
                score=item["score"],
                match_count=count,
                unique=(count == 1),
            )
        )
    return out


async def _count_matches(page: Any, locator: str) -> int | None:
    """Ask the page how many elements a locator matches. None if we can't tell.

    `locator` looks like `get_by_label("Email")`. We evaluate it the same way
    Optexity evaluates every `command`: `eval(f"page.{command}")`.
    """
    try:
        found = eval(f"page.{locator}")  # noqa: S307 - same as Optexity's command runner
        return await asyncio.wait_for(found.count(), timeout=_COUNT_TIMEOUT_SECONDS)
    except Exception as e:
        logger.debug(f"Could not count matches for {locator}: {type(e).__name__}: {e}")
        return None
