"""The shape of the "action cache" file.

What this file is for
---------------------
When browser-use runs an `agentic_task`, we write down one short note for every
action it takes ("typed xyz into the Address 1 box"). All the notes from one run
are saved together in one JSON file. This file defines exactly what that JSON
looks like, using Pydantic models.

Pydantic is a library that checks data: if a file does not match the shape
below, loading it raises a clear error instead of failing later in a confusing
way.

Where it fits
-------------
    recorder.py  --writes-->  ActionCache (actions.json)  --read by-->  compiler.py

Nothing in this file talks to a browser or an LLM. It is only data.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

# Bump this number if the shape of the file ever changes in a way that old files
# can no longer be read. The compiler refuses files with a different number.
CACHE_FORMAT_VERSION = 1


class ElementFacts(BaseModel):
    """What we know about the page element browser-use acted on.

    Think of it as a short ID card for the element. We keep only things that
    survive a page reload (tag, attributes, label, position in the page).
    We deliberately do NOT keep browser-use's temporary element number or
    Chrome's internal node id, because those change on every run.
    """

    tag: str | None = None  # e.g. "input", "button", "a"
    attributes: dict[str, str] = Field(default_factory=dict)  # id, name, placeholder...
    role: str | None = None  # what the element is, e.g. "textbox", "button", "link"
    name: str | None = None  # the label a screen reader would read out
    text: str | None = None  # visible text inside the element (shortened)
    xpath: str | None = None  # a "street address" of the element in the page tree


class LocatorCandidate(BaseModel):
    """One possible Playwright address for an element, plus our check on it.

    `locator` is written the way an Optexity `command` is written, for example
    `locator("input[name='address1']")`. It is NOT prefixed with `page.`.
    """

    locator: str
    kind: str  # why we trust it: "id", "name", "aria-label", "role+name", "xpath"...
    score: int  # higher = more stable. Comes from LocatorExtraction.
    # How many elements on the live page matched this locator when we checked.
    # None means "we could not check" (for example the page was not reachable).
    match_count: int | None = None
    # True only when we checked and exactly ONE element matched. Only unique
    # locators are safe to use as a `command`.
    unique: bool = False


class ActionNote(BaseModel):
    """One thing browser-use did, written down.

    One note = one action (for example one "input" or one "click").
    """

    step: int  # browser-use step number the action belongs to
    position: int = 0  # place of the action inside that step (0 = first)
    actions_in_step: int = 1  # how many actions the LLM chose in this step

    action: str  # browser-use's name for it: "input", "click", "navigate", "done"...
    params: dict[str, Any] = Field(default_factory=dict)  # e.g. {"text": "xyz"}

    element: ElementFacts | None = None
    # Best first. Already checked against the live page by the recorder.
    locators: list[LocatorCandidate] = Field(default_factory=list)

    next_goal: str | None = None  # what the LLM said it was trying to do this step
    # What the LLM said about the PREVIOUS step when it started this one, e.g.
    # "Typed into the wrong box. Verdict: Failure". The compiler uses it to spot
    # mistakes that technically "worked" (the typing happened) but were wrong.
    evaluation: str | None = None

    url_before: str | None = None
    url_after: str | None = None

    # None  = we never saw the action finish
    # True  = it worked
    # False = it failed (see `error`)
    success: bool | None = None
    error: str | None = None

    # True when the typed value is secret (for example a password field). The
    # real value is NOT stored in `params`; a placeholder is stored instead.
    sensitive: bool = False

    def best_locator(self) -> LocatorCandidate | None:
        """Return the best locator we are allowed to trust, or None.

        "Allowed to trust" means: we checked it on the live page and exactly one
        element matched. The list is already sorted best-first, so the first
        unique one wins.

        Example: locators = [id (2 matches), name (1 match)] -> returns the
        `name` one, because the `id` one was not unique.
        """
        for candidate in self.locators:
            if candidate.unique:
                return candidate
        return None


class ActionCache(BaseModel):
    """Everything we wrote down during one agentic run."""

    version: int = CACHE_FORMAT_VERSION
    goal: str  # the sentence a human wrote in the agentic_task
    start_url: str | None = None  # the page the run started on
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    notes: list[ActionNote] = Field(default_factory=list)
