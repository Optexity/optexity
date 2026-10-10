# Results for every scenario

Each folder has `agentic.json` (the goal), `actions.json` (what the recorder wrote) and, when the run
was good enough to cache, `cached.json` (the compiled workflow). Timings are wall-clock for one task,
including browser start and stop (about 10 s of each number).

| Scenario | Agentic run | Compiled? | Cached replay | Verdict |
|---|---|---|---|---|
| `roboform` (4 text inputs) | ~42 s, 2 LLM steps | yes, 4 cached steps | ~19 s, 0 LLM, 0 fallbacks | **Works.** See `roboform/RESULTS.md`. |
| `stockanalysis` (type, pick result, open tab) | ~67 s, 8 steps | yes: 3 cached + 1 agentic step | ~75 s | **Partly works.** Cached input and first click ran with no LLM. The "Financials" link timed out and the normal Optexity fallback rescued it, so the replay was not faster. |
| `saucedemo` (login, cart, checkout) | 197 s, hit the 30 step limit | **refused** | n/a | The agent itself never finished (its cart clicks did not register in this Chrome). The compiler correctly refused to cache a run where the agent said `success: false`. Secret handling was still exercised: the password was stored as `{secret_1[0]}` and the real value never reached disk. |
| `action_types` (dropdown, checkboxes, download) | 152 s, hit the 20 step limit | **refused** | n/a | The agent could not leave the first page (its `navigate` calls were ignored on the-internet.herokuapp.com, whose first load had timed out). The dropdown `select_dropdown` note itself was recorded correctly with a unique `id` locator, but we do not cache half a run. |

## What the failed runs taught us (all fixed or documented)

1. **Never cache a run the agent says failed.** `compile_cache` now raises if any `done` note has
   `success: false`. Before this rule the saucedemo run compiled into 28 junk steps.
2. **A locator is only trusted if it was counted on the live page and matched exactly 1 element.**
   On stockanalysis `get_by_role("link", name=...)` matched 8 elements for the result row, so the
   recorder skipped it and picked the one unique candidate instead.
3. **`LocatorExtraction` can build invalid CSS** for Tailwind classes that contain `:`
   (for example `lg:rounded-3xl`). The count check throws, the locator is marked "could not verify",
   and it is never used. Escaping those characters in `LocatorExtraction` would be a good follow-up.
4. **A click can "succeed" without doing anything.** On stockanalysis the same result was clicked
   twice, once with no effect. A compiler rule that drops a click when the page did not change and the
   same element is clicked again right after is a good follow-up.
5. A locator can be unique and still be fragile (xpath, or a class like `active-search-result` that only
   exists while hovering). The runtime fallback (prompt, then agentic) is what protects those cases.

## Not run

The download action and the "website changed" resilience test were not completed because the agent
could not reach those pages on the day (see `action_types` above).
