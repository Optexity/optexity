# Memory layer: run an agentic task once, replay it without an LLM

An `agentic_task` asks browser-use (an LLM) to work out the clicks on every run. This package records
what the agent did once and turns it into a normal Optexity workflow whose steps use exact Playwright
`command`s. Later runs use no LLM.

```
agentic run ──> recorder ──> actions.json ──> compiler ──> cached.json ──> normal Optexity run
 (browser-use)   (live page)   (the notes)     (no browser)   (commands)      (0 LLM calls)
                                                                    │
                       command breaks, prompt fallback fixes it ────┴──> writeback patches the file
```

| File | What it does |
|---|---|
| `schema.py` | Pydantic models of `actions.json` (one note per action, with checked locators). |
| `recorder.py` | Listens to browser-use through `register_action_callback`; counts matches for every locator on the live page; writes `actions.json` and `cached.json`. |
| `compiler.py` | Pure function: notes in, validated Optexity `Automation` out. Drops failed, overwritten and useless steps; uses only locators that matched exactly 1 element; anything unsafe stays a small `agentic_task`; secrets become placeholders; refuses runs where the agent said it failed. |
| `writeback.py` | Self-healing: if a command breaks and the fallback fixes it, saves the repaired command (only when it is unique and the step succeeded) and keeps the old one in `<file>.repairs.jsonl`. |

## Switches (all off by default)

| Env var | Meaning |
|---|---|
| `OPTEXITY_TEST_AUTOMATION` | Local workflow file that replaces the server's workflow (dev only). Default `test_automation.json`. |
| `OPTEXITY_RECORD_ACTIONS=true` | Record every `agentic_task` and write `actions.json` + `cached.json` into `OPTEXITY_CACHE_DIR` (default `cache/`). |
| `OPTEXITY_WRITE_BACK=true` | Enable self-healing of the local workflow file. |

Compile by hand: `python -m optexity.memory.compiler actions.json cached.json`.

Tests: `pytest tests/memory` (no browser, no LLM). Evidence from real runs: `samples/memory/RESULTS.md`.
