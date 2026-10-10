# Roboform: agentic run vs cached run

Goal (`agentic.json`): fill Full Name = `myname`, Address 1 = `xyz`, Address 2 = `abc`, City = `SF`
on https://www.roboform.com/filling-test-all-fields.

| | Agentic run (browser-use + LLM) | Cached run (compiled, no LLM) |
|---|---|---|
| Workflow file | `agentic.json` | `cached.json` |
| Wall time, task start to worker finished | ~42 s | ~19 s |
| Time spent on the 4 form actions | ~14 s (2 LLM steps) + final-answer/judge calls | ~3.5 s |
| LLM calls for the form | 2 agent steps + browser-use's end-of-run calls | **0** |
| Steps that needed a fallback | n/a | **0** (all 4 passed on try 1) |

The wall-time gap is smaller than the action-time gap because browser start-up and shut-down cost the
same in both runs (about 10 s here). The 4 actions themselves are roughly 4x faster, and the LLM bill goes to zero.

## What the recorder wrote

* `actions.json`: one note per browser-use action (5 notes: 4 inputs and `done`). Every input has its
  element ID card and locator list, each locator checked on the live page (`match_count`, `unique`).
* `cached.json`: the compiled workflow. 4 steps, all `input_text` with an exact Playwright `command`.
  The `done` action is dropped by the compiler. It validates against Optexity's own `Automation` model.

## Honest notes

* These fields only exposed an `xpath` locator. Their ids look like `04fullname`, which the existing
  `LocatorExtraction._looks_dynamic` rule treats as auto-generated (letters mixed with digits), so it
  skipped them. The xpath was checked to match exactly 1 element, so it is safe today, but it is the
  weakest kind of locator. If the page layout changes, step 1 of the cached file fails and the normal
  Optexity prompt/agentic fallback takes over.
* LLM call counts for the agentic run come from the log (2 steps). Token and cost totals are not
  copied here because browser-use prints them only at the end of the run.

## How to reproduce

```bash
cd ~/optexity/optexity
# 1) Record: runs agentic.json and writes cache/<endpoint>/step_0/{actions,cached}.json
OPTEXITY_RECORD_ACTIONS=true OPTEXITY_TEST_AUTOMATION=$PWD/samples/memory/roboform/agentic.json \
  ENV_PATH=~/optexity/.env optexity inference --port 9000 --child_process_id 0
# 2) Replay: runs the compiled file
OPTEXITY_TEST_AUTOMATION=$PWD/samples/memory/roboform/cached.json \
  ENV_PATH=~/optexity/.env optexity inference --port 9000 --child_process_id 0
# trigger either with:
curl -X POST localhost:9000/inference -H 'Content-Type: application/json' \
  -d '{"endpoint_name":"<your endpoint>","input_parameters":{"stock_ticker":["AAPL"]}}'
```

## Resilience test (self-healing, `resilience/`)

We broke step 2 of `cached.json` on purpose (`broken.json`: its command points at `#field-that-no-longer-exists`)
and ran it twice with `OPTEXITY_WRITE_BACK=true`.

| Run | What happened | Time |
|---|---|---|
| 1 | Step 2's command failed after 10 tries, the normal prompt fallback (1 LLM call) found the box and typed. The new locator matched exactly 1 element, so it was written into the file (`healed.json`). The old command is kept in `healed.repairs.jsonl`. | ~43 s |
| 2 | All 4 steps passed on the first try with their commands. No fallback, no LLM. | ~27 s |

Run 1 paid for one LLM call and about 14 s of failed retries once. Run 2 and every run after it do not.
