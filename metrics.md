# Metrics — agentic vs cached (roboform)

Site: https://www.roboform.com/filling-test-all-fields  
Fields: Full Name=`myname`, Address1=`xyz`, Address2=`abc`, City=`SF`

## Agentic run (browser-use + Gemini)

Source: Phase 1 / T1 export run, task `527d7841-8473-4f53-82f1-7a9746cc842d`  
(server log timestamps 16:22:47 → 16:24:27 IST on 2026-08-12)

| Metric | Value |
| --- | --- |
| Wall time | ~100 s (allocate → worker exit) |
| LLM calls | **4** (`📍 Step 1` timed out + Steps 2–4 succeeded; each step is one Gemini `ainvoke`) |
| Tokens | not exposed in worker logs for this run |
| Outcome | success; exporter wrote 4 `input_text` + 1 `done` |

## Cached run (deterministic Playwright commands, 0 LLM)

Source: Phase 4 / T4, override=`test_automation_cached.json`, 3 consecutive `/inference` runs  
(server log 22:16:14 → 22:17:21 IST on 2026-08-12)

| Run | task_id | Wall time (allocate → idle) | Status | LLM calls |
| --- | --- | --- | --- | --- |
| 1 | `24bf99b4-c26b-4ea0-bfd4-35dfd642770b` | 24.7 s | success | **0** |
| 2 | `c833c245-56ef-472e-b603-032bb5690fca` | 22.4 s | success | **0** |
| 3 | `acf7b041-ea9e-49cf-83fa-1196ba59fca4` | 24.5 s | success | **0** |

LLM-call greps on the cached-run server log (all must be 0):

| Pattern | Count |
| --- | --- |
| `Starting a browser-use agent` | 0 |
| `provider=gemini` | 0 |
| `ChatLiteLLM` / `ainvoke` / `litellm.acompletion` | 0 |
| `agentic_task` | 0 |
| `prompt_tokens` / `TokenUsage` | 0 |

Content verification (`python -m optexity.tools.verify_form_fill`): **PASS** — all 4 field values read back correctly.

## Delta

| | Agentic | Cached (avg of 3) |
| --- | --- | --- |
| Wall time | ~100 s | ~23.9 s (**~4.2× faster**) |
| LLM calls | 4 | **0** |
| Correctness | fill observed in agent log | 3/3 success + read-back PASS |
