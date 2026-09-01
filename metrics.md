# Step-cache memory layer — metrics

End-to-end comparison of **agentic** (browser-use LLM reasoning) vs
**deterministic** (step-cache replay) runs for the two take-home flows.

All numbers were measured from real runs on this machine. Cache files that
include the per-run metrics: `/home/shubham/agent_step_cache.json` (Roboform)
and `/home/shubham/agent_step_cache_gutenberg.json` (Gutenberg).

## Roboform form fill

Task: *fill the full name as `myname`, address line one as `xyz` and line 2 as
`abc`, city as `SF`* on `roboform.com/filling-test-all-fields`.

| Metric | Agentic | Deterministic | Δ |
|---|---|---|---|
| LLM prompt tokens | 24,366 | 0 | ∞ |
| LLM completion tokens | 2,059 | 0 | ∞ |
| LLM cost | $0.0125 | $0 | ∞ |
| Agent execution time | 14.06 s | ~0 s | — |
| Steps / actions | 5 LLM round-trips | 4 Playwright commands | — |

Cached automation: `test_automation_cached.json`

```json
[
  {"input_text": {"command": "locator(\"[name='04fullname']\").first", "input_text": "myname"}},
  {"input_text": {"command": "locator(\"[name='10address1']\").first", "input_text": "xyz"}},
  {"input_text": {"command": "locator(\"[name='11address2']\").first", "input_text": "abc"}},
  {"input_text": {"command": "locator(\"[name='13adr_city']\").first", "input_text": "SF"}}
]
```

## Project Gutenberg (multi-step + download)

Task: *search for the book Frankenstein by Mary Shelley, open its book page,
and download the 'Plain Text UTF-8' file* on `gutenberg.org`.

| Metric | Agentic | Deterministic | Δ |
|---|---|---|---|
| LLM prompt tokens | 49,923 | 0 | ∞ |
| LLM completion tokens | 3,834 | 0 | ∞ |
| LLM cost | $0.0246 | $0 | ∞ |
| Agent execution time | 35.80 s | ~0 s (actions) | — |
| Steps / actions | 8 (1 redundant scroll) | 7 nodes | — |
| Download | none (text rendered inline) | `pg84.txt` (448,885 B) ✅ | — |

Cached automation: `test_automation_gutenberg_cached.json`

```json
[
  {"input_text": {"command": "locator(\"[name='query']\").first", "input_text": "Frankenstein"}},
  {"click_element": {"command": "get_by_role(\"button\", name=\"Go!\")"}},
  {"click_element": {"command": "locator(\"a[href='/ebooks/84']\").first"}},
  {"click_element": {"command": "get_by_text(\"Other formats & older devices\")"}},
  {"python_script_action": {"execution_code": "fetch(cached_landing_url) + ctx.save_download('pg84.txt')"}}
]
```

### Why the download is a script node (worth reading)

Gutenberg's `.txt.utf-8` link renders **inline** in the browser (no
`Content-Disposition: attachment`), so it never fires a Playwright download.
The cache detects this and compiles the click into a deterministic
`fetch()` + `ctx.save_download()` python-script node. It fetches the **landing
URL** recorded by the cache (`/cache/epub/84/pg84.txt`) — not the raw href —
because the href 302-redirects through `http`, and the in-page `fetch()` API
rejects the `https→http` downgrade with `TypeError: Failed to fetch`.

Verified end-to-end:

```
save_download: saved 'pg84.txt' (448885 bytes) to /tmp/optexity/<task_id>/downloads
found 1 download file(s): [('pg84.txt', 448885)]
uploaded 'pg84.txt' (448885 bytes) to S3 (s3://optexity-task-downloads/<task_id>/pg84.txt)
```

## Notes

- Deterministic wall-clock is dominated by browser startup + output
  uploading; the replay actions themselves take a few seconds total.
- Token/cost savings are the headline: **0 tokens for any cached run**, vs
  26k–55k for the agentic baselines.
- Classification is rule-based and auditable (every step keeps a reason).
- Consecutive duplicate actions are collapsed for replay while the full audit
  trail stays in the cache.