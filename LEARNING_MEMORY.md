# Cross-run procedural memory

This feature learns an executable workflow from a successful Browser Use run and
reuses it on later executions of the same Optexity workflow. It is an
evidence-gated replay system, not a prompt cache and not a website-specific rules
engine.

## What it does

1. **Discover:** the original `agentic_task` runs with Browser Use. Optexity saves
   `raw_history.json`, and the paired Browser Use compiler creates
   `browser_use_action_cache.json`.
2. **Convert:** deterministic adapters translate the ordered cache into typed
   Optexity nodes. A schema-constrained LLM may resolve an unsupported step, but
   it cannot invent locator commands or input values.
3. **Store a draft:** the generated Automation, locator alternatives, provenance,
   and source-run evidence are saved as a workflow-scoped draft. Drafts are not
   trusted yet.
4. **Validate on the next real run:** Playwright first checks the remembered
   locator immediately. If it fails for a potentially temporary reason, the
   resolver waits once for a bounded readiness window and tries once more.
5. **Replay:** validated commands execute through the existing Optexity action
   handlers. Input, select, check, uncheck, and upload actions also verify their
   resulting DOM state.
6. **Judge and promote:** one final LLM judge compares the original task with the
   final browser state. A passing draft becomes the active memory version.
7. **Keep learning:** active versions are validated on every run. Failed
   candidates are degraded, alternatives and older validated generations can be
   tried before execution, and a safe miss returns to fresh Browser Use discovery.

The first run therefore pays the full agent cost. A healthy replay replaces the
multi-turn planning loop with local Playwright checks plus one workflow-level
judge call.

## Memory identity and parameters

The implementation is website-agnostic, while each memory is deliberately scoped
to one Optexity workflow and agentic node using company, workspace, user,
recording, Automation version, and node path.

Memory compatibility fingerprints parameter **names, arity, and types**, not
their current values. For example, a memory learned with
`{stock_ticker[0]} = AAPL` is reused when the next run supplies `NVDA`.

- Input, secure, and generated values are mapped back to their original
  placeholders before memory is persisted.
- The current run resolves those placeholders immediately before replay.
- Static workflow constants, such as a fixed menu choice, remain literal.
- If one observed value could refer to multiple parameters, learning stops rather
  than guessing.
- Resolved secret values are held only in ephemeral binding objects and are not
  written to the learned Automation.

## Success and trust model

There is no universal deterministic rule that can prove every natural-language
browser task succeeded. A unique locator proves target identity; an input-value
check proves that a field changed; neither proves a broader outcome such as
"the correct flight was booked."

The current promotion gate therefore combines:

- an explicit successful Browser Use source-run judge verdict;
- deterministic locator, capability, and state-effect checks during replay;
- a final semantic LLM judge using the task, ordered execution trace, screenshot,
  URL, title, and accessibility tree; and
- a page signature retained as supplemental regression evidence.

An explicit deterministic postcondition should override an LLM judge only when it
is part of the workflow contract and fully specifies the desired outcome (for
example, an exact confirmation ID plus the expected backend state). The framework
does not infer or hardcode such conditions per website.

## Failure behavior

- A cache miss runs the original Browser Use agent and creates a new draft.
- A transient readiness timeout is recorded separately and does not poison an
  otherwise valid memory.
- A hard locator or action failure degrades/rejects that generation and preserves
  evidence for later ranking.
- An older compatible version may be tried only before any cached action mutates
  the page.
- Once a replay has partially mutated the page, the run fails closed instead of
  launching an agent on unknown state.
- Corrupt memory and persistence failures are best-effort: they do not break the
  original non-learning execution path.

## Current safety boundary

The first production-shaped checkpoint learns a final top-level Browser Use
`agentic_task` with no post-processing nodes. This makes fallback safe because no
downstream state has been consumed. Nested, looped, or mid-workflow agentic nodes
need an explicit checkpoint/reset contract before they can safely use automatic
fallback.

## Local run guide

The feature uses the sibling Browser Use checkout because its history compiler is
newer than the currently published `optexity-browser-use` package.

From the Optexity repository:

```bash
export PYTHONPATH="$(pwd)/../browser-use:$(pwd)"
export LEARNING_MEMORY_ENABLED=true
export LEARNING_MEMORY_DIRECTORY=/tmp/optexity-learning-memory
export LEARNING_MEMORY_SOFT_TARGET_MS=50
export LEARNING_MEMORY_CANDIDATE_TIMEOUT_MS=250
export LEARNING_MEMORY_READINESS_WAIT_MS=3000
export LEARNING_MEMORY_REPAIR_BUDGET_MS=750

python -c "import browser_use; print(browser_use.__file__); import browser_use.agent.history_compiler"
optexity inference --host 127.0.0.1 --port 9000 --child_process_id 0
```

Run the same dashboard workflow at least three times:

1. Run 1: Browser Use discovery; a draft generation is created.
2. Run 2: draft replay; a passing judge promotes it to active.
3. Run 3: active replay; validation and learning remain enabled.

Inspect:

- durable local memory under `LEARNING_MEMORY_DIRECTORY`;
- `learning_memory_observation.json` in each task log directory;
- `raw_history.json`, `browser_use_action_cache.json`, conversion plans, and
  generated Automation files in the source agentic step directory.

The memory directory is an assignment/local-filesystem implementation. A
multi-worker production deployment should provide the same versioned contract
through a shared transactional store.

## Verification

```bash
PYTHONPATH="$(pwd)/../browser-use:$(pwd)" \
  python -m unittest discover -s tests -p 'test_*.py' -v

PYTHONPATH="$(pwd)/../browser-use:$(pwd)" \
  python -m pyright \
  optexity/inference/core/automation_cache \
  optexity/inference/core/learning_memory
```

The focused tests cover optional imports, parameter rebinding, workflow identity,
source-judge gating, atomic version transitions, and stale replay protection.

## What is novel here

- **Proof-carrying procedural memory:** executable nodes retain action, locator,
  source-step, and validation provenance.
- **Bounded candidate repair:** a 50 ms soft target measures the healthy locator
  path; correctness uses a per-candidate timeout, at most two alternatives, and a
  bounded one-time readiness retry.
- **Canary promotion and rollback:** new memories start as drafts, active memory is
  continuously revalidated, and failed generations do not overwrite the last
  known evidence.
- **Parameter-agnostic replay:** one workflow memory serves different runtime
  values without persisting those values as the learned procedure.
- **Hybrid correctness gate:** deterministic mechanics reduce hallucination while
  one semantic judge handles outcomes that cannot be expressed generically.
