# Optexity

**Build custom browser agents** with AI-powered automation. Record browser interactions, extract data, and run complex workflows via a simple API. You can extract data from websites, fill out forms, do QA testing, and more.

## Features

- 🎯 **Visual Recording**: Record browser interactions with the Optexity Recorder Chrome extension
- 🤖 **AI-Powered**: Uses LLMs to handle dynamic content and find elements intelligently
- 📊 **Data Extraction**: Extract structured data from web pages using LLM-based extraction
- 🔄 **Workflow Automation**: Chain multiple actions together for complex browser workflows
- 🚀 **API-First**: Run automations via REST API with simple JSON requests
- 🎨 **Dashboard**: Manage and monitor your automations through the Optexity dashboard

## Quick Start

### 1. Create an Account

Head to [dashboard.optexity.com](https://dashboard.optexity.com) and sign up for a free account.

### 2. Get Your API Key

Once logged in, navigate to the **API Keys** section in your dashboard and create a new key.

### 3. Install the Recorder Extension

Install the **Optexity Recorder** extension from the [Chrome Web Store](https://chromewebstore.google.com/detail/optexity-recorder/pbaganbicadeoacahamnbgohafchgakp). This extension captures your browser interactions and converts them into automation workflows.

### Prerequisites

- Python 3.11+
- Git

## Create and Activate a Python Environment (Optional)

Choose **one** of the options below.

#### Option A – Conda (includes Python 3.11 and Node.js)

```bash
conda create -n optexity python=3.11
conda activate optexity
```

Install miniconda here: https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html#installing-in-silent-mode

#### Option B – Python `venv`

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Installation

### Quick Installation (from PyPI)

Install Optexity directly from PyPI:

```bash
pip install optexity
optexity install-browsers
```

**OR**

### Installation from Source

If you want to clone and edit from source:

```bash
git clone git@github.com:Optexity/optexity.git
cd optexity
pip install -e .
optexity install-browsers
```

## Set required environment variables:

```bash
OPTEXITY_API_KEY=YOUR_OPTEXITY_API_KEY           # API key used for authenticated requests
GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY      # API key used for Google Gemini
DEPLOYMENT=dev                          # or "prod" in production
```

You can get your free Google Gemini API key from the [Google AI Studio Console](https://aistudio.google.com).

### Choosing the LLM

Optexity runs on [LiteLLM](https://docs.litellm.ai/docs/providers), so any provider it
supports works. Set a primary model and, optionally, a fallback used when the primary
fails — the two can be on different providers, each with its own key:

```bash
LLM_MODEL=anthropic/claude-sonnet-4-6
LLM_MODEL_API_KEY=YOUR_ANTHROPIC_API_KEY

LLM_MODEL_FALLBACK=openai/gpt-4.1-mini
LLM_MODEL_FALLBACK_API_KEY=YOUR_OPENAI_API_KEY
```

`LLM_MODEL` defaults to `gemini/gemini-3.5-flash-lite`. Either key may be omitted, in which
case the provider's own environment variable is used (`GOOGLE_API_KEY` /
`GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, ...).

## Recording Your First Automation

The fastest way to create an automation is by recording your actions directly in the browser.

### Steps

1. **Navigate to the target website**: Open Chrome and go to the website you want to automate (e.g., `https://stockanalysis.com/`)

2. **Start capturing**: Click the Optexity Recorder extension icon and hit **Start Capture**

3. **Perform your actions**:
    - Click on the "Search" button
    - Enter the stock symbol in the search bar
    - Click on the first result in the search results

4. **Stop and save**: When finished, click **Complete Capture**. The automation is automatically saved to your dashboard as a JSON file.

### Recording Tips

- Perform actions slowly and deliberately for better accuracy
- Avoid unnecessary scrolling or hovering
- The recorder captures clicks, text input, and form selections

## Running Your Automation

### Start the Inference Server

The primary way to run browser automations locally is via the inference server.

```bash
optexity inference --port 9000 --child_process_id 0
```

Key parameters:

- **`--port`**: HTTP port the local inference server listens on (e.g. `9000`).
- **`--child_process_id`**: Integer identifier for this worker. Use different IDs if you run multiple workers in parallel.

When this process starts, it exposes:

- `GET /health` – health and queue status
- `GET /is_task_running` – whether a task is currently executing
- `POST /inference` – main endpoint to allocate and execute tasks

### Call the `/inference` Endpoint

With the server running on `http://localhost:9000`, you can allocate a task by sending an `InferenceRequest` to `/inference`.

#### Request Schema

- **`endpoint_name`**: Name of the automation endpoint to execute. This must match a recording/automation defined in the Optexity dashboard.
- **`input_parameters`**: `dict[str, list[str]]` – all input values for the automation, as lists of strings.
- **`unique_parameter_names`**: `list[str]` – subset of keys from `input_parameters` that uniquely identify this task (used for deduplication and validation). Only one task with the same `unique_parameter_names` will be allocated. If no `unique_parameter_names` are provided, the task will be allocated immediately.

#### Example `curl` Request

```bash
curl -X POST http://localhost:9000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "endpoint_name": "extract_price_stockanalysis",
    "input_parameters": {
      "search_term": ["NVDA"]
    },
    "unique_parameter_names": []
  }'
```

On success, the inference server:

1. Forwards the request to your control plane at `inference-api.optexity.com` using `INFERENCE_ENDPOINT` (defaults to `api/v1/inference`).
2. Receives a serialized `Task` object from the control plane.
3. Enqueues that `Task` locally and starts processing it in the background.
4. Returns a `202 Accepted` response:

```json
{
    "success": true,
    "message": "Task has been allocated"
}
```

> Task execution (browser automation, screenshots, outputs, etc.) happens asynchronously in the background worker. You can see it running locally in your browser.

### Monitor Execution

You can monitor the task on the dashboard. It will show the status, errors, outputs, and all the downloaded files.

## Video Tutorial

[![Watch the video](https://img.youtube.com/vi/q51r3idYtxo/0.jpg)](https://www.youtube.com/watch?v=q51r3idYtxo)

## Documentation

For detailed documentation, visit our [documentation site](https://docs.optexity.com):

- [Recording First Automation](https://docs.optexity.com/docs/getting_started/recording-first-inference)
- [Running First Inference](https://docs.optexity.com/docs/getting_started/running-first-inference)
- [Local Setup](https://docs.optexity.com/docs/building-automations/local-setup)
- [Building Automations](https://docs.optexity.com/docs/building-automations/quickstart)
- [API Reference](https://docs.optexity.com/docs/api-reference/introduction)

## Roadmap

We're actively working on improving Optexity. Here's what's coming:

- 🔜 **Self Improvement**: Agent adaption using self exploration
- 🔜 **More Action Types**: Additional interaction and extraction capabilities
- 🔜 **Performance Optimizations**: Faster execution and reduced resource usage
- 🔜 **Advanced Scheduling**: Built-in task scheduling and cron support
- 🔜 **Cloud Deployment**: Simplified cloud deployment options

Have ideas or feature requests? [Open an issue](https://github.com/Optexity/optexity/issues) or [join our Discord](https://discord.gg/VsRSAZSw7m) to discuss!

## Contributing

We welcome contributions! Here's how you can help:

### Reporting Issues

Found a bug or have a feature request? Please [open an issue](https://github.com/Optexity/optexity/issues) on GitHub. Include:

- A clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)

### Discussions

Have questions, ideas, or want to discuss the project? Use [GitHub Discussions](https://github.com/Optexity/optexity/discussions) to:

- Ask questions
- Share ideas
- Discuss best practices
- Get help from the community

### Community

Join our Discord community to:

- Chat with the founders directly
- Get real-time support
- Share your automations
- Connect with other users

[**Join Discord →**](https://discord.gg/VsRSAZSw7m)

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run pre-commit checks: `pre-commit run --all-files`
5. Commit your changes (`git commit -m 'Add some amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Releasing to PyPI and GitHub (maintainers)

Releases are automated via GitHub Actions:

- **When**: Every push/merge to `main` bumps the **4th version component** in `pyproject.toml` (e.g. `0.1.5.5` → `0.1.5.6`), commits that change, creates a **GitHub Release** (tag + release notes), and publishes the new version to **PyPI**.
- **Setup** (one-time):
    1. In PyPI: [Account settings → API tokens](https://pypi.org/manage/account/token/) — create a token with scope **Entire account** (or limit to project `optexity`).
    2. In GitHub: **Repository** or **Organization** → **Settings → Secrets and variables → Actions** — add a secret:
        - **Name**: `PYPI_API_TOKEN`
        - **Value**: your PyPI API token (starts with `pypi-`).
- **Flow**: Merge a PR to `main` → workflow runs → version bump commit is pushed → GitHub Release (e.g. `v0.1.5.6`) is created with generated release notes → package is built and uploaded to PyPI. The workflow skips when the last commit is the automated bump, so it does not loop.

## Examples

Check out our examples directory for sample automations:

- [I94 extraction](https://docs.optexity.com/examples/data_extraction/i94)
- [Healthcare Form Automation](https://docs.optexity.com/examples/healthcare/peachstate-medicaid)
- [QA Testing](https://docs.optexity.com/examples/qa_testing/supabase-login)

## Deterministic Cache (`feature/deterministic-cache`)

Run an agentic browser task **once**, record what it did, then replay it with
**zero LLM calls**. Same result, faster and deterministic.

Needs the matching [`rangerfc56-sys/browser-use`](https://github.com/rangerfc56-sys/browser-use)
fork (branch `feature/deterministic-cache`) for the record + filter steps.

### How it works

```
agentic run  →  export actions  →  filter noise  →  build locators  →  replay (0 LLM)
 (Gemini)       cached_run.json    keep real steps   test_..._cached.json   /inference
```

| Step | File |
| --- | --- |
| Export what the agent did | `browser_use/cache/action_export.py` |
| Drop the useless steps | `browser_use/cache/step_filter.py` |
| Turn steps into Playwright locators | `optexity/tools/build_cached_automation.py` |
| Load the cache instead of the LLM | `optexity/inference/child_process.py` |
| Build the cache via LLM (bonus) | `optexity/tools/llm_build_automation.py` |
| Check the replay actually worked | `optexity/tools/verify_form_fill.py`, `verify_final_page.py` |

### Try it (roboform)

Needs both forks installed editable, plus `OPTEXITY_API_KEY`, `GOOGLE_API_KEY`,
`DEPLOYMENT=dev`.

```bash
git clone -b feature/deterministic-cache https://github.com/rangerfc56-sys/optexity.git
git clone -b feature/deterministic-cache https://github.com/rangerfc56-sys/browser-use.git

python3 -m venv .venv && source .venv/bin/activate
# optexity ships its own browser_use copy; remove it so the fork wins.
pip install -e ./optexity
pip uninstall -y optexity-browser-use browser-use || true
rm -rf .venv/lib/python*/site-packages/browser_use
pip install -e ./browser-use
optexity install-browsers

export OPTEXITY_API_KEY=... GOOGLE_API_KEY=... DEPLOYMENT=dev

# start the server with the roboform cache pinned
cd optexity
export OPTEXITY_LOCAL_AUTOMATION=test_automation_cached.json
optexity inference --port 9000 --child_process_id 0

# in another shell: fire the task, then verify the fields
curl -s http://localhost:9000/inference -H 'Content-Type: application/json' \
  -d '{"endpoint_name":"fill_roboform_test_form-3b699ebe","input_parameters":{"first_name":["myname"]},"unique_parameter_names":[]}'
python -m optexity.tools.verify_form_fill
```

You should see `Loaded local override automation ...`, no Gemini calls, and
`verify_form_fill` **PASS**.

### Design choices

**Locators** — pick the most stable handle available, in order:
`id` → `name` → `placeholder`/`data-*`/`aria-label` → role + text → `xpath`.
Never browser-use bracket indices (`[67]`, per-run only). Never xpath inside
shadow DOM (it can't reach in) — the builder errors instead.

**Filter** — plain rules, each logged with a reason:
drop non-actions and failures, merge click-then-type and type-then-Enter,
keep the last action when an element was touched twice. Rules over an LLM
because it's cheap and auditable; swapping in an LLM filter later is possible.

**Waits** — no `time.sleep()`. When a step changed the page, the cache sets
`end_sleep_time`, which Optexity runs as `wait_for_load_state("load")`. If the
URL lags behind the click, the builder adds a `go_to_url` using the real
recorded URL.

### Results

| Site | Agentic | Cached (avg of 3) | LLM calls |
| --- | --- | --- | --- |
| Roboform | ~100 s | ~24 s (~4×) | 4 → **0** |
| Stockanalysis (NVDA Financials) | 64 s | ~24 s (~2.7×) | 5 → **0** |

Details: [`metrics.md`](metrics.md). LLM auto-builder vs hand-built: no diffs
([`llm_vs_handbuilt_diff.md`](llm_vs_handbuilt_diff.md)).

### Left out on purpose

- Iterative verify/repair loop (Bonus B) — validate + retry already enough.
- LLM-based filter — rules are simpler and traceable.

### PRs

- [optexity#1](https://github.com/rangerfc56-sys/optexity/pull/1)
- [browser-use#1](https://github.com/rangerfc56-sys/browser-use/pull/1)

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

## Support

- 📖 [Documentation](https://docs.optexity.com)
- 💬 [Discord Community](https://discord.gg/VsRSAZSw7m)
- 🐛 [Report Issues](https://github.com/Optexity/optexity/issues)
- 💭 [Discussions](https://github.com/Optexity/optexity/discussions)
- 📧 [Email Support](mailto:founders@optexity.com)

---

Made with ❤️ by the Optexity team
