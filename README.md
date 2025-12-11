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

## Installation

### Prerequisites

- Python 3.11+
- Node.js 18+ (included with Conda option)
- Git

### Step 1: Clone the Repository

```bash
git clone git@github.com:Optexity/optexity.git
cd optexity
git submodule sync
git submodule update --init --recursive
```

### Step 2: Create and Activate a Python Environment

Choose **one** of the options below.

#### Option A – Conda (includes Python 3.11 and Node.js)

```bash
conda create -n optexity python=3.11 nodejs
conda activate optexity
```

Install miniconda here: https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html#installing-in-silent-mode

#### Option B – Python `venv`

```bash
python3 -m venv .venv
source .venv/bin/activate
```

> If you pick `venv`, ensure Node.js 18+ is already available on your machine before continuing.

### Step 3: Install Dependencies

Run everything from the repository root:

```bash
pip install -e "external/browser-use"
pip install -e .
playwright install
pre-commit install --install-hooks
pre-commit install --hook-type pre-push
```

### Step 4: Configure Your Environment

Optexity reads configuration from a standard `.env` file via the `ENV_PATH` environment variable.

Create a `.env` file in the repo root:

```bash
touch .env
```

Add the required values:

```bash
API_KEY=YOUR_OPTEXITY_API_KEY           # API key used for authenticated requests
GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY      # API key used for Google Gemini
DEPLOYMENT=dev                          # or "prod" in production
```

You can get your free Google Gemini API key from the [Google AI Studio Console](https://aistudio.google.com).

Then export `ENV_PATH` when running processes that rely on these settings:

```bash
export ENV_PATH=.env
```

> If `ENV_PATH` is not set, the inference server will try to start with defaults and log a warning. For normal usage you should always point `ENV_PATH` at a real `.env` file.

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

The primary way to run browser automations locally is via the inference child process server.

From the repository root:

```bash
ENV_PATH=.env python optexity/inference/child_process.py --port 9000 --child_process_id 0
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
    "endpoint_name": "extract_stock_price",
    "input_parameters": {
      "search_term": ["NVDA"]
    },
    "unique_parameter_names": []
  }'
```

On success, the inference server:

1. Forwards the request to your control plane at `api.optexity.com` using `INFERENCE_ENDPOINT` (defaults to `api/v1/inference`).
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
- [Local Setup](https://docs.optexity.com/docs/building_automations/local-setup)
- [Building Automations](https://docs.optexity.com/docs/building_automations/quickstart)
- [API Reference](https://docs.optexity.com/docs/api-reference/introduction)

## Roadmap

We're actively working on improving Optexity. Here's what's coming:

- 🔜 **Self Improvement**: Agent adaption using self exploration
- 🔜 **More Action Types**: Additional interaction and extraction capabilities
- 🔜 **Performance Optimizations**: Faster execution and reduced resource usage
- 🔜 **Advanced Scheduling**: Built-in task scheduling and cron support
- 🔜 **Cloud Deployment**: Simplified cloud deployment options

Have ideas or feature requests? [Open an issue](https://github.com/Optexity/optexity/issues) or [join our Discord](https://discord.gg/ugeeGbme) to discuss!

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

[**Join Discord →**](https://discord.gg/ugeeGbme)

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run pre-commit checks: `pre-commit run --all-files`
5. Commit your changes (`git commit -m 'Add some amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Examples

Check out our examples directory for sample automations:

- [I94 extraction](https://docs.optexity.com/examples/data_extraction/i94)
- [Healthcare Form Automation](https://docs.optexity.com/examples/healthcare/peachstate-medicaid)
- [QA Testing](https://docs.optexity.com/examples/qa_testing/supabase-login)

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

## Support

- 📖 [Documentation](https://docs.optexity.com)
- 💬 [Discord Community](https://discord.gg/ugeeGbme)
- 🐛 [Report Issues](https://github.com/Optexity/optexity/issues)
- 💭 [Discussions](https://github.com/Optexity/optexity/discussions)
- 📧 [Email Support](mailto:founders@optexity.com)

---

Made with ❤️ by the Optexity team
