# Optexity

Build custom browser agents

1. **Repository Setup**
   Clone the necessary repositories:

```bash
mkdir optexity
cd optexity
git clone git@github.com:Optexity/optexity.git
cd optexity
git submodule sync
git submodule update --init --recursive
```

2. **Environment Setup**
   Create and activate a Conda environment with the required Python and Node.js versions:
   Install miniconda here - https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html#installing-in-silent-mode

```bash
conda create -n optexity python=3.11 nodejs
conda activate optexity
```

3. **Installing Dependencies**
   Install the required packages and build the Playwright framework. Run everything from repo root.

```bash
pip install -e "external/browser-use"
pip install -e .
playwright install
pre-commit install --install-hooks
pre-commit install --hook-type pre-push
```

Running manual pre-commit check `pre-commit install --hook-type pre-push`

## Running inference

```
python main.py \
  --api-key "API_KEY" \
  --base-url http://localhost:8000 \
  --host 0.0.0.0 \
  --port 8001
```
