# Security Onboarding — `optexity`

Supply-chain controls are enforced via `pre-commit` hooks and GitHub-side
Dependabot. This doc covers what every contributor needs to set up on their
machine, what runs when, and how to unblock yourself.

---

## One-time machine setup

```bash
# Install uv (Python package manager with built-in audit + age-gating)
brew install uv                                 # macOS / Linux
# or:  curl -LsSf https://astral.sh/uv/install.sh | sh

# Ensure pre-commit is available (you almost certainly already have it)
pip install pre-commit                          # or: brew install pre-commit

# Verify
uv --version                                    # any 0.4+
pre-commit --version                            # any recent version
```

## Per-clone activation

Run once per fresh clone:

```bash
cd optexity
pre-commit install --hook-type pre-commit --hook-type pre-push
```

This is required. Without it, the pre-push audit hook will not run locally.

---

## What runs when

| Trigger                              | Hooks that fire                                                                           |
| ------------------------------------ | ----------------------------------------------------------------------------------------- |
| `git commit`                         | `black`, `isort`, `prettier` (unchanged)                                                  |
| `git push`                           | `uv audit` — reports Python dependency advisories                                         |
| PR → GitHub Actions (`lint.yml`)     | commit-stage hooks only                                                                   |
| Merge to `main` → `release-pypi.yml` | Version bump + PyPI publish + **SBOM generation** (CycloneDX, attached to GitHub Release) |

`uv audit` currently exits 0 even when advisories exist — it surfaces them on
your screen but does not block the push. Real enforcement comes from
Dependabot on the GitHub side.

---

## Install-time protection — package age-gating

`pyproject.toml` sets `[tool.uv] exclude-newer = "7 days"`. Any Python package
published within the last 7 days will be rejected by `uv add`, `uv sync`, and
`uv lock`. This blocks typosquatting and dependency-confusion attacks where
malicious packages are published and consumed within hours.

If you legitimately need a very-new package, coordinate before bypassing.

---

## Troubleshooting

| Symptom                                | Fix                                                            |
| -------------------------------------- | -------------------------------------------------------------- |
| `uv audit fails: No project table`     | Pull latest — `pyproject.toml` should have `[project]`         |
| Pre-push hooks don't run at all        | You skipped `pre-commit install --hook-type pre-push` — run it |
| `uv lock` rejects a package as too new | Age-gating is working as intended — see above                  |

---

## Conventions

- **Do not** commit secrets. GitHub push-protection blocks known token formats
  server-side, but local awareness still matters. Keep `.env` out of git.
- **Do not** routinely use `git push --no-verify`. If a hook blocks you, fix
  the underlying issue or open a ticket explaining the exception.
- **Do not** downgrade the age-gating config locally to bypass a
  freshly-published package. Raise it in the team channel.

---

## Evidence for compliance auditors

| Control          | Evidence                                           |
| ---------------- | -------------------------------------------------- |
| Age-gating       | `[tool.uv] exclude-newer` in `pyproject.toml`      |
| Pre-push audit   | `uv-audit` entry in `.pre-commit-config.yaml`      |
| Lockfile pinning | `uv.lock` in repo root                             |
| Dependabot       | GitHub → Security tab → Dependabot alerts + PRs    |
| SBOM             | Attached to each GitHub Release as `sbom.cdx.json` |
