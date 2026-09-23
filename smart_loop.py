"""Smart iterative loop: agentic → cache → LLM build → replay → improve → repeat.

Each round:
  1. Run the current automation (or 2-node agentic for round 0).
  2. Read node_outcomes.json to classify each node's execution path.
  3. For prompt_fallback nodes: extract the winning LLM locator from optexity.log
     and promote it to `command` for the next round.
  4. For failed nodes: revert to an agentic_task sub-step.
  5. For command_success / deterministic nodes: lock them (no change).
  6. Use the LLM builder on the latest merged cache to produce the next automation.
  7. Repeat until all nodes are command_success / deterministic, or max_rounds.

Usage:
    python smart_loop.py \\
        --endpoint reserve_hotel_booking_com-074ae7dd \\
        --params booking_params.json \\
        --url https://www.booking.com \\
        --agentic-task "Search for hotels in {destination_city[0]}..." \\
        --max-rounds 5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SERVER_URL = "http://localhost:9000"
POLL_INTERVAL = 10   # seconds between status checks
MAX_WAIT = 900       # max seconds to wait for a task

# ---------------------------------------------------------------------------
# Task execution helpers
# ---------------------------------------------------------------------------

def trigger_task(endpoint: str, input_parameters: dict, timeout_min: int = 15) -> str | None:
    """POST to /inference and return the task_id."""
    payload = {
        "endpoint_name": endpoint,
        "input_parameters": input_parameters,
        "max_timeout_in_minutes": timeout_min,
    }
    try:
        resp = requests.post(f"{SERVER_URL}/inference", json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        task_id = data.get("task_id") or data.get("id")
        logger.info(f"Task triggered: {task_id}")
        return task_id
    except Exception as e:
        logger.error(f"Failed to trigger task: {e}")
        return None


def wait_for_task(task_id: str) -> dict | None:
    """Poll /task/<id> until done and return the result dict."""
    deadline = time.time() + MAX_WAIT
    while time.time() < deadline:
        try:
            resp = requests.get(f"{SERVER_URL}/task/{task_id}", timeout=15)
            if resp.ok:
                data = resp.json()
                status = data.get("status", "")
                if status in ("success", "failed", "error"):
                    logger.info(f"Task {task_id} finished: {status}")
                    return data
        except Exception:
            pass
        time.sleep(POLL_INTERVAL)
    logger.error(f"Task {task_id} timed out after {MAX_WAIT}s")
    return None


def find_task_dir(task_id: str, base: str = "/tmp/optexity") -> Path | None:
    p = Path(base) / task_id
    return p if p.exists() else None


# ---------------------------------------------------------------------------
# Outcome reading
# ---------------------------------------------------------------------------

def read_node_outcomes(task_dir: Path) -> list[dict]:
    path = task_dir / "logs" / "node_outcomes.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def outcomes_summary(outcomes: list[dict]) -> dict[str, int]:
    from collections import Counter
    return dict(Counter(o["outcome"] for o in outcomes))


def all_deterministic(outcomes: list[dict]) -> bool:
    return all(
        o["outcome"] in ("command_success", "deterministic", "skipped")
        for o in outcomes
    )


# ---------------------------------------------------------------------------
# Log parsing: extract winning LLM locator + failure errors per node
# ---------------------------------------------------------------------------

_LLM_FALLBACK_RE = re.compile(
    r"LLM fallback locator \[index \d+\]: (page\.[^\s(]+\([^)]*\)(?:\.[^\s(]+\([^)]*\))*)"
)
_NODE_START_RE = re.compile(r"-----Running node new (\d+)-----")
# "ClickElementAction failed after 10 tries: error: <message>"
_ACTION_FAIL_RE = re.compile(
    r"(\w+Action) failed after \d+ tries: (.+?)$"
)


def extract_fallback_locators(log_path: Path) -> dict[int, str]:
    """Parse optexity.log and return {node_index: winning_locator_command}.

    The locator is formatted as `page.get_by_role(...)` — we strip the leading
    `page.` so it can be used directly as an Optexity `command` value.
    Also strips the trailing `.click(...)` / `.fill(...)` method call since
    Optexity appends the method itself.
    """
    if not log_path.exists():
        return {}

    locators: dict[int, str] = {}
    current_node: int | None = None

    for line in log_path.read_text(errors="replace").splitlines():
        node_match = _NODE_START_RE.search(line)
        if node_match:
            current_node = int(node_match.group(1))
            continue

        fallback_match = _LLM_FALLBACK_RE.search(line)
        if fallback_match and current_node is not None:
            full_locator = fallback_match.group(1)
            # Strip leading "page." and trailing method call like ".click(...)" / ".fill(...)"
            cmd = re.sub(r"^page\.", "", full_locator)
            cmd = re.sub(r"\.(click|fill|select_option|check|uncheck)\([^)]*\)$", "", cmd)
            if current_node not in locators:  # keep first (highest-confidence)
                locators[current_node] = cmd

    return locators


def extract_failed_errors(log_path: Path) -> dict[int, str]:
    """Parse optexity.log and return {node_index: short_error} for failed nodes.

    Only captures the first failure per node (the command-based attempt).
    Truncated to 120 chars so it stays compact in the LLM prompt.
    """
    if not log_path.exists():
        return {}

    errors: dict[int, str] = {}
    current_node: int | None = None

    for line in log_path.read_text(errors="replace").splitlines():
        node_match = _NODE_START_RE.search(line)
        if node_match:
            current_node = int(node_match.group(1))
            continue

        fail_match = _ACTION_FAIL_RE.search(line)
        if fail_match and current_node is not None and current_node not in errors:
            raw_error = fail_match.group(2).strip()
            # Keep only the first sentence / first 120 chars — enough for the LLM
            short = raw_error.split("\n")[0][:120]
            errors[current_node] = short

    return errors


def build_failed_history(
    round_num: int,
    automation: dict,
    outcomes: list[dict],
    failed_errors: dict[int, str],
    history_path: Path,
) -> list[dict]:
    """Append failed-node records to a persistent loop_history.json.

    Only records nodes whose outcome is 'failed' — prompt_fallback and
    command_success nodes don't need history (they either have a winning
    locator or are already locked).
    Returns the full accumulated history list.
    """
    existing: list[dict] = []
    if history_path.exists():
        try:
            existing = json.loads(history_path.read_text())
        except Exception:
            pass

    outcome_by_node = {o["node"]: o["outcome"] for o in outcomes}
    nodes = automation.get("nodes", [])

    for idx, node in enumerate(nodes):
        if outcome_by_node.get(idx) != "failed":
            continue
        ia = node.get("interaction_action", {})
        tried_cmd = ""
        for atype in ("click_element", "input_text", "select_option"):
            action = ia.get(atype)
            if action:
                tried_cmd = action.get("command", "")
                break
        existing.append({
            "round": round_num,
            "node": idx,
            "tried_command": tried_cmd,
            "error": failed_errors.get(idx, "unknown error"),
        })

    history_path.write_text(json.dumps(existing, indent=2))
    return existing


def format_failed_history(history: list[dict]) -> str:
    """Render failed history as compact lines for the LLM prompt.

    Example output:
      [r1, n5] tried: locator('[data-testid="hotel"]') → strict mode violation: resolved to 15 elements
      [r2, n5] tried: get_by_test_id("title-link").nth(0) → TimeoutError: element not found
    """
    if not history:
        return ""
    lines = [
        f"[r{h['round']}, n{h['node']}] tried: {h['tried_command']} → {h['error']}"
        for h in history
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Automation patching based on outcomes
# ---------------------------------------------------------------------------

def patch_automation(
    automation: dict,
    outcomes: list[dict],
    fallback_locators: dict[int, str],
    agentic_task_text: str,
    input_parameters: dict,
) -> dict:
    """Return an updated automation dict based on per-node outcomes.

    Strategy per node:
    - command_success / deterministic / skipped → keep as-is (locked)
    - prompt_fallback → if we have a winning locator from log, promote it to command
    - failed → replace interaction_action with agentic_task fallback
    - agentic → keep as-is (already LLM-driven)
    """
    import copy

    patched = copy.deepcopy(automation)
    nodes = patched.get("nodes", [])

    # outcomes list is indexed by automation step (step_index is 0-based within automation)
    outcome_by_node: dict[int, str] = {}
    for o in outcomes:
        # node field in outcomes is memory.automation_state.step_index which starts at 0
        outcome_by_node[o["node"]] = o["outcome"]

    for idx, node in enumerate(nodes):
        outcome = outcome_by_node.get(idx, "unknown")

        if outcome in ("command_success", "deterministic", "skipped", "agentic"):
            continue  # locked

        if outcome == "prompt_fallback":
            winning = fallback_locators.get(idx)
            if winning:
                ia = node.get("interaction_action", {})
                for action_key in ("click_element", "input_text", "select_option"):
                    if action_key in ia and ia[action_key] is not None:
                        ia[action_key]["command"] = winning
                        logger.info(f"Node {idx}: promoted fallback locator → {winning}")
                        break

        elif outcome == "failed":
            # Replace with agentic_task that describes what this node should do
            original_hint = ""
            ia = node.get("interaction_action", {})
            for action_key in ("click_element", "input_text", "select_option"):
                action_data = ia.get(action_key)
                if action_data:
                    original_hint = action_data.get("prompt_instructions", "")
                    break

            task_description = original_hint or agentic_task_text
            node["interaction_action"] = {
                "agentic_task": {
                    "task": task_description,
                    "max_steps": 5,
                    "backend": "browser_use",
                }
            }
            logger.info(f"Node {idx}: failed → reverted to agentic_task")

    return patched


# ---------------------------------------------------------------------------
# Cache merging helper
# ---------------------------------------------------------------------------

def find_latest_cache(task_dir: Path) -> Path | None:
    """Find the action_cache.json in the task logs."""
    for pattern in ["logs/step_*/action_cache.json", "logs/action_cache.json"]:
        matches = list(task_dir.glob(pattern))
        if matches:
            return sorted(matches)[-1]
    return None


# ---------------------------------------------------------------------------
# Round 0: build initial 2-node agentic automation
# ---------------------------------------------------------------------------

def make_agentic_automation(
    start_url: str,
    agentic_task: str,
    input_parameters: dict,
) -> dict:
    return {
        "url": start_url,
        "parameters": {
            "input_parameters": input_parameters,
            "generated_parameters": {},
        },
        "nodes": [
            {
                "type": "action_node",
                "interaction_action": {
                    "close_overlay_popup": {
                        "task": "Close any popup or overlay. Look for X, close, or dismiss buttons.",
                        "max_steps": 3,
                        "backend": "browser_use",
                    }
                },
            },
            {
                "type": "action_node",
                "interaction_action": {
                    "agentic_task": {
                        "task": agentic_task,
                        "max_steps": 30,
                        "backend": "browser_use",
                    }
                },
            },
        ],
    }


# ---------------------------------------------------------------------------
# Server restart helper
# ---------------------------------------------------------------------------

def restart_server(automation_path: str, env_path: str = ".env") -> None:
    """Kill the current optexity server and restart with a new automation override."""
    subprocess.run(["pkill", "-f", "optexity inference"], capture_output=True)
    time.sleep(2)
    env = {
        **os.environ,
        "OPTEXITY_LOCAL_AUTOMATION": str(automation_path),
        "ENV_PATH": env_path,
    }
    subprocess.Popen(
        ["conda", "run", "-n", "optexity", "optexity", "inference",
         "--port", "9000", "--child_process_id", "0"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Wait for server to be ready
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            r = requests.get(f"{SERVER_URL}/health", timeout=3)
            if r.ok:
                logger.info("Server ready.")
                return
        except Exception:
            pass
        time.sleep(1)
    logger.warning("Server may not be ready yet — continuing anyway.")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_loop(
    endpoint: str,
    input_parameters: dict,
    start_url: str,
    agentic_task: str,
    max_rounds: int = 5,
    env_path: str = ".env",
    output_dir: str = ".",
) -> dict | None:
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    current_automation: dict | None = None
    current_automation_path: Path | None = None

    for round_num in range(max_rounds):
        logger.info(f"\n{'='*60}")
        logger.info(f"ROUND {round_num}")
        logger.info(f"{'='*60}")

        # --- Build automation for this round ---
        if round_num == 0:
            # Pure agentic: let browser-use discover everything
            auto = make_agentic_automation(start_url, agentic_task, input_parameters)
            auto_path = output_dir / "automation_round_0.json"
            with open(auto_path, "w") as f:
                json.dump(auto, f, indent=2)
            logger.info(f"Round 0: running 2-node agentic automation → {auto_path}")
        else:
            auto_path = current_automation_path
            logger.info(f"Round {round_num}: using {auto_path}")

        # --- Restart server with this automation ---
        restart_server(str(auto_path), env_path)

        # --- Trigger task ---
        task_id = trigger_task(endpoint, input_parameters)
        if task_id is None:
            logger.error("Could not trigger task — aborting loop.")
            break

        result = wait_for_task(task_id)
        if result is None:
            logger.error(f"Round {round_num}: task timed out.")
            break

        task_dir = find_task_dir(task_id)
        if task_dir is None:
            logger.error(f"Round {round_num}: task directory not found for {task_id}.")
            break

        # --- Read outcomes ---
        outcomes = read_node_outcomes(task_dir)
        summary = outcomes_summary(outcomes)
        logger.info(f"Round {round_num} outcomes: {summary}")

        # --- Check convergence ---
        if all_deterministic(outcomes) and round_num > 0:
            logger.info(f"Converged at round {round_num}: all nodes deterministic.")
            return current_automation

        # --- Find action cache ---
        cache_path = find_latest_cache(task_dir)
        if cache_path is None:
            logger.warning(f"Round {round_num}: no action cache found — skipping build.")
            break

        logger.info(f"Round {round_num}: cache at {cache_path}")

        log_path = task_dir / "logs" / "optexity.log"
        history_path = output_dir / "loop_history.json"

        # --- Collect failed-node history from this round ---
        failed_errors = extract_failed_errors(log_path)
        history = build_failed_history(round_num, current_automation or {}, outcomes, failed_errors, history_path)
        failed_history_str = format_failed_history(history) or None
        if failed_history_str:
            logger.info(f"Failed history ({len(history)} entries):\n{failed_history_str}")

        # --- Build next automation from cache (LLM builder) ---
        from llm_cache_to_automation import llm_convert
        try:
            next_auto = llm_convert(cache_path, input_parameters, start_url, failed_history=failed_history_str)
        except Exception as e:
            logger.warning(f"LLM builder failed ({e}), using rule-based fallback.")
            from cache_to_automation import convert_cache_to_automation
            next_auto = convert_cache_to_automation(cache_path, input_parameters, start_url)

        # --- Patch based on previous round's outcomes (round > 0) ---
        if round_num > 0 and outcomes:
            fallback_locators = extract_fallback_locators(log_path)
            if fallback_locators:
                logger.info(f"Extracted fallback locators for nodes: {list(fallback_locators.keys())}")
            next_auto = patch_automation(
                next_auto, outcomes, fallback_locators, agentic_task, input_parameters
            )

        # --- Validate ---
        from optexity.schema.automation import Automation
        try:
            Automation.model_validate(next_auto)
            n_nodes = len(next_auto.get("nodes", []))
            logger.info(f"Round {round_num}: built {n_nodes}-node automation, schema valid.")
        except Exception as e:
            logger.error(f"Round {round_num}: automation schema invalid — {e}")
            break

        # --- Save ---
        next_path = output_dir / f"automation_round_{round_num + 1}.json"
        with open(next_path, "w") as f:
            json.dump(next_auto, f, indent=2)
        logger.info(f"Saved → {next_path}")

        current_automation = next_auto
        current_automation_path = next_path

    logger.info("Loop ended.")
    return current_automation


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Smart iterative automation loop")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--params", required=True, help="JSON file with input_parameters")
    parser.add_argument("--url", required=True, help="Start URL")
    parser.add_argument("--agentic-task", required=True, help="Task description for agentic nodes")
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--env-path", default=".env")
    parser.add_argument("--output-dir", default="./loop_output")
    args = parser.parse_args()

    with open(args.params) as f:
        p = json.load(f)
    input_parameters = (
        p.get("input_parameters")
        or p.get("parameters", {}).get("input_parameters", {})
        or p
    )

    final = run_loop(
        endpoint=args.endpoint,
        input_parameters=input_parameters,
        start_url=args.url,
        agentic_task=args.agentic_task,
        max_rounds=args.max_rounds,
        env_path=args.env_path,
        output_dir=args.output_dir,
    )

    if final:
        out = Path(args.output_dir) / "automation_final.json"
        with open(out, "w") as f:
            json.dump(final, f, indent=2)
        print(f"\nFinal automation saved → {out}")
    else:
        print("\nLoop did not converge or failed.")


if __name__ == "__main__":
    main()
