"""Iterative improvement loop: agentic → cache → deterministic → re-cache → merge → repeat.

Each iteration produces a more deterministic automation by:
1. Running the current automation (agentic on first run, deterministic on subsequent)
2. Collecting action caches from both deterministic and fallback-agentic steps
3. Merging all caches across iterations
4. Converting to an improved deterministic automation

Usage:
    python iterative_loop.py \
        --endpoint "reserve_hotel_booking_com-074ae7dd" \
        --params booking_params.json \
        --server http://localhost:9000 \
        --iterations 3

Set OPTEXITY_LOCAL_AUTOMATION env var on the server to enable automation override
between iterations.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

from cache_to_automation import convert_cache_to_automation


def trigger_task(server_url: str, endpoint_name: str, input_params: dict) -> str | None:
    resp = requests.post(
        f"{server_url}/inference",
        json={
            "endpoint_name": endpoint_name,
            "input_parameters": input_params,
            "max_timeout_in_minutes": 10,
        },
    )
    data = resp.json()
    if data.get("success"):
        return data["task_id"]
    print(f"  Failed to trigger: {data.get('error', data)}", file=sys.stderr)
    return None


def wait_for_task(server_url: str, poll_interval: int = 10, max_wait: int = 600) -> bool:
    start = time.time()
    while time.time() - start < max_wait:
        try:
            resp = requests.get(f"{server_url}/is_task_running")
            if not resp.json():
                return True
        except requests.ConnectionError:
            pass
        time.sleep(poll_interval)
    return False


def collect_caches(task_id: str) -> list[Path]:
    base = Path(f"/tmp/optexity/{task_id}/logs")
    return sorted(base.glob("*/action_cache.json"))


def merge_deterministic_actions(cache_files: list[Path]) -> list[dict]:
    """Merge deterministic actions from multiple cache files, preserving order."""
    all_actions: list[dict] = []
    seen_keys: set[str] = set()

    for cache_path in cache_files:
        with open(cache_path) as f:
            cache = json.load(f)
        for action in cache.get("deterministic_actions_list", []):
            el = action.get("element") or {}
            xpath = el.get("xpath", "")
            ax = el.get("ax_name", "")
            key = f"{action.get('action_type')}:{xpath}:{ax}"
            if key not in seen_keys:
                seen_keys.add(key)
                all_actions.append(action)

    return all_actions


def count_action_types(cache_files: list[Path]) -> dict[str, int]:
    """Count how many actions were deterministic vs agentic across cache files."""
    deterministic = 0
    total = 0
    for cache_path in cache_files:
        with open(cache_path) as f:
            cache = json.load(f)
        total += cache.get("total_actions", 0)
        deterministic += len(cache.get("deterministic_actions_list", []))
    return {"total": total, "deterministic": deterministic, "agentic": total - deterministic}


def run_iteration(
    server_url: str,
    endpoint_name: str,
    input_params: dict,
    iteration: int,
    output_dir: Path,
    all_prior_caches: list[Path],
) -> tuple[list[Path], dict | None]:
    """Run one iteration and return (new_cache_files, automation_dict)."""
    print(f"\n{'='*50}")
    print(f"  Iteration {iteration}")
    print(f"{'='*50}")

    task_id = trigger_task(server_url, endpoint_name, input_params)
    if not task_id:
        return [], None
    print(f"  Task: {task_id}")

    print("  Waiting for completion...", end="", flush=True)
    if not wait_for_task(server_url):
        print(" TIMEOUT")
        return [], None
    print(" done")

    new_caches = collect_caches(task_id)
    print(f"  Cache files: {len(new_caches)}")

    stats = count_action_types(new_caches)
    print(f"  Actions: {stats['total']} total, {stats['deterministic']} deterministic, {stats['agentic']} agentic")

    combined = all_prior_caches + new_caches
    merged_actions = merge_deterministic_actions(combined)
    print(f"  Merged actions (across all iterations): {len(merged_actions)}")

    merged_cache = {
        "task_description": f"Merged from {len(combined)} cache files over {iteration} iterations",
        "start_url": "",
        "total_actions": len(merged_actions),
        "deterministic_actions_list": merged_actions,
    }
    merged_path = output_dir / f"merged_cache_iter{iteration}.json"
    with open(merged_path, "w") as f:
        json.dump(merged_cache, f, indent=2)

    automation = convert_cache_to_automation(str(merged_path), input_params)
    auto_path = output_dir / f"automation_iter{iteration}.json"
    with open(auto_path, "w") as f:
        json.dump(automation, f, indent=2)

    node_count = len(automation.get("nodes", []))
    print(f"  Generated: {node_count} deterministic nodes -> {auto_path}")

    return new_caches, automation


def main():
    parser = argparse.ArgumentParser(description="Iterative automation improvement loop")
    parser.add_argument("--endpoint", required=True, help="Endpoint name from dashboard")
    parser.add_argument("--params", required=True, help="JSON file with input_parameters")
    parser.add_argument("--server", default="http://localhost:9000", help="Optexity server URL")
    parser.add_argument("--iterations", type=int, default=3, help="Max iterations")
    parser.add_argument("--output-dir", default=".", help="Directory for output files")
    parser.add_argument("--auto-override", action="store_true",
                        help="Write automation as local override between iterations")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.params) as f:
        params_data = json.load(f)
    if "input_parameters" in params_data:
        input_params = params_data["input_parameters"]
    elif "parameters" in params_data and "input_parameters" in params_data["parameters"]:
        input_params = params_data["parameters"]["input_parameters"]
    else:
        input_params = params_data

    print("Iterative Automation Improvement Loop")
    print(f"  Endpoint: {args.endpoint}")
    print(f"  Server: {args.server}")
    print(f"  Max iterations: {args.iterations}")

    all_caches: list[Path] = []
    prev_node_count = 0

    for i in range(1, args.iterations + 1):
        new_caches, automation = run_iteration(
            args.server, args.endpoint, input_params,
            i, output_dir, all_caches,
        )

        if not new_caches:
            print(f"\n  No caches from iteration {i}, stopping.")
            break

        all_caches.extend(new_caches)

        if automation and args.auto_override:
            override_path = output_dir / "test_automation.json"
            with open(override_path, "w") as f:
                json.dump(automation, f, indent=2)
            print(f"  Override written to {override_path}")
            print(f"  Set OPTEXITY_LOCAL_AUTOMATION={override_path} on server for next iteration")

        node_count = len(automation.get("nodes", [])) if automation else 0
        if node_count == prev_node_count and i > 1:
            print(f"\n  Converged at {node_count} deterministic nodes after {i} iterations.")
            break
        prev_node_count = node_count

    print(f"\n{'='*50}")
    print("  Summary")
    print(f"{'='*50}")
    print(f"  Total cache files: {len(all_caches)}")
    if all_caches:
        merged_actions = merge_deterministic_actions(all_caches)
        print(f"  Total merged deterministic actions: {len(merged_actions)}")
    final = output_dir / f"automation_iter{min(i, args.iterations)}.json"
    if final.exists():
        print(f"  Final automation: {final}")


if __name__ == "__main__":
    main()
