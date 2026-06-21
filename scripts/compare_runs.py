#!/usr/bin/env python3
"""
Compare agentic vs cached run logs and extract performance metrics.

Usage:
    python compare_runs.py <agentic_log_file> <cached_log_file>
"""

import re
import sys
from datetime import datetime


def parse_log(filepath: str) -> dict:
    """Extract key metrics from an optexity run log."""
    with open(filepath, "r") as f:
        lines = f.readlines()

    metrics = {
        "llm_steps": 0,
        "model_created": False,
        "nodes_run": 0,
        "actions": [],
        "first_timestamp": None,
        "last_timestamp": None,
        "task_status": "unknown",
    }

    ts_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")

    for line in lines:
        # Extract timestamps
        ts_match = ts_pattern.match(line)
        if ts_match:
            ts_str = ts_match.group(1)
            ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S,%f")
            if metrics["first_timestamp"] is None:
                metrics["first_timestamp"] = ts
            metrics["last_timestamp"] = ts

        # Count LLM steps (each is one LLM call)
        if "📍 Step" in line:
            metrics["llm_steps"] += 1

        # Check if LLM model was created
        if "Created model" in line or "provider=google" in line:
            metrics["model_created"] = True

        # Count nodes
        if "-----Running node new" in line:
            metrics["nodes_run"] += 1

        # Track action types
        if "Executing command-based action" in line:
            metrics["actions"].append("command-based")
        if "agentic_task" in line and "Running interaction action" in line:
            metrics["actions"].append("agentic")

        # Task status
        if "completed with status" in line:
            if "success" in line:
                metrics["task_status"] = "success"
            else:
                metrics["task_status"] = "failed"

        # Track fills and clicks
        if "InputTextAction successful" in line:
            metrics["actions"].append("fill_success")
        if "🖱️ Clicked" in line:
            metrics["actions"].append("click_success")
        if "⌨️ Typed" in line:
            metrics["actions"].append("type_success")

    # Calculate duration
    if metrics["first_timestamp"] and metrics["last_timestamp"]:
        delta = metrics["last_timestamp"] - metrics["first_timestamp"]
        metrics["total_seconds"] = delta.total_seconds()
    else:
        metrics["total_seconds"] = 0

    return metrics


def compare(agentic_path: str, cached_path: str):
    print("=" * 65)
    print("  AGENTIC vs CACHED RUN COMPARISON")
    print("=" * 65)

    agentic = parse_log(agentic_path)
    cached = parse_log(cached_path)

    rows = [
        ("Status", agentic["task_status"], cached["task_status"]),
        ("Total time (seconds)", f"{agentic['total_seconds']:.1f}s", f"{cached['total_seconds']:.1f}s"),
        ("LLM steps (calls)", str(agentic["llm_steps"]), str(cached["llm_steps"])),
        ("LLM model created", str(agentic["model_created"]), str(cached["model_created"])),
        ("Nodes executed", str(agentic["nodes_run"]), str(cached["nodes_run"])),
        ("Successful fills", str(agentic["actions"].count("type_success")), str(agentic["actions"].count("fill_success"))),
    ]

    print(f"\n{'Metric':<30} {'Agentic':<18} {'Cached':<18}")
    print("-" * 65)
    for label, a_val, c_val in rows:
        print(f"{label:<30} {a_val:<18} {c_val:<18}")

    # Savings
    if agentic["total_seconds"] > 0 and cached["total_seconds"] > 0:
        speedup = agentic["total_seconds"] / cached["total_seconds"]
        time_saved = agentic["total_seconds"] - cached["total_seconds"]
        print(f"\n{'Time saved':<30} {time_saved:.1f}s ({speedup:.1f}x faster)")
    print(f"{'LLM calls eliminated':<30} {agentic['llm_steps']}")
    print(f"{'Token cost':<30} Gemini free tier (0 billed)")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare_runs.py <agentic.log> <cached.log>")
        sys.exit(1)
    compare(sys.argv[1], sys.argv[2])