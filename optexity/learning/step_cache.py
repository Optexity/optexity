"""Convert browser-use step_cache.json into Optexity automation JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_optexity_automation(
    cache: dict[str, Any],
    url: str | None = None,
) -> dict[str, Any]:
    """Convert cached steps into an Optexity automation JSON document."""
    steps = cache.get("deterministic_steps", [])
    url = url or cache.get("start_url")

    nodes = []
    for step in steps:
        action_type = step["action_type"]
        payload: dict[str, Any] = {
            "command": step["command"],
            "prompt_instructions": step.get("prompt_instructions", ""),
            "skip_command": False,
            "skip_prompt": True,
        }
        if action_type == "input_text":
            payload["input_text"] = step.get("input_text", "")
            interaction = {"input_text": payload}
        elif action_type == "click_element":
            interaction = {"click_element": payload}
        else:
            continue

        nodes.append({"type": "action_node", "interaction_action": interaction})

    return {
        "url": url,
        "parameters": {"input_parameters": {}, "generated_parameters": {}},
        "nodes": nodes,
    }


def cache_to_optexity_file(
    cache_path: str | Path,
    output_path: str | Path,
    url: str | None = None,
) -> Path:
    with open(cache_path, encoding="utf-8") as f:
        cache_data = json.load(f)
    automation = build_optexity_automation(cache_data, url=url)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(automation, f, indent=2)
    return output
