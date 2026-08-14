"""Dot-path variable resolver for API call response dicts.

Resolves patterns like {var.field}, {var.nested.field}, {var.array[0].field}
in action node string fields. Only applies to dict-valued generated variables
(e.g., API call responses). Does NOT interfere with the existing {key[index]}
replacement system.
"""

import json
import logging
import re
from typing import Any

from optexity.guardrails.expressions import safe_evaluate

logger = logging.getLogger(__name__)

# Matches {identifier.path} where path must start with a dot-segment.
# This ensures {key[0]} (existing format) is NEVER matched.
# Examples that match:  {api_result.status}, {api_result.data[0].name}
# Examples that don't:  {key[0]}, {key[index]}
_API_VAR_PATTERN = re.compile(r"\{(\w+)(\.\w+(?:\.\w+|\[\d+\])*)\}")


def _parse_path_segments(path: str) -> list[tuple[str, str | int]]:
    """Parse '.foo.bar[0].baz' into [('attr','foo'), ('attr','bar'), ('index',0), ('attr','baz')]"""
    result = []
    for part in re.finditer(r"\.(\w+)|\[(\d+)\]", path):
        if part.group(1) is not None:
            result.append(("attr", part.group(1)))
        else:
            result.append(("index", int(part.group(2))))
    return result


def _resolve_path(data: Any, path_str: str) -> Any:
    """Walk a dict/list structure following a dot/bracket path.

    Returns the resolved value, or None if the path doesn't exist.
    """
    segments = _parse_path_segments(path_str)
    current = data
    for seg_type, seg_value in segments:
        if seg_type == "attr":
            if isinstance(current, dict) and seg_value in current:
                current = current[seg_value]
            else:
                return None
        elif seg_type == "index":
            if isinstance(current, (list, tuple)) and seg_value < len(current):
                current = current[seg_value]
            else:
                return None
    return current


def resolve_api_variables_in_node(action_node, generated_variables: dict) -> None:
    """Resolve all {var.path} patterns in an action node using dict-valued generated variables.

    Serializes the node to JSON to discover patterns, then uses the existing
    action_node.replace() infrastructure to perform substitutions.
    """
    node_json = action_node.model_dump_json()

    # Deduplicate patterns to avoid redundant replacements
    seen = set()
    for match in _API_VAR_PATTERN.finditer(node_json):
        full_pattern = match.group(0)
        if full_pattern in seen:
            continue
        seen.add(full_pattern)

        var_name = match.group(1)
        path_str = match.group(2)

        if var_name not in generated_variables:
            continue

        data = generated_variables[var_name]
        if not isinstance(data, dict):
            continue

        resolved = _resolve_path(data, path_str)
        if resolved is None:
            continue

        if isinstance(resolved, (dict, list)):
            replacement = json.dumps(resolved)
        else:
            replacement = str(resolved)

        action_node.replace(full_pattern, replacement)


def evaluate_poll_condition(condition: str, response: dict) -> bool:
    """Evaluate a poll condition expression against an API response dict.

    Supports both top-level keys and dot-path syntax:
        "status_code == 200"
        "body.status == 'completed'"
        "body.progress >= 100"

    All identifiers that match response keys (with or without dot-paths)
    are resolved before evaluation.
    """

    try:
        return bool(safe_evaluate(condition, response))
    except Exception as e:
        logger.warning(f"Poll condition evaluation failed: '{condition}' -> {e}")
        return False
