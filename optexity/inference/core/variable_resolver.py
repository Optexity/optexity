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

logger = logging.getLogger(__name__)

# Matches {identifier.path} where path must start with a dot-segment.
# This ensures {key[0]} (existing format) is NEVER matched.
# Examples that match:  {api_result.status}, {api_result.data[0].name}
# Examples that don't:  {key[0]}, {key[index]}
_API_VAR_PATTERN = re.compile(r"\{(\w+)(\.\w+(?:\.\w+|\[\d+\])*)\}")

# Matches {var[name]} where name is a non-numeric identifier (not the literal "index")
# Examples that match:  {row_coords_x[current_click_index]}, {data[my_idx]}
# Examples that don't:  {var[0]}, {var[index]}, {var.field}
_DYNAMIC_INDEX_PATTERN = re.compile(r"\{(\w+)\[([A-Za-z_]\w*)\]\}")


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

    def _resolve_identifier(match: re.Match) -> str:
        """Replace an identifier or dot-path with its resolved value."""
        full_path = match.group(0)
        segments = full_path.split(".")
        root = segments[0]

        # Only resolve if root is a key in the response
        if root not in response:
            return full_path

        if len(segments) == 1:
            # Top-level key like "status_code"
            resolved = response[root]
        else:
            # Dot-path like "body.status"
            resolved = _resolve_path(response, "." + full_path)

        if resolved is None:
            return "None"
        if isinstance(resolved, str):
            return repr(resolved)
        if isinstance(resolved, (dict, list)):
            return repr(resolved)
        return str(resolved)

    # Match identifiers: standalone words or dot-paths, optionally with [N]
    resolved_condition = re.sub(
        r"\b([a-zA-Z_]\w*(?:\.\w+)*(?:\[\d+\])?)\b", _resolve_identifier, condition
    )

    try:
        return bool(eval(resolved_condition))  # noqa: S307
    except Exception as e:
        logger.warning(f"Poll condition eval failed: '{resolved_condition}' -> {e}")
        return False


def resolve_dynamic_indices_in_node(action_node, generated_variables: dict) -> None:
    """Resolve {var[some_var]} patterns to {var[N]} where N is the value of some_var.

    Looks up `some_var` in generated_variables. If `some_var` is a list, uses [0].
    Skips `{var[index]}` (handled by for-loop expansion) and `{var[N]}` (numeric).
    Leaves untouched any pattern where the inner name is not in generated_variables.
    """
    node_json = action_node.model_dump_json()

    seen = set()
    for match in _DYNAMIC_INDEX_PATTERN.finditer(node_json):
        full_pattern = match.group(0)
        if full_pattern in seen:
            continue
        seen.add(full_pattern)

        outer_var = match.group(1)
        inner_name = match.group(2)

        # Skip the literal "index" — it's handled by for-loop expansion
        if inner_name == "index":
            continue

        # Skip if inner_name is not a known generated variable
        if inner_name not in generated_variables:
            continue

        value = generated_variables[inner_name]
        # If it's a list, use the first element
        if isinstance(value, list):
            if not value:
                continue
            index_value = value[0]
        else:
            index_value = value

        try:
            resolved_index = int(index_value)
        except (TypeError, ValueError):
            logger.warning(
                f"Dynamic index resolution: '{inner_name}' value '{index_value}' "
                f"is not convertible to int — skipping"
            )
            continue

        new_pattern = f"{{{outer_var}[{resolved_index}]}}"
        action_node.replace(full_pattern, new_pattern)
