"""Convert browser-use action cache to deterministic Optexity automation nodes.

Usage:
    python cache_to_automation.py --cache action_cache.json --params params.json [--output test_automation_cached.json]

Reads the action cache produced by browser-use's ActionCache and converts
each deterministic action into an Optexity automation node that uses
playwright locator commands instead of LLM reasoning. Values that match
input parameters are replaced with {key[index]} variable references.
"""

import argparse
import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Value matching: exact -> substring -> token overlap.  Zero hardcoded values.
# ---------------------------------------------------------------------------

def _build_reverse_param_map(input_parameters: dict) -> dict[str, str]:
    """Build value -> {key[index]} mapping for reverse substitution.

    Longer values are inserted first so substring matching prefers
    the most specific parameter.
    """
    reverse: dict[str, str] = {}
    pairs = []
    for key, values in input_parameters.items():
        if not isinstance(values, list):
            continue
        for i, val in enumerate(values):
            val_str = str(val)
            if val_str:
                pairs.append((val_str, f"{{{key}[{i}]}}"))
    pairs.sort(key=lambda p: len(p[0]), reverse=True)
    for val_str, ref in pairs:
        if val_str not in reverse:
            reverse[val_str] = ref
    return reverse


def _tokenize(text: str) -> set[str]:
    """Split text into lowercase word tokens."""
    return set(re.findall(r'\w+', text.lower()))


def _fuzzy_match(param_val: str, text: str) -> bool:
    """Check if param_val and text refer to the same value.

    Strategies (in order):
    1. Exact match (handled by caller before this)
    2. Substring: param value (>=3 chars) is contained in text
    3. Token overlap: >=60% of param tokens appear in text
    """
    if len(param_val) >= 3 and param_val in text:
        return True
    if len(text) >= 3 and text in param_val:
        return True

    param_tokens = _tokenize(param_val)
    if len(param_tokens) < 2:
        return False
    text_tokens = _tokenize(text)
    overlap = param_tokens & text_tokens
    return len(overlap) / len(param_tokens) >= 0.6


def _substitute_value(text: str, reverse_map: dict[str, str]) -> str:
    """Replace literal values with variable references.

    Tries exact match first, then fuzzy matching (substring/token overlap).
    """
    if text in reverse_map:
        return reverse_map[text]
    for literal, var_ref in reverse_map.items():
        if _fuzzy_match(literal, text):
            return var_ref
    return text


# ---------------------------------------------------------------------------
# Locator building
# ---------------------------------------------------------------------------

def _looks_dynamic(value: str) -> bool:
    """Detect IDs that are generated/positional and likely to change."""
    if re.search(r'[0-9a-f]{8,}', value):
        return True
    if re.search(r'\d{5,}', value):
        return True
    if re.search(r'-\d+$', value):
        return True
    return False


def _escape(s: str) -> str:
    return s.replace('"', '\\"').replace("'", "\\'")


def _build_locator_command(element: dict) -> str | None:
    """Build a Playwright locator command from cached element info.

    Priority: data-test > id (stable) > name > placeholder > role+ax_name > aria-label > xpath.
    Locators use captured values directly (no variable substitution) so
    they stay stable for replay.  If they break on different params, the
    agentic fallback handles it.
    """
    if not element:
        return None

    attrs = element.get('attributes', {})
    tag = element.get('tag_name', '')

    data_test = attrs.get('data-test', '') or attrs.get('data-testid', '')
    if data_test:
        return f'locator("[data-test=\\"{data_test}\\"]")'

    el_id = attrs.get('id', '')
    if el_id and not _looks_dynamic(el_id):
        return f'locator("#{el_id}")'

    name = attrs.get('name', '')
    if name:
        return f'locator("[name=\'{name}\']")'

    placeholder = attrs.get('placeholder', '')
    if tag == 'input' and placeholder:
        return f'get_by_placeholder("{_escape(placeholder)}")'

    ax_name = element.get('ax_name', '')
    role = attrs.get('role', '')
    if ax_name and role:
        return f'get_by_role("{role}", name="{_escape(ax_name)}")'
    if ax_name and tag in ('button', 'a', 'input', 'select', 'textarea'):
        return f'get_by_text("{_escape(ax_name)}")'

    aria_label = attrs.get('aria-label', '')
    if aria_label:
        return f'get_by_label("{_escape(aria_label)}")'

    xpath = element.get('xpath', '')
    if xpath:
        return f'locator("xpath={xpath}")'

    return None


def _field_description(element: dict | None) -> str:
    """Human-readable field description from element info."""
    if not element:
        return 'the field'
    attrs = element.get('attributes', {})
    ax_name = element.get('ax_name', '')
    if ax_name:
        return ax_name
    placeholder = attrs.get('placeholder', '')
    if placeholder:
        return placeholder
    name = attrs.get('name', '')
    if name:
        return name
    return element.get('tag_name', 'the field')


# ---------------------------------------------------------------------------
# Node conversion
# ---------------------------------------------------------------------------

def _is_low_quality_action(cached_action: dict) -> bool:
    """Filter out actions that are likely misclicks or non-interactive elements."""
    element = cached_action.get('element')
    if not element:
        return cached_action.get('action_type') in ('click', 'click_element')
    attrs = element.get('attributes', {})
    tag = element.get('tag_name', '')
    has_semantic = (
        attrs.get('data-test') or attrs.get('data-testid')
        or attrs.get('role') or attrs.get('aria-label')
        or attrs.get('name') or element.get('ax_name')
    )
    has_id = attrs.get('id', '') and not _looks_dynamic(attrs.get('id', ''))
    if not has_semantic and not has_id and tag in ('div', 'span', 'section', 'header', 'footer', 'main', 'nav'):
        return True
    return False


def _action_to_node(cached_action: dict, reverse_map: dict[str, str]) -> dict | None:
    """Convert a single cached action to an Optexity automation node."""
    if _is_low_quality_action(cached_action):
        return None

    action_type = cached_action.get('action_type', '')
    params = cached_action.get('action_params', {})
    element = cached_action.get('element')
    locator_cmd = _build_locator_command(element) if element else None
    new_tab = _opens_new_tab(element)

    if action_type in ('input_text', 'input'):
        text = params.get('text', '')
        if not locator_cmd:
            return None

        var_text = _substitute_value(text, reverse_map)
        field_desc = _field_description(element)
        hint = f"Enter the {field_desc} '{var_text}' into the field."

        node = {
            'type': 'action_node',
            'interaction_action': {
                'input_text': {
                    'command': locator_cmd,
                    'prompt_instructions': hint,
                    'input_text': var_text,
                }
            },
        }
        if new_tab:
            node['expect_new_tab'] = True
        return node

    elif action_type in ('click_element', 'click'):
        if not locator_cmd:
            return None

        ax_name = element.get('ax_name', '') if element else ''
        tag = element.get('tag_name', '') if element else ''
        desc = ax_name or tag
        sub_desc = _substitute_value(desc, reverse_map)

        node = {
            'type': 'action_node',
            'interaction_action': {
                'click_element': {
                    'command': locator_cmd,
                    'prompt_instructions': f"Click '{sub_desc}'.",
                }
            },
        }
        if new_tab:
            node['expect_new_tab'] = True
        return node

    elif action_type in ('navigate', 'go_to_url'):
        url = params.get('url', '')
        if not url:
            return None
        return {
            'type': 'action_node',
            'interaction_action': {
                'go_to_url': {
                    'url': _substitute_value(url, reverse_map),
                }
            },
        }

    elif action_type == 'scroll':
        down = params.get('down', True)
        return {
            'type': 'action_node',
            'interaction_action': {
                'scroll': {
                    'down': down,
                }
            },
        }

    elif action_type == 'select_dropdown':
        text = params.get('text', '')
        if not locator_cmd:
            return None
        var_text = _substitute_value(text, reverse_map)
        return {
            'type': 'action_node',
            'interaction_action': {
                'select_option': {
                    'command': locator_cmd,
                    'prompt_instructions': f'Select option: {var_text}',
                    'select_values': [var_text],
                }
            },
        }

    elif action_type == 'send_keys':
        keys = params.get('keys', '')
        return {
            'type': 'action_node',
            'interaction_action': {
                'key_press': {
                    'type': keys,
                    'prompt_instructions': f'Press {keys}',
                }
            },
        }

    return None


def _opens_new_tab(element: dict | None) -> bool:
    """Detect if clicking this element would open a new tab."""
    if not element:
        return False
    attrs = element.get('attributes', {})
    return attrs.get('target', '') == '_blank'


# ---------------------------------------------------------------------------
# Autocomplete gap detection
# ---------------------------------------------------------------------------

def _find_autocomplete_gaps(
    deterministic: list[dict],
    reverse_map: dict[str, str],
) -> dict[int, dict]:
    """Detect click-input -> click-option gaps needing a synthetic input_text.

    Returns a mapping from the raw deterministic index of the input-click
    to the synthetic input_text node to insert AFTER that action's node.
    """
    inserts: dict[int, dict] = {}
    if len(deterministic) < 2 or not reverse_map:
        return inserts

    for i in range(len(deterministic) - 1):
        cur = deterministic[i]
        nxt = deterministic[i + 1]

        if cur.get('action_type') not in ('click', 'click_element'):
            continue
        if nxt.get('action_type') not in ('click', 'click_element'):
            continue
        if _is_low_quality_action(cur) or _is_low_quality_action(nxt):
            continue

        cur_el = cur.get('element') or {}
        nxt_el = nxt.get('element') or {}
        cur_attrs = cur_el.get('attributes', {})
        nxt_attrs = nxt_el.get('attributes', {})

        is_input_click = (
            cur_el.get('tag_name') == 'input'
            or cur_attrs.get('role') in ('combobox', 'searchbox', 'textbox')
        )
        is_option_click = (
            nxt_el.get('tag_name') in ('li', 'option', 'div')
            or nxt_attrs.get('role') in ('option', 'listbox', 'menuitem')
        )

        if not is_input_click or not is_option_click:
            continue

        option_text = nxt_el.get('ax_name', '') or nxt_el.get('text_content', '')
        if not option_text:
            continue

        matched_var = None
        for literal, var_ref in reverse_map.items():
            if literal.lower() in option_text.lower() or option_text.lower() in literal.lower():
                matched_var = var_ref
                break

        if not matched_var:
            continue

        locator_cmd = _build_locator_command(cur_el)
        if not locator_cmd:
            continue

        field_desc = _field_description(cur_el)
        inserts[i] = {
            'type': 'action_node',
            'interaction_action': {
                'input_text': {
                    'command': locator_cmd,
                    'prompt_instructions': f"Type {matched_var} into the {field_desc}.",
                    'input_text': matched_var,
                }
            },
        }

    return inserts


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert_cache_to_automation(
    cache_path: str | Path,
    input_parameters: dict | None = None,
    start_url: str | None = None,
) -> dict:
    """Read an action cache file and produce a deterministic Optexity automation."""
    with open(cache_path) as f:
        cache = json.load(f)

    url = start_url or cache.get('start_url', '')
    deterministic = cache.get('deterministic_actions_list', [])
    params = input_parameters or {}
    reverse_map = _build_reverse_param_map(params)

    autocomplete_inserts = _find_autocomplete_gaps(deterministic, reverse_map)

    nodes = []
    for i, action in enumerate(deterministic):
        node = _action_to_node(action, reverse_map)
        if node:
            nodes.append(node)
        if i in autocomplete_inserts:
            nodes.append(autocomplete_inserts[i])

    automation = {
        'url': url,
        'parameters': {
            'input_parameters': params,
            'generated_parameters': {},
        },
        'nodes': nodes,
    }
    return automation


def main():
    parser = argparse.ArgumentParser(description='Convert action cache to deterministic automation')
    parser.add_argument('--cache', default='action_cache.json', help='Path to action cache JSON')
    parser.add_argument('--params', default=None, help='JSON file with input_parameters for variable substitution')
    parser.add_argument('--output', default='test_automation_cached.json', help='Output automation JSON path')
    parser.add_argument('--url', default=None, help='Override start URL')
    args = parser.parse_args()

    if not Path(args.cache).exists():
        print(f'Error: Cache file {args.cache} not found', file=sys.stderr)
        sys.exit(1)

    input_parameters = {}
    if args.params:
        with open(args.params) as f:
            params_data = json.load(f)
        if 'input_parameters' in params_data:
            input_parameters = params_data['input_parameters']
        elif 'parameters' in params_data and 'input_parameters' in params_data['parameters']:
            input_parameters = params_data['parameters']['input_parameters']
        else:
            input_parameters = params_data

    automation = convert_cache_to_automation(args.cache, input_parameters, args.url)

    with open(args.output, 'w') as f:
        json.dump(automation, f, indent=2)

    print(f'Generated {len(automation["nodes"])} deterministic nodes -> {args.output}')

    try:
        from optexity.schema.automation import Automation
        Automation.model_validate(automation)
        print('Schema validation: PASSED')
    except Exception as e:
        print(f'Schema validation: FAILED - {e}', file=sys.stderr)


if __name__ == '__main__':
    main()
