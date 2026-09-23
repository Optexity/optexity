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


def _build_reverse_param_map(input_parameters: dict) -> dict[str, str]:
    """Build value -> {key[index]} mapping for reverse substitution."""
    reverse: dict[str, str] = {}
    for key, values in input_parameters.items():
        if not isinstance(values, list):
            continue
        for i, val in enumerate(values):
            val_str = str(val)
            if val_str and val_str not in reverse:
                reverse[val_str] = f"{{{key}[{i}]}}"
    return reverse


def _substitute_value(text: str, reverse_map: dict[str, str]) -> str:
    """Replace literal values with variable references where they match."""
    if text in reverse_map:
        return reverse_map[text]
    return text


def _build_locator_command(element: dict) -> str | None:
    """Build a Playwright locator command from cached element info.

    Prioritizes stable selectors:
    1. data-test attribute (test IDs, most stable)
    2. CSS id
    3. name attribute
    4. placeholder
    5. role + accessible name
    6. aria-label
    7. XPath fallback
    """
    if not element:
        return None

    attrs = element.get('attributes', {})
    tag = element.get('tag_name', '')

    data_test = attrs.get('data-test', '')
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
    if ax_name:
        return f'get_by_text("{_escape(ax_name)}")'

    aria_label = attrs.get('aria-label', '')
    if aria_label:
        return f'get_by_label("{_escape(aria_label)}")'

    xpath = element.get('xpath', '')
    if xpath:
        return f'locator("xpath={xpath}")'

    return None


def _looks_dynamic(value: str) -> bool:
    if re.search(r'[0-9a-f]{8,}', value):
        return True
    if re.search(r'\d{5,}', value):
        return True
    return False


def _escape(s: str) -> str:
    return s.replace('"', '\\"').replace("'", "\\'")


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


def _action_to_node(cached_action: dict, reverse_map: dict[str, str]) -> dict | None:
    """Convert a single cached action to an Optexity automation node."""
    action_type = cached_action.get('action_type', '')
    params = cached_action.get('action_params', {})
    element = cached_action.get('element')
    locator_cmd = _build_locator_command(element) if element else None

    if action_type in ('input_text', 'input'):
        text = params.get('text', '')
        if not locator_cmd:
            return None

        var_text = _substitute_value(text, reverse_map)
        field_desc = _field_description(element)
        hint = f"Enter the {field_desc} '{var_text}' into the field."

        return {
            'type': 'action_node',
            'interaction_action': {
                'input_text': {
                    'command': locator_cmd,
                    'prompt_instructions': hint,
                    'input_text': var_text,
                }
            },
        }

    elif action_type in ('click_element', 'click'):
        if not locator_cmd:
            return None

        ax_name = element.get('ax_name', '') if element else ''
        tag = element.get('tag_name', '') if element else ''
        desc = ax_name or tag

        return {
            'type': 'action_node',
            'interaction_action': {
                'click_element': {
                    'command': locator_cmd,
                    'prompt_instructions': f"Click '{desc}'.",
                }
            },
        }

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

    nodes = []
    for action in deterministic:
        node = _action_to_node(action, reverse_map)
        if node:
            nodes.append(node)

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
