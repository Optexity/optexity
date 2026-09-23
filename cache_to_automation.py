"""Convert browser-use action cache to deterministic Optexity automation nodes.

Usage:
    python cache_to_automation.py [--cache action_cache.json] [--output test_automation_cached.json]

Reads the action cache produced by browser-use's ActionCache and converts
each deterministic action into an Optexity automation node that uses
playwright locator commands instead of LLM reasoning.
"""

import argparse
import json
import re
import sys
from pathlib import Path


def _build_locator_command(element: dict) -> str | None:
    """Build a Playwright locator command from cached element info.

    Prioritizes stable selectors in order:
    1. CSS id (#id)
    2. name attribute (input[name=...])
    3. role + accessible name
    4. XPath fallback
    """
    if not element:
        return None

    attrs = element.get('attributes', {})
    tag = element.get('tag_name', '')

    # 1. ID-based selector
    el_id = attrs.get('id', '')
    if el_id and not _looks_dynamic(el_id):
        return f'locator("#{el_id}").first'

    # 2. Name attribute (common for form fields)
    name = attrs.get('name', '')
    if name:
        return f'locator("[name=\'{name}\']").first'

    # 3. Type attribute for inputs
    input_type = attrs.get('type', '')
    placeholder = attrs.get('placeholder', '')
    if tag == 'input' and placeholder:
        return f'get_by_placeholder("{_escape(placeholder)}").first'

    # 4. Role + accessible name
    ax_name = element.get('ax_name', '')
    role = attrs.get('role', '')
    if ax_name and role:
        return f'get_by_role("{role}", name="{_escape(ax_name)}").first'
    if ax_name:
        return f'get_by_text("{_escape(ax_name)}").first'

    # 5. aria-label
    aria_label = attrs.get('aria-label', '')
    if aria_label:
        return f'get_by_label("{_escape(aria_label)}").first'

    # 6. XPath fallback
    xpath = element.get('xpath', '')
    if xpath:
        return f'locator("xpath={xpath}").first'

    return None


def _looks_dynamic(value: str) -> bool:
    """Heuristic: IDs with long hex strings or UUID patterns are likely dynamic."""
    if re.search(r'[0-9a-f]{8,}', value):
        return True
    if re.search(r'\d{5,}', value):
        return True
    return False


def _escape(s: str) -> str:
    """Escape quotes for use in Playwright locator strings."""
    return s.replace('"', '\\"').replace("'", "\\'")


def _action_to_node(cached_action: dict) -> dict | None:
    """Convert a single cached action to an Optexity automation node."""
    action_type = cached_action.get('action_type', '')
    params = cached_action.get('action_params', {})
    element = cached_action.get('element')
    locator_cmd = _build_locator_command(element) if element else None

    if action_type in ('input_text', 'input'):
        text = params.get('text', '')
        if not locator_cmd:
            return None

        tag = element.get('tag_name', '') if element else ''
        attrs = element.get('attributes', {}) if element else {}
        field_hint = attrs.get('name', attrs.get('placeholder', attrs.get('aria-label', tag)))

        return {
            'type': 'action_node',
            'interaction_action': {
                'input_text': {
                    'command': locator_cmd,
                    'prompt_instructions': f'Enter {field_hint} value in the field',
                    'input_text': text,
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
                    'prompt_instructions': f'Click on {desc}',
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
                    'url': url,
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
        return {
            'type': 'action_node',
            'interaction_action': {
                'select_option': {
                    'command': locator_cmd,
                    'prompt_instructions': f'Select option: {text}',
                    'select_values': [text],
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


def convert_cache_to_automation(cache_path: str | Path, start_url: str | None = None) -> dict:
    """Read an action cache file and produce a deterministic Optexity automation."""
    with open(cache_path) as f:
        cache = json.load(f)

    url = start_url or cache.get('start_url', '')
    deterministic = cache.get('deterministic_actions_list', [])

    nodes = []
    for action in deterministic:
        node = _action_to_node(action)
        if node:
            nodes.append(node)

    automation = {
        'url': url,
        'parameters': {
            'input_parameters': {},
            'generated_parameters': {},
        },
        'nodes': nodes,
    }
    return automation


def main():
    parser = argparse.ArgumentParser(description='Convert action cache to deterministic automation')
    parser.add_argument('--cache', default='action_cache.json', help='Path to action cache JSON')
    parser.add_argument('--output', default='test_automation_cached.json', help='Output automation JSON path')
    parser.add_argument('--url', default=None, help='Override start URL')
    args = parser.parse_args()

    if not Path(args.cache).exists():
        print(f'Error: Cache file {args.cache} not found', file=sys.stderr)
        sys.exit(1)

    automation = convert_cache_to_automation(args.cache, args.url)

    with open(args.output, 'w') as f:
        json.dump(automation, f, indent=2)

    print(f'Generated {len(automation["nodes"])} deterministic nodes -> {args.output}')

    # Also validate against Optexity schema
    try:
        from optexity.schema.automation import Automation
        Automation.model_validate(automation)
        print('Schema validation: PASSED')
    except Exception as e:
        print(f'Schema validation: FAILED - {e}', file=sys.stderr)


if __name__ == '__main__':
    main()
