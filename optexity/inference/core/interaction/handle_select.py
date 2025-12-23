import logging
import re

from optexity.inference.core.interaction.utils import (
    command_based_action_with_retry,
    get_index_from_prompt,
    handle_download,
)
from optexity.inference.infra.browser import Browser
from optexity.schema.actions.interaction_action import Locator, SelectOptionAction
from optexity.schema.memory import Memory
from optexity.schema.task import Task

logger = logging.getLogger(__name__)


async def handle_select_option(
    select_option_action: SelectOptionAction,
    task: Task,
    memory: Memory,
    browser: Browser,
    max_timeout_seconds_per_try: float,
    max_tries: int,
):

    if select_option_action.command:
        last_error = await command_based_action_with_retry(
            lambda: select_option_locator(
                select_option_action, browser, max_timeout_seconds_per_try, memory, task
            ),
            select_option_action.command,
            max_tries,
            max_timeout_seconds_per_try,
            select_option_action.assert_locator_presence,
        )

        if last_error is None:
            return

    if not select_option_action.skip_prompt:
        await select_option_index(select_option_action, browser, memory, task)


def score_match(pat: str, val: str) -> int:
    # higher is better
    if pat == val:
        return 100
    if val.startswith(pat):
        return 80
    if pat in val:
        return 60
    return 0


async def smart_select(locator: Locator, patterns: list[str]):
    # Get all options from the <select>
    options: list[dict[str, str]] = await locator.evaluate(
        """
        sel => Array.from(sel.options).map(o => ({
            value: o.value,
            label: o.label || o.textContent
        }))
    """
    )

    matched_values = []

    for p in patterns:
        # If pattern contains regex characters, treat as regex
        is_regex = p.startswith("^") or p.endswith("$") or ".*" in p

        ## Check if reggex pattern and then try finding the option by value and label
        if is_regex:
            regex = re.compile(p)
            for opt in options:
                if regex.search(opt["value"]) or regex.search(opt["label"]):
                    matched_values.append(opt["value"])
        else:
            # try exact match
            for opt in options:
                if opt["value"] == p or opt["label"] == p:
                    matched_values.append(opt["value"])

    if len(matched_values) == 0:
        ## If no matches, check if all values are unique

        processed_values = [
            (v["value"].lower().replace(" ", ""), v["value"]) for v in options
        ]

        if len(processed_values) == len(set(processed_values)):
            for p in patterns:
                processed_pattern = p.lower().replace(" ", "")

                best_score = 0
                best_value = None

                for processed_value, value in processed_values:
                    score = score_match(processed_pattern, processed_value)
                    if score > best_score:
                        best_score = score
                        best_value = value

                if best_value:
                    matched_values.append(best_value)

    if len(matched_values) == 0:
        matched_values = patterns

    return matched_values


async def select_option_locator(
    select_option_action: SelectOptionAction,
    browser: Browser,
    max_timeout_seconds_per_try: float,
    memory: Memory,
    task: Task,
):
    async def _actual_select_option():
        locator = await browser.get_locator_from_command(select_option_action.command)

        matched_values = await smart_select(locator, select_option_action.select_values)

        logger.debug(
            f"Matched values for {select_option_action.command}: {matched_values}"
        )

        await locator.select_option(
            matched_values,
            no_wait_after=True,
            timeout=max_timeout_seconds_per_try * 1000,
        )

    if select_option_action.expect_download:
        await handle_download(
            _actual_select_option,
            memory,
            browser,
            task,
            select_option_action.download_filename,
        )
    else:
        await _actual_select_option()


async def select_option_index(
    select_option_action: SelectOptionAction,
    browser: Browser,
    memory: Memory,
    task: Task,
):

    try:
        index = await get_index_from_prompt(
            memory, select_option_action.prompt_instructions, browser
        )
        if index is None:
            return

        async def _actual_select_option():
            action_model = browser.backend_agent.ActionModel(
                **{
                    "select_dropdown": {
                        "index": int(index),
                        "text": select_option_action.select_values[0],
                    }
                }
            )
            await browser.backend_agent.multi_act([action_model])

        if select_option_action.expect_download:
            await handle_download(
                _actual_select_option,
                memory,
                browser,
                task,
                select_option_action.download_filename,
            )
        else:
            await _actual_select_option()
    except Exception as e:
        logger.error(f"Error in select_option_index: {e}")
        return
