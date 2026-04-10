import logging
import re

from pydantic import BaseModel

from optexity.inference.agents.select_value_prediction.select_value_prediction import (
    SelectValuePredictionAgent,
)
from optexity.inference.models import get_llm_model_with_fallback
from optexity.schema.actions.interaction_action import Locator
from optexity.schema.memory import Memory
from optexity.schema.task import Task

logger = logging.getLogger(__name__)

_select_prediction_cache: dict[tuple, SelectValuePredictionAgent] = {}


def _get_select_prediction_agent(task: Task) -> SelectValuePredictionAgent:
    cache_key = (task.llm_provider, task.llm_model_name)
    if cache_key not in _select_prediction_cache:
        model = get_llm_model_with_fallback(
            task.llm_provider, task.llm_model_name, True
        )
        _select_prediction_cache[cache_key] = SelectValuePredictionAgent(model)
    return _select_prediction_cache[cache_key]


class SelectOptionValue(BaseModel):
    value: str
    label: str


def llm_select_match(
    options: list[SelectOptionValue], patterns: list[str], memory: Memory, task: Task
) -> list[str]:
    final_prompt, response, token_usage = _get_select_prediction_agent(
        task
    ).predict_select_value([o.model_dump() for o in options], patterns)
    memory.token_usage += token_usage
    memory.browser_states[-1].final_prompt = final_prompt
    memory.browser_states[-1].llm_response = response.model_dump()

    matched_values = response.matched_values

    all_values = [o.value for o in options]

    final_matched_values = []
    for value in matched_values:
        if value in all_values:
            final_matched_values.append(value)

    return final_matched_values


def score_match(pat: str, val: str) -> int:
    # higher is better
    if pat == val:
        return 100
    if val.startswith(pat):
        return 80
    if pat in val:
        return 60
    return 0


async def smart_select(
    options: list[SelectOptionValue], patterns: list[str], memory: Memory, task: Task
):
    # Get all options from the <select>
    ## TODO: remove this once we have a better way to handle select one
    matched_values = []

    if len(options) == 0:
        return []
    if len(options) == 1:
        return [options[0].value]
    if len(options) == 2 and "Select One" in [o.value for o in options]:
        if options[0].value == "Select One":
            return [options[1].value]
        else:
            return [options[0].value]

    for p in patterns:
        # If pattern contains regex characters, treat as regex
        is_regex = p.startswith("^") or p.endswith("$") or ".*" in p

        ## Check if reggex pattern and then try finding the option by value and label
        if is_regex:
            regex = re.compile(p)
            for opt in options:
                if regex.search(opt.value) or regex.search(opt.label):
                    matched_values.append(opt.value)
        else:
            # try exact match
            for opt in options:
                if opt.value == p or opt.label == p:
                    matched_values.append(opt.value)

    if len(matched_values) == 0:
        ## If no matches, check if all values are unique and try score matching of values

        processed_values = [
            (v.value.lower().replace(" ", ""), v.value) for v in options
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

                if best_value is not None and best_score > 0:
                    matched_values.append(best_value)

    if len(matched_values) == 0:
        processed_labels = [
            (v.label.lower().replace(" ", ""), v.label) for v in options
        ]

        if len(processed_labels) == len(set(processed_labels)):
            for p in patterns:
                processed_pattern = p.lower().replace(" ", "")

                best_score = 0
                best_label = None
                best_value = None

                for opt in options:
                    processed_label = opt.label.lower().replace(" ", "")
                    score = score_match(processed_pattern, processed_label)
                    if score > best_score:
                        best_score = score
                        best_label = opt.label
                        best_value = opt.value

                if best_label is not None and best_score > 0:
                    matched_values.append(best_value)

    if len(matched_values) == 0:
        matched_values = llm_select_match(options, patterns, memory, task)

    if len(matched_values) == 0:
        matched_values = patterns

    return matched_values
