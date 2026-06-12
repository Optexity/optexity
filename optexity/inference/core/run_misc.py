import asyncio
import logging
import traceback

from optexity.inference.infra.browser import Browser
from optexity.inference.models import get_llm_model_with_fallback
from optexity.schema.actions.misc_action import (
    FailStateAction,
    LLMQueryAction,
    SetVariableAction,
    SleepAction,
)
from optexity.schema.memory import Memory, OutputData
from optexity.schema.task import Task

logger = logging.getLogger(__name__)


async def run_sleep_action(sleep_action: SleepAction):
    logger.debug(
        f"---------Running sleep action {sleep_action.model_dump_json()}---------"
    )
    await asyncio.sleep(sleep_action.sleep_time)


async def run_fail_state_action(
    fail_state_action: FailStateAction, memory: Memory, browser: Browser, task: Task
):
    logger.debug(
        f"---------Running fail state action {fail_state_action.model_dump_json()}---------"
    )
    raise Exception(fail_state_action.failure_message)


async def run_set_variable_action(
    set_variable_action: SetVariableAction,
    memory: Memory,
):
    logger.debug(
        f"---------Running set_variable action {set_variable_action.model_dump_json()}---------"
    )
    name = set_variable_action.name
    if set_variable_action.value is not None:
        memory.variables.generated_variables[name] = [set_variable_action.value]
    else:
        result = eval(set_variable_action.expression)  # noqa: S307
        memory.variables.generated_variables[name] = [result]
    logger.debug(
        f"Set variable '{name}' = {memory.variables.generated_variables[name]}"
    )


async def run_llm_query_action(
    llm_query_action: LLMQueryAction,
    memory: Memory,
    task: Task,
    unique_identifier: str | None = None,
):
    logger.debug(
        f"---------Running LLM query action {llm_query_action.model_dump_json()}---------"
    )

    system_instruction = (
        "You are a helpful assistant. Follow the instructions and return your answer "
        "in the structured format requested."
    )

    provider = llm_query_action.llm_provider or task.llm_provider
    model_name_str = llm_query_action.llm_model_name or task.llm_model_name

    try:
        llm_model = get_llm_model_with_fallback(provider, model_name_str, True)
    except Exception as e:
        logger.error(
            f"Failed to initialise LLM model (provider={provider}, model={model_name_str}): {e}\n"
            f"{traceback.format_exc()}"
        )
        raise

    try:
        response, token_usage = llm_model.get_model_response_with_structured_output(
            prompt=llm_query_action.prompt_instructions,
            response_schema=llm_query_action.build_model(),
            system_instruction=system_instruction,
        )
    except Exception as e:
        logger.error(
            f"LLM query inference failed: {e}\n"
            f"prompt_instructions: {llm_query_action.prompt_instructions}\n"
            f"{traceback.format_exc()}"
        )
        raise

    response_dict = response.model_dump()
    memory.token_usage += token_usage

    logger.debug(f"LLM query response: {response_dict}")

    output_data = OutputData(
        unique_identifier=unique_identifier, json_data=response_dict
    )
    memory.variables.output_data.append(output_data)

    if llm_query_action.output_variable_names is not None:
        for output_variable_name in llm_query_action.output_variable_names:
            v = response_dict.get(output_variable_name)
            if v is None:
                logger.warning(
                    f"Output variable '{output_variable_name}' is None in LLM query response"
                )
                memory.variables.generated_variables[output_variable_name] = [None]
            elif isinstance(v, list):
                memory.variables.generated_variables[output_variable_name] = v
            elif isinstance(v, (str, int, float, bool)):
                memory.variables.generated_variables[output_variable_name] = [v]
            else:
                raise ValueError(
                    f"Output variable '{output_variable_name}' must be a string, int, float, bool, "
                    f"or a list thereof. Got: {type(v).__name__} = {v!r}"
                )

    logger.debug(
        f"---------Finished LLM query action (unique_identifier={unique_identifier})---------"
    )
    return output_data
