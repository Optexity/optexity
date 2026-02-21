import logging
from copy import deepcopy

from optexity.inference.core.run_extraction import handle_llm_extraction
from optexity.inference.infra.browser import Browser
from optexity.schema.actions.assertion_action import AssertionAction, LLMAssertion
from optexity.schema.memory import Memory
from optexity.schema.task import Task

logger = logging.getLogger(__name__)

async def run_assertion_action(
    assertion_action: AssertionAction,
    memory: Memory,
    browser: Browser,
    task: Task,
):
    logger.debug(
        f"---------Running assertion action {assertion_action.model_dump_json()}---------"
    )

    if assertion_action.llm:
        await handle_llm_assertion(assertion_action.llm, memory, browser, task)
    elif assertion_action.network_call:
        raise ValueError("Network call assertions are not supported yet")
        # await handle_network_call_assertion(
        #     assertion_action.network_call, memory, browser
        # )
    elif assertion_action.python_script:
        raise ValueError("Python script assertions are not supported yet")
        # await handle_python_script_assertion(
        #     assertion_action.python_script, memory, browser
        # )


async def handle_llm_assertion(
    llm_assertion: LLMAssertion, memory: Memory, browser: Browser, task: Task
):
    extra_instruction = """You are a helpful assistant that verifies if the condition is met. 
        Use the info supplied below to verify the condition.
        The assertion_reason should be a short explanation of why the condition was met or not met.
        The assertion_result should be True if the condition is met, False otherwise.
        """
    llm_assertion_new = deepcopy(llm_assertion)
    llm_assertion_new.extraction_instructions = (
        extra_instruction + "\n" + llm_assertion_new.extraction_instructions
    )
    output_data = await handle_llm_extraction(llm_assertion_new, memory, browser, task)

    if output_data.json_data["assertion_result"]:
        return True
    else:
        raise AssertionError(
            f"Assertion failed on node {memory.automation_state.step_index}: {output_data.json_data['assertion_reason']}"
        )
