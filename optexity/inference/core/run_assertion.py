import logging

from optexity.inference.core.run_extraction import handle_llm_extraction
from optexity.inference.infra.browser import Browser
from optexity.inference.models import GeminiModels, get_llm_model
from optexity.schema.actions.assertion_action import AssertionAction, LLMAssertion
from optexity.schema.memory import Memory

logger = logging.getLogger(__name__)

llm_model = get_llm_model(GeminiModels.GEMINI_2_5_FLASH, True)


async def run_assertion_action(
    assertion_action: AssertionAction, memory: Memory, browser: Browser
):
    logger.debug(
        f"---------Running assertion action {assertion_action.model_dump_json()}---------"
    )

    if assertion_action.llm:
        await handle_llm_assertion(assertion_action.llm, memory, browser)
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
    llm_assertion: LLMAssertion, memory: Memory, browser: Browser
):
    output_data = await handle_llm_extraction(llm_assertion, memory, browser)
    if output_data.json_data["assertion_result"]:
        return True
    else:
        raise AssertionError(
            f"Assertion failed on node {memory.automation_state.step_index}: {output_data.json_data['assertion_reason']}"
        )
