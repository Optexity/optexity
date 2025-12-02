import logging

from optexity.inference.infra.browser import Browser
from optexity.inference.models import GeminiModels, get_llm_model
from optexity.schema.actions.extraction_action import (
    ExtractionAction,
    LLMExtraction,
    NetworkCallExtraction,
    ScreenshotExtraction,
)
from optexity.schema.memory import Memory, OutputData, ScreenshotData

logger = logging.getLogger(__name__)

llm_model = get_llm_model(GeminiModels.GEMINI_2_5_FLASH, True)


async def run_extraction_action(
    extraction_action: ExtractionAction, memory: Memory, browser: Browser
):
    logger.debug(
        f"---------Running extraction action {extraction_action.model_dump_json()}---------"
    )

    if extraction_action.llm:
        await handle_llm_extraction(extraction_action.llm, memory, browser)
    elif extraction_action.network_call:
        await handle_network_call_extraction(
            extraction_action.network_call, memory, browser
        )
    elif extraction_action.screenshot:
        await handle_screenshot_extraction(
            extraction_action.screenshot, memory, browser
        )


async def handle_screenshot_extraction(
    screenshot_extraction: ScreenshotExtraction, memory: Memory, browser: Browser
):

    screenshot_base64 = await browser.get_screenshot(
        full_page=screenshot_extraction.full_page
    )
    if screenshot_base64 is None:
        return

    memory.variables.output_data.append(
        OutputData(
            screenshot=ScreenshotData(
                filename=screenshot_extraction.filename, base64=screenshot_base64
            )
        )
    )


async def handle_llm_extraction(
    llm_extraction: LLMExtraction, memory: Memory, browser: Browser
):
    # TODO: fix this double calling of screenshot and axtree
    if "axtree" in llm_extraction.source:
        axtree = memory.browser_states[-1].axtree
    else:
        axtree = None

    if "screenshot" in llm_extraction.source:
        screenshot = memory.browser_states[-1].screenshot
    else:
        screenshot = None

    system_instruction = f"""
    You are an expert in extracting information from a website. You will be given an axtree of a webpage.
    Your task is to extract the information from the webpage and return it in the format specified by the instructions.
    {llm_extraction.extraction_instructions}
    """

    prompt = f"""
    [INPUT]
    Axtree: {axtree}
    [/INPUT]
    """

    if llm_extraction.llm_provider == "gemini":
        model_name = GeminiModels(llm_extraction.llm_model_name)
        llm_model.model_name = model_name
    else:
        raise ValueError(f"Invalid LLM provider: {llm_extraction.llm_provider}")

    response, token_usage = llm_model.get_model_response_with_structured_output(
        prompt=prompt,
        response_schema=llm_extraction.build_model(),
        screenshot=screenshot,
        system_instruction=system_instruction,
    )
    response_dict = response.model_dump()
    output_data = OutputData(json_data=response_dict)

    logger.debug(f"Response: {response_dict}")

    memory.token_usage += token_usage
    memory.variables.output_data.append(output_data)

    if llm_extraction.output_variable_names is not None:
        for output_variable_name in llm_extraction.output_variable_names:
            if isinstance(response_dict[output_variable_name], list):
                memory.variables.generated_variables[output_variable_name] = (
                    response_dict[output_variable_name]
                )
            elif isinstance(response_dict[output_variable_name], str):
                memory.variables.generated_variables[output_variable_name] = [
                    response_dict[output_variable_name]
                ]
            else:
                raise ValueError(
                    f"Output variable {output_variable_name} must be a string or a list of strings"
                )
    return output_data


async def handle_network_call_extraction(
    network_call_extraction: NetworkCallExtraction, memory: Memory, browser: Browser
):

    for network_call in browser.network_calls:
        if network_call_extraction.url_pattern in network_call.url:
            memory.variables.output_data.append(
                OutputData(json_data=network_call.model_dump())
            )
