import base64
import logging

import aiofiles

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
    page = await browser.get_current_page()
    if page is None:
        return

    screenshot_bytes = await page.screenshot(full_page=True)
    screenshot_base64 = base64.b64encode(screenshot_bytes)

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

    prompt = "Extract the following from the axtree: " + axtree
    response, token_usage = llm_model.get_model_response_with_structured_output(
        prompt=prompt,
        response_schema=llm_extraction.build_model(),
        screenshot=screenshot,
    )
    response_dict = response.model_dump()

    logger.debug(f"Response: {response_dict}")

    memory.token_usage += token_usage
    memory.variables.output_data.append(OutputData(dict_data=response_dict))

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


async def handle_network_call_extraction(
    network_call_extraction: NetworkCallExtraction, memory: Memory, browser: Browser
):

    for network_call in browser.network_calls:
        if network_call_extraction.url_pattern in network_call.url:
            memory.variables.output_data.append(
                OutputData(dict_data=network_call.model_dump())
            )
