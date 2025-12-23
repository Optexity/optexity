import logging
import traceback

import aiofiles
import httpx

from optexity.inference.infra.browser import Browser
from optexity.inference.models import GeminiModels, get_llm_model
from optexity.schema.actions.extraction_action import (
    ExtractionAction,
    LLMExtraction,
    NetworkCallExtraction,
    ScreenshotExtraction,
    StateExtraction,
)
from optexity.schema.memory import (
    Memory,
    NetworkRequest,
    NetworkResponse,
    OutputData,
    ScreenshotData,
)
from optexity.schema.task import Task

logger = logging.getLogger(__name__)

llm_model = get_llm_model(GeminiModels.GEMINI_2_5_FLASH, True)


async def run_extraction_action(
    extraction_action: ExtractionAction, memory: Memory, browser: Browser, task: Task
):
    logger.debug(
        f"---------Running extraction action {extraction_action.model_dump_json()}---------"
    )

    if extraction_action.llm:
        await handle_llm_extraction(
            extraction_action.llm, memory, browser, extraction_action.unique_identifier
        )
    elif extraction_action.network_call:
        await handle_network_call_extraction(
            extraction_action.network_call,
            memory,
            browser,
            task,
            extraction_action.unique_identifier,
        )
    elif extraction_action.screenshot:
        await handle_screenshot_extraction(
            extraction_action.screenshot,
            memory,
            browser,
            extraction_action.unique_identifier,
        )
    elif extraction_action.state:
        await handle_state_extraction(
            extraction_action.state,
            memory,
            browser,
            extraction_action.unique_identifier,
        )


async def handle_state_extraction(
    state_extraction: StateExtraction,
    memory: Memory,
    browser: Browser,
    unique_identifier: str | None = None,
):
    page = await browser.get_current_page()
    if page is None:
        return

    memory.variables.output_data.append(
        OutputData(
            unique_identifier=unique_identifier,
            json_data={"page_url": page.url, "page_title": await page.title()},
        )
    )


async def handle_screenshot_extraction(
    screenshot_extraction: ScreenshotExtraction,
    memory: Memory,
    browser: Browser,
    unique_identifier: str | None = None,
):

    screenshot_base64 = await browser.get_screenshot(
        full_page=screenshot_extraction.full_page
    )
    if screenshot_base64 is None:
        return

    memory.variables.output_data.append(
        OutputData(
            unique_identifier=unique_identifier,
            screenshot=ScreenshotData(
                filename=screenshot_extraction.filename, base64=screenshot_base64
            ),
        )
    )


async def handle_llm_extraction(
    llm_extraction: LLMExtraction,
    memory: Memory,
    browser: Browser,
    unique_identifier: str | None = None,
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
    output_data = OutputData(
        unique_identifier=unique_identifier, json_data=response_dict
    )

    logger.debug(f"Response: {response_dict}")

    memory.token_usage += token_usage
    memory.variables.output_data.append(output_data)

    if llm_extraction.output_variable_names is not None:
        for output_variable_name in llm_extraction.output_variable_names:
            v = response_dict[output_variable_name]
            if isinstance(v, list):
                memory.variables.generated_variables[output_variable_name] = v
            elif (
                isinstance(v, str)
                or isinstance(v, int)
                or isinstance(v, float)
                or isinstance(v, bool)
            ):
                memory.variables.generated_variables[output_variable_name] = [v]
            else:
                raise ValueError(
                    f"Output variable {output_variable_name} must be a string, int, float, bool, or a list of strings, ints, floats, or bools. Extracted values: {response_dict[output_variable_name]}"
                )
    return output_data


async def handle_network_call_extraction(
    network_call_extraction: NetworkCallExtraction,
    memory: Memory,
    browser: Browser,
    task: Task,
    unique_identifier: str | None = None,
):

    for network_call in browser.network_calls:
        if network_call_extraction.url_pattern not in network_call.url:
            continue

        if network_call_extraction.download_from == "request" and isinstance(
            network_call, NetworkRequest
        ):
            await download_request(
                network_call, network_call_extraction.download_filename, task, memory
            )

        if (
            network_call_extraction.extract_from == "request"
            and isinstance(network_call, NetworkRequest)
        ) or (
            network_call_extraction.extract_from == "response"
            and isinstance(network_call, NetworkResponse)
        ):
            memory.variables.output_data.append(
                OutputData(
                    unique_identifier=unique_identifier,
                    json_data=network_call.model_dump(include={"body"}),
                )
            )


async def download_request(
    network_call: NetworkRequest, download_filename: str, task: Task, memory: Memory
):
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.request(
                network_call.method,
                network_call.url,
                headers=network_call.headers,
                content=network_call.body,  # not data=
            )

            response.raise_for_status()

        # Save raw response to PDF
        download_path = task.downloads_directory / download_filename
        async with aiofiles.open(download_path, "wb") as f:
            await f.write(response.content)

        memory.downloads.append(download_path)
    except Exception as e:
        logger.error(f"Failed to download request: {e}, {traceback.format_exc()}")
