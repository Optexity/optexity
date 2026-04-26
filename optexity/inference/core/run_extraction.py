import base64
import logging
import traceback

import aiofiles
import httpx

from optexity.inference.core.interaction.utils import (
    crop_screenshot_to_bbox,
    find_keyword_in_results,
    get_coordinates_from_ocr_result,
    resolve_bounding_box_variables,
)
from optexity.inference.core.run_two_fa import run_two_fa_action
from optexity.inference.core.vision.ocr.aws_textract import AWSTextract
from optexity.inference.infra.browser import Browser
from optexity.inference.models import (
    GeminiModels,
    get_llm_model,
    get_llm_model_with_fallback,
)
from optexity.schema.actions.extraction_action import (
    ExtractionAction,
    LLMExtraction,
    LocatorExtraction,
    NetworkCallExtraction,
    OCRCoordinatesExtraction,
    PDFExtraction,
    PythonScriptExtraction,
    ScreenshotExtraction,
    StateExtraction,
    VisionExtraction,
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
ocr = AWSTextract()


async def run_extraction_action(
    extraction_action: ExtractionAction, memory: Memory, browser: Browser, task: Task
):
    logger.debug(
        f"---------Running extraction action {extraction_action.model_dump_json()}---------"
    )

    if extraction_action.llm:
        await handle_llm_extraction(
            extraction_action.llm,
            memory,
            browser,
            task,
            extraction_action.unique_identifier,
        )
    elif extraction_action.network_call:
        await handle_network_call_extraction(
            extraction_action.network_call,
            memory,
            browser,
            task,
            extraction_action.unique_identifier,
        )
    elif extraction_action.python_script:
        await handle_python_script_extraction(
            extraction_action.python_script,
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
    elif extraction_action.two_fa_action:
        await run_two_fa_action(extraction_action.two_fa_action, memory, task)
    elif extraction_action.pdf:
        await handle_pdf_extraction(extraction_action.pdf, memory, task)
    elif extraction_action.ocr_coordinates:
        await handle_ocr_coordinates_extraction(
            extraction_action.ocr_coordinates,
            memory,
            browser,
            extraction_action.unique_identifier,
        )
    elif extraction_action.locator:
        await handle_locator_extraction(
            extraction_action.locator,
            memory,
            browser,
            task,
            extraction_action.unique_identifier,
        )
    elif extraction_action.vision:
        await handle_vision_extraction(
            extraction_action.vision,
            memory,
            browser,
            task,
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

    # Get localStorage
    local_storage = await page.evaluate("""() => {
            const items = {};
            for (let i = 0; i < localStorage.length; i++) {
                const key = localStorage.key(i);
                items[key] = localStorage.getItem(key);
            }
            return items;
        }""")

    # Get sessionStorage
    session_storage = await page.evaluate("""() => {
            const items = {};
            for (let i = 0; i < sessionStorage.length; i++) {
                const key = sessionStorage.key(i);
                items[key] = sessionStorage.getItem(key);
            }
            return items;
        }""")

    # Get cookies (both structured and document.cookie)
    cookies = await page.context.cookies()
    document_cookie = await page.evaluate("document.cookie")

    memory.variables.output_data.append(
        OutputData(
            unique_identifier=unique_identifier,
            json_data={
                "page_url": page.url,
                "page_title": await page.title(),
                "local_storage": local_storage,
                "session_storage": session_storage,
                "cookies": cookies,
                "document_cookie": document_cookie,
            },
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
    task: Task,
    unique_identifier: str | None = None,
):
    browser_state = await browser.get_browser_state_summary(
        remove_empty_nodes=task.automation.remove_empty_nodes_in_axtree,
        include_full_page=llm_extraction.include_full_page,
    )
    memory.browser_states[-1] = browser_state

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
    Your task is to extract the information from the webpage and return it in the format specified by the instructions. You will be first provided the instructions and then the axtree.
    Instructions: {llm_extraction.extraction_instructions}
    """

    prompt = f"""
    [INPUT]
    Axtree: {axtree}
    [/INPUT]
    """

    provider = llm_extraction.llm_provider or task.llm_provider
    model_name_str = llm_extraction.llm_model_name or task.llm_model_name
    llm_model = get_llm_model_with_fallback(provider, model_name_str, True)

    response, token_usage = llm_model.get_model_response_with_structured_output(
        prompt=prompt,
        response_schema=llm_extraction.build_model(),
        screenshot=screenshot,
        recording_screenshot=llm_extraction.recording_screenshot,
        system_instruction=system_instruction,
    )
    response_dict = response.model_dump()
    output_data = OutputData(
        unique_identifier=unique_identifier, json_data=response_dict
    )

    logger.debug(f"Response: {response_dict}")

    memory.token_usage += token_usage
    memory.variables.output_data.append(output_data)

    memory.browser_states[-1].final_prompt = f"{system_instruction}\n{prompt}"

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


async def handle_locator_extraction(
    locator_extraction: LocatorExtraction,
    memory: Memory,
    browser: Browser,
    task: Task,
    unique_identifier: str | None = None,
):
    var_name = locator_extraction.output_variable_name
    extracted_value = None
    locator_failed = False

    try:
        locator = await browser.get_locator_from_command(locator_extraction.command)
        if locator is None:
            raise ValueError(
                f"Locator returned None for command: {locator_extraction.command}"
            )
        text = await locator.first.inner_text(timeout=5000)
        if text is None:
            raise ValueError(
                f"No text content found for locator: {locator_extraction.command}"
            )
        extracted_value = text.strip()
        memory.variables.generated_variables[var_name] = [extracted_value]
        logger.debug(f"Locator extracted {var_name}={extracted_value!r}")
    except Exception as e:
        logger.warning(f"Locator extraction failed for {var_name!r}: {e}")
        locator_failed = True
        memory.variables.generated_variables[var_name] = [None]

    if locator_failed:
        if locator_extraction.extraction_instructions is not None:
            try:
                llm_extraction = LLMExtraction(
                    extraction_format=locator_extraction.extraction_format,
                    extraction_instructions=locator_extraction.extraction_instructions,
                    output_variable_names=[var_name],
                    llm_provider=locator_extraction.llm_provider,
                    llm_model_name=locator_extraction.llm_model_name,
                )
                output = await handle_llm_extraction(
                    llm_extraction, memory, browser, task, unique_identifier
                )
                if output is not None and var_name in output.json_data:
                    extracted_value = output.json_data[var_name]
                logger.debug(f"LLM fallback extracted {var_name}={extracted_value!r}")
            except Exception as e:
                logger.warning(f"LLM fallback also failed for {var_name!r}: {e}")
        else:
            logger.warning(
                f"No extraction_instructions for LLM fallback; {var_name} set to None"
            )

    memory.variables.output_data.append(
        OutputData(
            unique_identifier=unique_identifier,
            json_data={var_name: extracted_value},
        )
    )


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


async def handle_python_script_extraction(
    python_script_extraction: PythonScriptExtraction,
    memory: Memory,
    browser: Browser,
    task: Task,
    unique_identifier: str | None = None,
):
    local_vars = {}
    exec(python_script_extraction.script, {}, local_vars)
    code_fn = local_vars["code_fn"]
    axtree = memory.browser_states[-1].axtree
    result = await code_fn(axtree, browser)
    if result is not None:
        memory.variables.output_data.append(
            OutputData(
                unique_identifier=unique_identifier,
                json_data=result,
            )
        )
    else:
        logger.warning(
            f"No result from Python script extraction: {python_script_extraction.script}"
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


async def handle_pdf_extraction(
    pdf_extraction: PDFExtraction, memory: Memory, task: Task
):
    """
    Expects the PDF file to be in the downloads directory and the filename to be the same as the one specified in the PDFExtraction schema.
    If the PDF file is not found, it will use the first PDF file in the downloads directory.
    If there are multiple PDF files in the downloads directory, it will raise an error.
    TODO: handle multiple PDF files in the downloads directory.
    """
    pdf_file = None
    for path in memory.downloads:
        if pdf_extraction.filename in path.name:
            pdf_file = path
            break
    if pdf_file is None:
        if len(memory.downloads) == 1:
            pdf_file = memory.downloads[0]
        else:
            logger.error(
                f"No matching PDF file found in downloads with filename {pdf_extraction.filename}. Total downloads: {len(memory.downloads)}"
            )
            return None

    provider = pdf_extraction.llm_provider or task.llm_provider
    model_name_str = pdf_extraction.llm_model_name or task.llm_model_name
    llm_model = get_llm_model_with_fallback(provider, model_name_str, True)

    system_instruction = "Extract the information from the PDF file and return it in the format specified by the instructions."
    response, token_usage = llm_model.get_model_response_with_structured_output(
        prompt=pdf_extraction.extraction_instructions,
        response_schema=pdf_extraction.build_model(),
        pdf_url=pdf_file,
        system_instruction=system_instruction,
    )
    response_dict = response.model_dump()
    output_data = OutputData(
        unique_identifier=str(pdf_file.name), json_data=response_dict
    )

    logger.debug(f"Response: {response_dict}")

    memory.token_usage += token_usage
    memory.variables.output_data.append(output_data)

    memory.browser_states[-1].final_prompt = (
        f"{system_instruction}\n{pdf_extraction.extraction_instructions}"
    )

    return output_data


async def handle_ocr_coordinates_extraction(
    ocr_extraction: OCRCoordinatesExtraction,
    memory: Memory,
    browser: "Browser",
    unique_identifier: str | None = None,
):
    """Run OCR once on the screenshot, match all names from source_variable, store x/y as parallel lists.

    - Duplicate names: each gets a different OCR result (position order).
    - Score < 80: stores -1, -1 (interaction action will fallback to full OCR + LLM).
    - bounding_box_variables: crops screenshot before OCR; offsets results back to full-screen coords.
    """

    source_var = ocr_extraction.source_variable
    if source_var in memory.variables.generated_variables:
        names = memory.variables.generated_variables[source_var]
    else:
        logger.error(f"Source variable '{source_var}' not found in generated variables")
        return

    screenshot = await browser.get_screenshot()
    if screenshot is None:
        logger.error("Screenshot is None, cannot run OCR")
        return

    # Apply bounding box crop if specified
    crop_offset_x, crop_offset_y = 0, 0
    ocr_screenshot = screenshot
    if ocr_extraction.bounding_box_variables:
        bbox = resolve_bounding_box_variables(
            ocr_extraction.bounding_box_variables, memory
        )
        if bbox:
            crop_offset_x, crop_offset_y = bbox[0], bbox[1]
            ocr_screenshot = crop_screenshot_to_bbox(screenshot, *bbox)

    results, base64_image_sent_to_ocr = ocr.ocr(ocr_screenshot)

    # Offset bounding boxes back to full-screen coordinates
    if crop_offset_x or crop_offset_y:
        for r in results:
            r.bounding_box.x += crop_offset_x
            r.bounding_box.y += crop_offset_y

    annotated, canvas = ocr.visualize(screenshot, results)
    memory.browser_states[-1].ocr_annotated = base64.b64encode(annotated).decode(
        "utf-8"
    )
    memory.browser_states[-1].ocr_canvas = base64.b64encode(canvas).decode("utf-8")
    memory.browser_states[-1].ocr_image_sent_to_ocr.append(base64_image_sent_to_ocr)

    names_str = [str(n) for n in names]
    coords_x: list[int] = []
    coords_y: list[int] = []
    used_result_ids: set[int] = set()

    for name in names_str:
        available = [r for r in results if id(r) not in used_result_ids]
        result = find_keyword_in_results(available, name, score_threshold=80.0)
        if result is not None:
            used_result_ids.add(id(result))
            x, y = get_coordinates_from_ocr_result(result)
            coords_x.append(x)
            coords_y.append(y)
            logger.info(f"OCR matched '{name}' at ({x}, {y})")
        else:
            coords_x.append(-1)
            coords_y.append(-1)
            logger.warning(f"OCR score below threshold for '{name}', storing -1,-1")

    memory.variables.generated_variables[ocr_extraction.output_x_variable] = coords_x
    memory.variables.generated_variables[ocr_extraction.output_y_variable] = coords_y

    result_data = {name: [coords_x[i], coords_y[i]] for i, name in enumerate(names_str)}
    memory.variables.output_data.append(
        OutputData(unique_identifier=unique_identifier, json_data=result_data)
    )


async def handle_vision_extraction(
    vision_extraction: VisionExtraction,
    memory: Memory,
    browser: "Browser",
    task: "Task",
    unique_identifier: str | None = None,
):
    """Use computer use model to locate a UI element and store its (x, y) coordinates.

    Stores -1, -1 if the model cannot locate the element.
    """
    screenshot = await browser.get_screenshot()
    if screenshot is None:
        logger.error("[vision_extraction] screenshot is None")
        memory.variables.generated_variables[vision_extraction.output_x_variable] = [-1]
        memory.variables.generated_variables[vision_extraction.output_y_variable] = [-1]
        return

    memory.browser_states[-1].computer_use_screenshots.append(screenshot)
    model = get_llm_model(GeminiModels.GEMINI_2_5_COMPUTER_USE, True)

    x, y = -1, -1
    try:
        coordinates, token_usage = model.get_computer_use_model_response(
            prompt=vision_extraction.prompt,
            screenshot=screenshot,
        )
        memory.token_usage += token_usage
        if coordinates:
            x, y = coordinates[0], coordinates[1]
    except Exception as e:
        logger.error(f"[vision_extraction] computer use model failed: {e}")

    logger.info(f"[vision_extraction] '{vision_extraction.prompt}' → ({x}, {y})")
    memory.variables.generated_variables[vision_extraction.output_x_variable] = [x]
    memory.variables.generated_variables[vision_extraction.output_y_variable] = [y]
    memory.variables.output_data.append(
        OutputData(
            unique_identifier=unique_identifier,
            json_data={
                vision_extraction.output_x_variable: x,
                vision_extraction.output_y_variable: y,
            },
        )
    )
