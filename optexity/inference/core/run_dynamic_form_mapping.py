import json
import logging

import httpx

from optexity.exceptions import DynamicFormMappingException
from optexity.inference.core.run_extraction import handle_llm_extraction
from optexity.inference.infra.browser import Browser
from optexity.inference.infra.browser_health import fetch_browser_state_for_classifier
from optexity.schema.actions.dynamic_form_mapping_action import DynamicFormMappingAction
from optexity.schema.actions.extraction_action import LLMExtraction
from optexity.schema.memory import Memory, OutputData
from optexity.schema.task import Task, validate_callback_url_ssrf
from optexity.utils.settings import settings

logger = logging.getLogger(__name__)


def _serialize_form_values(body) -> str:
    if isinstance(body, str):
        return body
    return json.dumps(body, ensure_ascii=False)


def _live_stream_url(task: Task) -> str:
    return f"{settings.FRONTEND_URL.rstrip('/')}/task-logs?task_id={task.task_id}"


async def run_dynamic_form_mapping_action(
    action: DynamicFormMappingAction,
    task: Task,
    memory: Memory,
    browser: Browser,
) -> None:
    logger.debug(
        "---------Running dynamic form mapping action %s---------",
        action.model_dump_json(exclude={"callback_url"}),
    )

    llm_extraction = LLMExtraction(
        source=action.source,
        extraction_format=action.extraction_format,
        extraction_instructions=action.extraction_instructions,
        llm_provider=action.llm_provider,
        llm_model_name=action.llm_model_name,
    )
    extraction_output = await handle_llm_extraction(
        llm_extraction,
        memory,
        browser,
        task,
        unique_identifier="extracted_fields",
    )
    extracted_fields = (
        extraction_output.json_data if extraction_output else None
    ) or {}

    screenshot = None
    if action.include_screenshot:
        screenshot = await browser.get_screenshot(full_page=action.full_page_screenshot)
        if screenshot is None:
            logger.warning(
                "Failed to capture screenshot for dynamic form mapping " "on task %s",
                task.task_id,
            )

    axtree = None
    if action.include_axtree:
        axtree = memory.browser_states[-1].axtree if memory.browser_states else None
        if axtree is None:
            summary = await fetch_browser_state_for_classifier(browser, memory, task)
            if summary is not None and memory.browser_states:
                axtree = memory.browser_states[-1].axtree

    live_stream_url = _live_stream_url(task) if action.include_live_stream_url else None

    payload = {
        "task_id": task.task_id,
        "extracted_fields": extracted_fields,
        "screenshot": screenshot,
        "axtree": axtree,
        "live_stream_url": live_stream_url,
    }

    callback = action.callback_url
    validate_callback_url_ssrf(callback.url)

    headers: dict[str, str] = {}
    auth = None
    if callback.api_key:
        headers["x-api-key"] = callback.api_key
    elif callback.username is not None and callback.password is not None:
        auth = (callback.username, callback.password)

    logger.info(
        "Dynamic form mapping POST to %s for task %s "
        "(include_screenshot=%s, full_page_screenshot=%s, "
        "include_axtree=%s, include_live_stream_url=%s, "
        "extracted_fields_keys=%s, screenshot_bytes=%s)",
        callback.url,
        task.task_id,
        action.include_screenshot,
        action.full_page_screenshot,
        action.include_axtree,
        action.include_live_stream_url,
        (
            list(extracted_fields.keys())
            if isinstance(extracted_fields, dict)
            else type(extracted_fields).__name__
        ),
        len(screenshot) if screenshot else 0,
    )

    try:
        timeout = httpx.Timeout(
            action.max_wait_time, connect=min(10.0, action.max_wait_time)
        )
        async with httpx.AsyncClient(
            timeout=timeout, follow_redirects=True, auth=auth
        ) as client:
            response = await client.post(callback.url, headers=headers, json=payload)
    except httpx.TimeoutException as e:
        raise DynamicFormMappingException(
            f"Dynamic form mapping timed out after {action.max_wait_time}s "
            f"waiting for {callback.url}: {e}"
        ) from e
    except httpx.HTTPError as e:
        raise DynamicFormMappingException(
            f"Dynamic form mapping request to {callback.url} failed: {e}"
        ) from e

    if response.status_code < 200 or response.status_code >= 300:
        body_preview = response.text[:500]
        raise DynamicFormMappingException(
            f"Dynamic form mapping callback returned HTTP {response.status_code}: "
            f"{body_preview}"
        )

    try:
        response_body = response.json()
    except Exception as e:
        raise DynamicFormMappingException(
            f"Dynamic form mapping callback returned non-JSON body: {e}"
        ) from e

    if response_body is None:
        raise DynamicFormMappingException(
            "Dynamic form mapping callback returned JSON null"
        )

    serialized = _serialize_form_values(response_body)
    var_name = action.output_variable_name
    memory.variables.generated_variables[var_name] = [serialized]
    memory.variables.output_data.append(
        OutputData(
            unique_identifier=var_name,
            json_data={var_name: serialized},
        )
    )
    logger.info(
        "Stored dynamic form mapping result in generated_variables[%s] "
        "(%s chars) for task %s",
        var_name,
        len(serialized),
        task.task_id,
    )
