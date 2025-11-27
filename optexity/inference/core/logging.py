import base64
import io
import json
import logging
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import aiofiles
import httpx

from optexity.schema.automation import ActionNode
from optexity.schema.memory import Memory
from optexity.schema.task import Task
from optexity.schema.token_usage import TokenUsage
from optexity.utils.settings import settings
from optexity.utils.utils import save_screenshot

logger = logging.getLogger(__name__)


def create_tar_in_memory(directory: Path | str, name: str) -> io.BytesIO:
    if isinstance(directory, str):
        directory = Path(directory)
    tar_bytes = io.BytesIO()
    with tarfile.open(fileobj=tar_bytes, mode="w:gz") as tar:
        tar.add(directory, arcname=name)
    tar_bytes.seek(0)  # rewind to start
    return tar_bytes


async def create_task_in_server(task: Task):
    try:
        url = urljoin(settings.SERVER_URL, settings.CREATE_TASK_ENDPOINT)
        headers = {"x-api-key": settings.API_KEY}
        body = {
            "task_id": task.task_id,
            "recording_id": task.recording_id,
            "input_parameters": task.input_parameters,
            "unique_parameters": task.unique_parameters,
            "created_at": task.created_at.isoformat(),
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url,
                headers=headers,
                json=body,
            )

            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        raise ValueError(
            f"Failed to create task in server: {e.response.status_code} - {e.response.text}"
        )
    except Exception as e:
        raise ValueError(f"Failed to create task in server: {e}")


async def start_task_in_server(task: Task):
    try:
        task.started_at = datetime.now(timezone.utc)
        task.status = "running"

        url = urljoin(settings.SERVER_URL, settings.START_TASK_ENDPOINT)
        headers = {"x-api-key": settings.API_KEY}
        body = {
            "task_id": task.task_id,
            "started_at": task.started_at.isoformat(),
        }
        if task.allocated_at:
            body["allocated_at"] = task.allocated_at.isoformat()
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url,
                headers=headers,
                json=body,
            )

            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        raise ValueError(
            f"Failed to start task in server: {e.response.status_code} - {e.response.text}"
        )
    except Exception as e:
        raise ValueError(f"Failed to start task in server: {e}")


async def complete_task_in_server(
    task: Task, token_usage: TokenUsage, child_process_id: int
):
    try:
        task.completed_at = datetime.now(timezone.utc)

        url = urljoin(settings.SERVER_URL, settings.COMPLETE_TASK_ENDPOINT)
        headers = {"x-api-key": settings.API_KEY}
        body = {
            "task_id": task.task_id,
            "child_process_id": child_process_id,
            "completed_at": task.completed_at.isoformat(),
            "status": task.status,
            "error": task.error,
            "token_usage": token_usage.model_dump(),
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url,
                headers=headers,
                json=body,
            )

            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(
            f"Failed to complete task in server: {e.response.status_code} - {e.response.text}"
        )

    except Exception as e:
        logger.error(f"Failed to complete task in server: {e}")


async def save_output_data_in_server(task: Task, memory: Memory):
    try:
        if len(memory.variables.output_data) == 0 and memory.final_screenshot is None:
            return

        url = urljoin(settings.SERVER_URL, settings.SAVE_OUTPUT_DATA_ENDPOINT)
        headers = {"x-api-key": settings.API_KEY}

        output_data = [
            output_data.model_dump(exclude_none=True, exclude={"screenshot"})
            for output_data in memory.variables.output_data
        ]
        body = {
            "task_id": task.task_id,
            "output_data": output_data,
            "final_screenshot": memory.final_screenshot,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url,
                headers=headers,
                json=body,
            )

            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(
            f"Failed to save output data in server: {e.response.status_code} - {e.response.text}"
        )
    except Exception as e:
        logger.error(f"Failed to save output data in server: {e}")


async def save_downloads_in_server(task: Task, memory: Memory):
    try:
        if len(memory.downloads) == 0:
            return

        url = urljoin(settings.SERVER_URL, settings.SAVE_DOWNLOADS_ENDPOINT)
        headers = {"x-api-key": settings.API_KEY}

        data = {
            "task_id": task.task_id,  # form field
        }

        tar_bytes = create_tar_in_memory(task.downloads_directory, task.task_id)
        files = {
            "compressed_downloads": (
                f"{task.task_id}.tar.gz",
                tar_bytes,
                "application/gzip",
            )
        }
        async with httpx.AsyncClient() as client:

            response = await client.post(url, headers=headers, data=data, files=files)

            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(
            f"Failed to save downloads in server: {e.response.status_code} - {e.response.text}"
        )
    except Exception as e:
        logger.error(f"Failed to save downloads in server: {e}")


async def save_trajectory_in_server(task: Task, memory: Memory):
    try:
        url = urljoin(settings.SERVER_URL, settings.SAVE_TRAJECTORY_ENDPOINT)
        headers = {"x-api-key": settings.API_KEY}

        data = {
            "task_id": task.task_id,  # form field
        }

        tar_bytes = create_tar_in_memory(task.task_directory, task.task_id)
        files = {
            "compressed_trajectory": (
                f"{task.task_id}.tar.gz",
                tar_bytes,
                "application/gzip",
            )
        }
        async with httpx.AsyncClient() as client:

            response = await client.post(url, headers=headers, data=data, files=files)

            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(
            f"Failed to save trajectory in server: {e.response.status_code} - {e.response.text}"
        )
    except Exception as e:
        logger.error(f"Failed to save trajectory in server: {e}")


async def save_memory_state_locally(
    task: Task, memory: Memory, node: ActionNode | None
):

    browser_state = memory.browser_states[-1]
    automation_state = memory.automation_state
    step_directory = task.logs_directory / f"step_{str(automation_state.step_index)}"
    step_directory.mkdir(parents=True, exist_ok=True)

    save_screenshot(browser_state.screenshot, step_directory / "screenshot.png")

    state_dict = {
        "title": browser_state.title,
        "url": browser_state.url,
        "step_index": automation_state.step_index,
        "try_index": automation_state.try_index,
        "downloaded_files": [
            downloaded_file.name for downloaded_file in memory.downloads
        ],
        "token_usage": memory.token_usage.model_dump(),
    }

    async with aiofiles.open(step_directory / "state.json", "w") as f:
        await f.write(json.dumps(state_dict, indent=4))

    if browser_state.axtree:
        async with aiofiles.open(step_directory / "axtree.txt", "w") as f:
            await f.write(browser_state.axtree)

    if browser_state.final_prompt:
        async with aiofiles.open(step_directory / "final_prompt.txt", "w") as f:
            await f.write(browser_state.final_prompt)

    if browser_state.llm_response:
        async with aiofiles.open(step_directory / "llm_response.json", "w") as f:
            await f.write(json.dumps(browser_state.llm_response, indent=4))

    if node:
        async with aiofiles.open(step_directory / "action_node.json", "w") as f:
            await f.write(json.dumps(node.model_dump(), indent=4))

    async with aiofiles.open(step_directory / "input_variables.json", "w") as f:
        await f.write(json.dumps(memory.variables.input_variables, indent=4))

    async with aiofiles.open(step_directory / "generated_variables.json", "w") as f:
        await f.write(json.dumps(memory.variables.generated_variables, indent=4))

    async with aiofiles.open(step_directory / "output_data.json", "w") as f:
        await f.write(
            json.dumps(
                [
                    output_data.model_dump(exclude_none=True, exclude={"screenshot"})
                    for output_data in memory.variables.output_data
                ],
                indent=4,
            )
        )

    for output_data in memory.variables.output_data:
        if output_data.screenshot:
            async with aiofiles.open(
                step_directory / f"screenshot_{output_data.screenshot.filename}.png",
                "wb",
            ) as f:
                await f.write(base64.b64decode(output_data.screenshot.base64))
