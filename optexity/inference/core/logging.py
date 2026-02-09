import base64
import io
import json
import logging
import shutil
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


async def start_task_in_server(task: Task):
    try:
        task.started_at = datetime.now(timezone.utc)
        task.status = "running"

        url = urljoin(settings.SERVER_URL, settings.START_TASK_ENDPOINT)
        headers = {"x-api-key": task.api_key}
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
        headers = {"x-api-key": task.api_key}
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
        headers = {"x-api-key": task.api_key}

        output_data = [
            output_data.model_dump(exclude_none=True, exclude={"screenshot"})
            for output_data in memory.variables.output_data
        ]
        output_data = [data for data in output_data if data and len(data.keys()) > 0]
        body = {
            "task_id": task.task_id,
            "output_data": output_data,
            "final_screenshot": memory.final_screenshot,
            "unique_child_arn": memory.unique_child_arn,
            "system_info": [
                system_info.model_dump(mode="json")
                for system_info in memory.system_info_tracking
            ],
        }

        for_loop_status = []
        for loop_status in memory.variables.for_loop_status:
            loop_status = [item.model_dump(exclude_none=True) for item in loop_status]
            for_loop_status.append(loop_status)

        if len(for_loop_status) > 0:
            body["for_loop_status"] = for_loop_status

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
        # if len(memory.downloads) == 0:
        #     return

        url = urljoin(settings.SERVER_URL, settings.SAVE_DOWNLOADS_ENDPOINT)
        headers = {"x-api-key": task.api_key}

        payload = {
            "task_id": task.task_id,  # form field
        }

        files = []
        downloads = [
            download
            for download in task.downloads_directory.iterdir()
            if download.is_file()
        ]
        if len(downloads) > 0:
            tar_bytes = create_tar_in_memory(task.downloads_directory, task.task_id)
            # add tar.gz
            files.append(
                (
                    "compressed_downloads",
                    (f"{task.task_id}.tar.gz", tar_bytes, "application/gzip"),
                )
            )

        # add screenshots
        for data in memory.variables.output_data:
            if data.screenshot:
                files.append(
                    (
                        "screenshots",
                        (
                            data.screenshot.filename,
                            base64.b64decode(data.screenshot.base64),
                            "image/png",
                        ),
                    )
                )

        if memory.final_screenshot:
            files.append(
                (
                    "screenshots",
                    (
                        "final_screenshot.png",
                        base64.b64decode(memory.final_screenshot),
                        "image/png",
                    ),
                )
            )

        if len(files) == 0:
            return

        async with httpx.AsyncClient(timeout=30.0) as client:

            response = await client.post(
                url, headers=headers, data=payload, files=files
            )

            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(
            f"Failed to save downloads in server: {e.response.status_code} - {e.response.text}"
        )
    except Exception as e:
        logger.error(f"Failed to save downloads in server: {e}")


async def save_trajectory_in_server(task: Task):
    try:
        url = urljoin(settings.SERVER_URL, settings.SAVE_TRAJECTORY_ENDPOINT)
        headers = {"x-api-key": task.api_key}

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
        async with httpx.AsyncClient(timeout=30.0) as client:

            response = await client.post(url, headers=headers, data=data, files=files)

            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(
            f"Failed to save trajectory in server: {e.response.status_code} - {e.response.text}"
        )
    except Exception as e:
        logger.error(f"Failed to save trajectory in server: {e}")


async def initiate_callback(task: Task):

    if settings.DEPLOYMENT == "dev" and settings.LOCAL_CALLBACK_URL is not None:
        logger.info("initiating local callback")
        callback_data = None
        try:
            url = urljoin(settings.SERVER_URL, settings.GET_CALLBACK_DATA_ENDPOINT)
            headers = {"x-api-key": task.api_key}
            data = {
                "task_id": task.task_id,
                "endpoint_name": task.endpoint_name,
            }
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, headers=headers, json=data)
                response.raise_for_status()
                callback_data = response.json()["data"]
        except Exception as e:
            logger.error(f"Failed to get callback data: {e}")
            return

        if callback_data is None:
            return

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    settings.LOCAL_CALLBACK_URL, json=callback_data
                )
                response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to initiate local callback: {e}")
            return

        return

    try:
        logger.info("initiating callback")
        if task.callback_url is None:
            return

        url = urljoin(settings.SERVER_URL, settings.INITIATE_CALLBACK_ENDPOINT)
        headers = {"x-api-key": task.api_key}

        data = {
            "task_id": task.task_id,
            "endpoint_name": task.endpoint_name,
            "callback_url": task.callback_url.model_dump(),
        }

        async with httpx.AsyncClient(timeout=30.0) as client:

            response = await client.post(url, headers=headers, json=data)

            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(
            f"Failed to save trajectory in server: {e.response.status_code} - {e.response.text}"
        )
    except Exception as e:
        logger.error(f"Failed to save trajectory in server: {e}")


async def save_latest_memory_state_locally(
    task: Task, memory: Memory, node: ActionNode | None
):

    try:
        browser_state = memory.browser_states[-1]
        automation_state = memory.automation_state
        step_directory = (
            task.logs_directory / f"step_{str(automation_state.step_index)}"
        )
        step_directory.mkdir(parents=True, exist_ok=True)

        if browser_state.screenshot:
            save_screenshot(browser_state.screenshot, step_directory / "screenshot.png")
        else:
            logger.warning(
                "No screenshot found for step %s", automation_state.step_index
            )

        state_dict = {
            "title": browser_state.title,
            "url": browser_state.url,
            "step_index": automation_state.step_index,
            "try_index": automation_state.try_index,
            "downloaded_files": [
                downloaded_file.name for downloaded_file in memory.downloads
            ],
            "token_usage": memory.token_usage.model_dump(),
            "unique_child_arn": memory.unique_child_arn,
            "system_info": browser_state.system_info.model_dump(mode="json"),
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
                await f.write(
                    json.dumps(
                        node.model_dump(exclude_none=True, exclude_defaults=True),
                        indent=4,
                    )
                )

        async with aiofiles.open(step_directory / "input_parameters.json", "w") as f:
            await f.write(json.dumps(task.input_parameters, indent=4))

        async with aiofiles.open(step_directory / "secure_parameters.json", "w") as f:
            secure_parameters = {
                key: [
                    a.model_dump(exclude_none=True, exclude_defaults=True)
                    for a in value
                ]
                for key, value in task.secure_parameters.items()
            }
            await f.write(json.dumps(secure_parameters, indent=4))

        async with aiofiles.open(step_directory / "generated_variables.json", "w") as f:
            await f.write(json.dumps(memory.variables.generated_variables, indent=4))

        async with aiofiles.open(step_directory / "output_data.json", "w") as f:
            await f.write(
                json.dumps(
                    [
                        output_data.model_dump(
                            exclude_none=True,
                            exclude={"screenshot"},
                            exclude_defaults=True,
                        )
                        for output_data in memory.variables.output_data
                    ],
                    indent=4,
                )
            )

        for output_data in memory.variables.output_data:
            if output_data.screenshot:
                async with aiofiles.open(
                    step_directory
                    / f"screenshot_{output_data.screenshot.filename}.png",
                    "wb",
                ) as f:
                    await f.write(base64.b64decode(output_data.screenshot.base64))
    except Exception as e:
        logger.error(f"Failed to save latest memory state locally: {e}")


async def delete_local_data(task: Task):
    try:
        if settings.DEPLOYMENT == "dev" or task.task_directory is None:
            return

        shutil.rmtree(task.task_directory, ignore_errors=True)
    except Exception as e:
        logger.error(f"Failed to delete local data: {e}")
