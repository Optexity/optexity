import io
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import httpx

from optexity.schema.memory import Memory
from optexity.utils.settings import settings


def create_tar_in_memory(directory: Path | str, name: str) -> io.BytesIO:
    if isinstance(directory, str):
        directory = Path(directory)
    tar_bytes = io.BytesIO()
    with tarfile.open(fileobj=tar_bytes, mode="w:gz") as tar:
        tar.add(directory, arcname=name)
    tar_bytes.seek(0)  # rewind to start
    return tar_bytes


async def create_task_in_server(
    task_id: str,
    recording_id: str,
    input_parameters: dict,
    unique_parameters: dict,
    created_at: datetime,
):
    url = urljoin(settings.SERVER_URL, settings.CREATE_TASK_ENDPOINT)
    headers = {"x-api-key": settings.API_KEY}
    body = {
        "task_id": task_id,
        "recording_id": recording_id,
        "input_parameters": input_parameters,
        "unique_parameters": unique_parameters,
        "created_at": created_at.isoformat(),
    }
    print(body)
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            url,
            headers=headers,
            json=body,
        )

        response.raise_for_status()
        try:
            return response.json()
        except httpx.HTTPStatusError as e:
            raise ValueError(
                f"Failed to create task in server: {e.response.status_code} - {e.response.text}"
            )
        except Exception as e:
            raise ValueError(f"Failed to create task in server: {e}")


async def start_task_in_server(memory: Memory):
    memory.started_at = datetime.now(timezone.utc)
    memory.status = "running"

    url = urljoin(settings.SERVER_URL, settings.START_TASK_ENDPOINT)
    headers = {"x-api-key": settings.API_KEY}
    body = {
        "task_id": memory.task_id,
        "started_at": memory.started_at.isoformat(),
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            url,
            headers=headers,
            json=body,
        )

        response.raise_for_status()
        try:
            return response.json()
        except httpx.HTTPStatusError as e:
            raise ValueError(
                f"Failed to start task in server: {e.response.status_code} - {e.response.text}"
            )
        except Exception as e:
            raise ValueError(f"Failed to start task in server: {e}")


async def complete_task_in_server(memory: Memory):
    memory.completed_at = datetime.now(timezone.utc)

    url = urljoin(settings.SERVER_URL, settings.COMPLETE_TASK_ENDPOINT)
    headers = {"x-api-key": settings.API_KEY}
    body = {
        "task_id": memory.task_id,
        "completed_at": memory.completed_at.isoformat(),
        "status": "success" if memory.status == "success" else "failed",
        "error": memory.error,
        "token_usage": memory.token_usage.model_dump(exclude_none=True),
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            url,
            headers=headers,
            json=body,
        )

        response.raise_for_status()
        try:
            return response.json()
        except httpx.HTTPStatusError as e:
            raise ValueError(
                f"Failed to complete task in server: {e.response.status_code} - {e.response.text}"
            )
        except Exception as e:
            raise ValueError(f"Failed to complete task in server: {e}")


async def save_output_data_in_server(memory: Memory):
    if len(memory.variables.output_data) == 0 and memory.final_screenshot is None:
        return

    url = urljoin(settings.SERVER_URL, settings.SAVE_OUTPUT_DATA_ENDPOINT)
    headers = {"x-api-key": settings.API_KEY}

    output_data = [
        output_data.model_dump(exclude_none=True, exclude={"screenshot"})
        for output_data in memory.variables.output_data
    ]
    body = {
        "task_id": memory.task_id,
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
        try:
            return response.json()
        except httpx.HTTPStatusError as e:
            raise ValueError(
                f"Failed to save output data in server: {e.response.status_code} - {e.response.text}"
            )
        except Exception as e:
            raise ValueError(f"Failed to save output data in server: {e}")


async def save_downloads_in_server(memory: Memory):
    if len(memory.downloads) == 0:
        return

    url = urljoin(settings.SERVER_URL, settings.SAVE_DOWNLOADS_ENDPOINT)
    headers = {"x-api-key": settings.API_KEY}

    data = {
        "task_id": memory.task_id,  # form field
    }

    tar_bytes = create_tar_in_memory(memory.downloads_directory, memory.task_id)
    files = {
        "compressed_downloads": (
            f"{memory.task_id}.tar.gz",
            tar_bytes,
            "application/gzip",
        )
    }
    async with httpx.AsyncClient() as client:

        response = await client.post(url, headers=headers, data=data, files=files)
        response.raise_for_status()
        try:
            return response.json()
        except httpx.HTTPStatusError as e:
            raise ValueError(
                f"Failed to save downloads in server: {e.response.status_code} - {e.response.text}"
            )
        except Exception as e:
            raise ValueError(f"Failed to save downloads in server: {e}")


async def save_trajectory_in_server(memory: Memory):
    url = urljoin(settings.SERVER_URL, settings.SAVE_TRAJECTORY_ENDPOINT)
    headers = {"x-api-key": settings.API_KEY}

    data = {
        "task_id": memory.task_id,  # form field
    }

    tar_bytes = create_tar_in_memory(memory.task_directory, memory.task_id)
    files = {
        "compressed_trajectory": (
            f"{memory.task_id}.tar.gz",
            tar_bytes,
            "application/gzip",
        )
    }
    async with httpx.AsyncClient() as client:

        response = await client.post(url, headers=headers, data=data, files=files)
        response.raise_for_status()
        try:
            return response.json()
        except httpx.HTTPStatusError as e:
            raise ValueError(
                f"Failed to save trajectory in server: {e.response.status_code} - {e.response.text}"
            )
        except Exception as e:
            raise ValueError(f"Failed to save trajectory in server: {e}")
