import json
import logging

from urllib.parse import urljoin

import aiofiles
import httpx

from optexity.schema.memory import Memory
from optexity.schema.task import Task

from optexity.utils.settings import settings
import os

logger = logging.getLogger(__name__)


def deep_sort(value):
    if isinstance(value, dict):
        return {key: deep_sort(val) for key, val in sorted(value.items())}

    if isinstance(value, list):
        items = [deep_sort(item) for item in value]
        return sorted(
            items,
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ),
        )

    return value


def canonical_json(data):
    return json.dumps(
        deep_sort(data),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def get_portal_endpoint(task: Task) -> str:
    return "/".join(task.endpoint_name.split("/")[:-1])


def get_cookie_credential_indexing_keys_endpoint(portal_endpoint: str) -> str:

    url = os.path.join(settings.MARKETPLACE_SERVER_URL, portal_endpoint)
    url = os.path.join(url, "get_cookies_credential_indexing_keys")
    return url


def get_processed_cookie_data_endpoint(portal_endpoint: str) -> str:
    url = os.path.join(settings.MARKETPLACE_SERVER_URL, portal_endpoint)
    url = os.path.join(url, "get_processed_cookies_data")
    return url


async def get_cookie_credential_indexing_data(task: Task, portal_endpoint: str) -> str:
    cookie_indexing_keys_endpoint = get_cookie_credential_indexing_keys_endpoint(
        portal_endpoint
    )

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(cookie_indexing_keys_endpoint)
        if response.status_code != 200:
            raise Exception(f"Failed to run API automation: {response.text}")
        cookie_keys = response.json()["data"]

    indexing_data = {}
    for key in cookie_keys:
        indexing_data[key] = task.input_parameters[key]

    indexing_data_json = canonical_json(indexing_data)
    return indexing_data_json


async def get_processed_cookie_data(
    cookies: list[dict], output_data: list[dict], portal_endpoint: str
) -> str:
    processed_cookie_data_endpoint = get_processed_cookie_data_endpoint(portal_endpoint)
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            processed_cookie_data_endpoint,
            json={"cookies": cookies, "output_data": output_data},
        )
        if response.status_code != 200:
            raise Exception(f"Failed to run API automation: {response.text}")
        processed_cookie_data = response.json()["data"]

    return processed_cookie_data


async def get_cookies(task: Task) -> list[dict]:
    try:
        download_path = task.downloads_directory / "cookies.json"
        async with aiofiles.open(download_path, "r") as f:
            cookies = await f.read()
            cookies = json.loads(cookies) if cookies else []
        return cookies
    except Exception as e:
        return []


async def save_processed_cookie_data_in_server(task: Task, memory: Memory):
    if not task.is_marketplace or not task.is_browser:
        return

    try:
        portal_endpoint = get_portal_endpoint(task)
        credential_index = await get_cookie_credential_indexing_data(
            task, portal_endpoint
        )
        cookies = await get_cookies(task)
        output_data = [
            output_data.model_dump(exclude_none=True, exclude={"screenshot"})
            for output_data in memory.variables.output_data
        ]
        processed_cookie_data = await get_processed_cookie_data(
            cookies, output_data, portal_endpoint
        )

        url = urljoin(settings.SERVER_URL, settings.SAVE_PROCESSED_COOKIE_DATA_ENDPOINT)
        headers = {"x-api-key": task.api_key}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url,
                headers=headers,
                json={
                    "portal_endpoint": portal_endpoint,
                    "credential_index": credential_index,
                    "processed_cookie_data": processed_cookie_data,
                },
            )

            response.raise_for_status()
            return response.json()

    except httpx.HTTPStatusError as e:
        logger.error(
            f"Failed to save cookies processed cookie data in server: {e.response.status_code} - {e.response.text}"
        )
    except Exception as e:
        logger.error(f"Failed to save cookies processed cookie data in server: {e}")


async def get_processed_cookie_data_in_server(task: Task):
    if not task.is_marketplace:
        return

    try:
        portal_endpoint = get_portal_endpoint(task)
        credential_index = await get_cookie_credential_indexing_data(
            task, portal_endpoint
        )

        url = urljoin(settings.SERVER_URL, settings.GET_PROCESSED_COOKIE_DATA_ENDPOINT)
        headers = {"x-api-key": task.api_key}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url,
                headers=headers,
                json={
                    "portal_endpoint": portal_endpoint,
                    "credential_index": credential_index,
                },
            )

            response.raise_for_status()
            return response.json()["data"]

    except httpx.HTTPStatusError as e:
        logger.error(
            f"Failed to save cookies processed cookie data in server: {e.response.status_code} - {e.response.text}"
        )
    except Exception as e:
        logger.error(f"Failed to save cookies processed cookie data in server: {e}")
