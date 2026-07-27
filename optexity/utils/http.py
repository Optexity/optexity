import asyncio
import logging
from typing import Any, Literal

import httpx

logger = logging.getLogger(__name__)


async def request_with_backoff(
    url: str,
    *,
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"] = "GET",
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
    max_backoff_seconds: float = 240.0,
    initial_backoff_seconds: float = 15.0,
    max_single_backoff_seconds: float = 120.0,
    client_error_retries: int = 3,
    client_error_wait_seconds: float = 3.0,
    log_label: str = "request",
) -> tuple[httpx.Response | None, int]:
    """HTTP request with exponential backoff when the server is down.

    Retries on 5xx and transport errors (connection/timeout) until
    ``max_backoff_seconds`` of wait time is exhausted. Client errors (status
    < 500) are retried up to ``client_error_retries`` times with a fixed
    ``client_error_wait_seconds`` wait between attempts.

    Returns ``(response, attempts)``. ``response`` is set only on 2xx success.
    """
    backoff_seconds = initial_backoff_seconds
    total_backoff = 0.0
    attempt = 0
    client_error_attempts = 0

    while True:
        attempt += 1
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.request(method, url, headers=headers or {})
                if response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"Server unavailable (HTTP {response.status_code})",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                return response, attempt
        except httpx.HTTPStatusError as err:
            status = err.response.status_code if err.response is not None else None
            logger.warning(f"{log_label} attempt {attempt} failed: {err}")
            if status is None or status < 500:
                client_error_attempts += 1
                if client_error_attempts >= client_error_retries:
                    return None, attempt
                logger.info(
                    f"Waiting {client_error_wait_seconds}s before retry "
                    f"({client_error_attempts}/{client_error_retries}) for {log_label}"
                )
                await asyncio.sleep(client_error_wait_seconds)
                continue
        except httpx.TransportError as err:
            logger.warning(f"{log_label} attempt {attempt} failed: {err}")
        except Exception as err:
            logger.warning(f"{log_label} attempt {attempt} failed: {err}")
            return None, attempt

        if total_backoff >= max_backoff_seconds:
            logger.warning(f"Exhausted {max_backoff_seconds}s backoff for {log_label}")
            return None, attempt

        sleep_time = min(backoff_seconds, max_backoff_seconds - total_backoff)
        logger.info(
            f"Server appears down; waiting {sleep_time}s before retry "
            f"({total_backoff + sleep_time}/{max_backoff_seconds}s backoff) "
            f"for {log_label}"
        )
        await asyncio.sleep(sleep_time)
        total_backoff += sleep_time
        backoff_seconds = min(backoff_seconds * 2, max_single_backoff_seconds)


async def make_api_request(
    url: str,
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"] = "GET",
    headers: dict[str, str] | None = None,
    body: dict | str | None = None,
    query_params: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Make an HTTP request and return a result dict with status_code, headers, and body."""
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            kwargs: dict[str, Any] = {
                "method": method,
                "url": url,
                "headers": headers or {},
                "params": query_params or {},
                "timeout": timeout,
            }

            if body is not None:
                if isinstance(body, dict):
                    kwargs["json"] = body
                else:
                    kwargs["content"] = body

            response = await client.request(**kwargs)

        try:
            response_body = response.json()
        except Exception:
            response_body = response.text

        return {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "body": response_body,
        }

    except httpx.TimeoutException as e:
        logger.error(f"API call timed out: {url} - {e}")
        return {
            "error": "timeout",
            "message": str(e),
            "status_code": None,
            "body": None,
            "headers": {},
        }

    except httpx.HTTPError as e:
        logger.error(f"API call HTTP error: {url} - {e}")
        return {
            "error": "http_error",
            "message": str(e),
            "status_code": None,
            "body": None,
            "headers": {},
        }
