import logging
from typing import Any, Literal

import httpx

logger = logging.getLogger(__name__)


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
