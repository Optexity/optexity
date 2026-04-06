import base64
import json
import logging
import os
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import aiofiles
import pyotp
from async_lru import alru_cache
from cryptography.fernet import Fernet
from onepassword import Client as OnePasswordClient
from pydantic import create_model

from optexity.utils.settings import Settings

settings = Settings()
logger = logging.getLogger(__name__)


def decrypt_fernet_payload(encrypted_data: str) -> dict:
    if not settings.FERNET_SECRET_KEY:
        raise ValueError("FERNET_SECRET_KEY must be set in settings to decrypt secrets")
    fernet = Fernet(settings.FERNET_SECRET_KEY.encode())
    return json.loads(fernet.decrypt(encrypted_data.encode()).decode())


# Cached clients keyed by the service-account token so a single process can
# serve multiple workspaces without re-authenticating unnecessarily.
_onepassword_clients: dict[str, OnePasswordClient] = {}


async def _get_onepassword_token(workspace_id: str | None) -> str:
    """Resolve the 1Password service-account token.

    Prefers fetching the 'one_password' integration secret from the opbackend API
    when workspace_id is provided; falls back to the OP_SERVICE_ACCOUNT_TOKEN env var.
    """
    if workspace_id is not None:
        try:
            from optexity.utils.integration_secrets import (
                fetch_decrypted_integration_secret,
            )

            data = await fetch_decrypted_integration_secret(
                workspace_id, "one_password"
            )
            token = data.get("service_account_token")
            if token:
                return token
            logger.warning(
                f"Integration secret for workspace={workspace_id} missing token, "
                "falling back to env var"
            )
        except Exception as e:
            logger.warning(
                f"Failed to fetch 1Password token from API for workspace={workspace_id}: {e}. "
                "Falling back to env var"
            )

    token = os.getenv("OP_SERVICE_ACCOUNT_TOKEN")
    if token:
        return token

    raise ValueError(
        "1Password token could not be resolved: API fetch failed or workspace_id not provided, "
        "and OP_SERVICE_ACCOUNT_TOKEN env var is not set"
    )


async def get_onepassword_client(workspace_id: str | None = None) -> OnePasswordClient:
    token = await _get_onepassword_token(workspace_id)
    if token not in _onepassword_clients:
        _onepassword_clients[token] = await OnePasswordClient.authenticate(
            auth=token,
            integration_name="Optexity 1Password Integration",
            integration_version="v1.0.0",
        )
    return _onepassword_clients[token]


def build_model(schema: dict, model_name="AutoModel"):
    fields = {}
    for key, value in schema.items():
        if isinstance(value, str):  # primitive type
            py_type = eval(value)  # e.g., "str" -> str
            fields[key] = (Optional[py_type], None)
        elif isinstance(value, dict):  # nested object
            sub_model = build_model(value, model_name=f"{model_name}_{key}")
            fields[key] = (Optional[sub_model], None)
        elif isinstance(value, list):  # list of objects or primitives
            if len(value) > 0 and isinstance(value[0], dict):
                sub_model = build_model(value[0], model_name=f"{model_name}_{key}")
                fields[key] = (Optional[List[sub_model]], None)
            else:  # list of primitives
                py_type = eval(value[0])
                fields[key] = (Optional[List[py_type]], None)
    return create_model(model_name, **fields)


async def save_screenshot(screenshot: str, path: Path | str):
    """Asynchronously save a base64-encoded screenshot to disk."""
    # Ensure we write bytes and use aiofiles for non-blocking I/O
    async with aiofiles.open(path, "wb") as f:
        await f.write(base64.b64decode(screenshot))


async def save_and_clear_downloaded_files(content: bytes | str, filename: Path):
    if isinstance(content, bytes):
        async with aiofiles.open(filename, "wb") as f:
            await f.write(content)
    elif isinstance(content, str):
        async with aiofiles.open(filename, "w") as f:
            await f.write(content)
    else:
        logger.error(f"Unsupported content type: {type(content)}")


def get_totp_code(totp_secret: str, digits: int | None = None):
    if digits is None:
        digits = 6
    totp = pyotp.TOTP(totp_secret, digits=digits)
    return totp.now()


@alru_cache(maxsize=1000)
async def get_onepassword_value(
    vault_name: str,
    item_name: str,
    field_name: str,
    workspace_id: str | None = None,
) -> str:
    client = await get_onepassword_client(workspace_id)
    return await client.secrets.resolve(f"op://{vault_name}/{item_name}/{field_name}")


def clean_url(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        url = "http://" + url  # needed for urlparse

    parsed = urlparse(url)
    domain = parsed.netloc.lower()

    if domain.startswith("www."):
        domain = domain[4:]

    return domain


def is_url(value: str | Path) -> bool:
    try:
        result = urlparse(str(value))
        return result.scheme in {"http", "https"} and bool(result.netloc)
    except Exception:
        return False


def is_local_path(value: str | Path) -> bool:
    try:
        return Path(str(value)).expanduser().exists()
    except Exception:
        return False
