import base64
import logging
import os
from pathlib import Path
from typing import List, Optional
import asyncio
import json
import boto3
from botocore.exceptions import ClientError
import aiofiles
import pyotp
from async_lru import alru_cache
from onepassword import Client as OnePasswordClient
from pydantic import create_model

logger = logging.getLogger(__name__)

_onepassword_client = None


async def get_onepassword_client():
    global _onepassword_client
    if _onepassword_client is None:
        _onepassword_client = await OnePasswordClient.authenticate(
            auth=os.getenv("OP_SERVICE_ACCOUNT_TOKEN"),
            integration_name="Optexity 1Password Integration",
            integration_version="v1.0.0",
        )
    return _onepassword_client


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


def save_screenshot(screenshot: str, path: str):
    with open(path, "wb") as f:
        f.write(base64.b64decode(screenshot))


async def save_and_clear_downloaded_files(content: bytes | str, filename: Path):
    if isinstance(content, bytes):
        async with aiofiles.open(filename, "wb") as f:
            await f.write(content)
    elif isinstance(content, str):
        async with aiofiles.open(filename, "w") as f:
            await f.write(content)
    else:
        logger.error(f"Unsupported content type: {type(content)}")


def get_totp_code(totp_secret: str, digits: int = 6):
    totp = pyotp.TOTP(totp_secret, digits=digits)
    return totp.now()


@alru_cache(maxsize=1000)
async def get_onepassword_value(vault_name: str, item_name: str, field_name: str):
    client = await get_onepassword_client()
    str_value = await client.secrets.resolve(
        f"op://{vault_name}/{item_name}/{field_name}"
    )

    return str_value

# adding a get aws secrets function
@alru_cache(maxsize=1000)
async def get_aws_secrets_manager_value(secret_id: str, secret_key: str | None = None, region: str | None = None):

    def _get_secret():
        region_name = region or os.getenv("AWS_REGION", "us-east-1")
        client = boto3.client("secretsmanager", region_name=region_name)
        
        try:
            response = client.get_secret_value(SecretId=secret_id)
        except ClientError as e:
            logger.error(f"Failed to retrieve secret {secret_id}: {e}")
            raise
        
        if "SecretString" in response:
            secret_value = response["SecretString"]
        else:
            secret_value = base64.b64decode(response["SecretBinary"]).decode("utf-8")
        
        # parsing json to get the key
        if secret_key:
            try:
                secret_dict = json.loads(secret_value)
                return secret_dict.get(secret_key)
            except json.JSONDecodeError:
                logger.error(f"Secret {secret_id} is not valid JSON but secret_key was provided")
                raise ValueError(f"Secret {secret_id} is not valid JSON")
        
        return secret_value

    # Run boto3 sync call in thread pool to avoid blocking
    return await asyncio.to_thread(_get_secret)
