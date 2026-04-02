import asyncio
import json
import logging
import os
from functools import partial

import boto3
from async_lru import alru_cache

logger = logging.getLogger(__name__)


def _get_aws_credentials() -> tuple[str, str]:
    """
    Required environment variables:
        AWS_ACCESS_KEY_ID     — your AWS access key
        AWS_SECRET_ACCESS_KEY — your AWS secret access key
    """
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

    if not access_key:
        raise ValueError("AWS_ACCESS_KEY_ID environment variable is not set")
    if not secret_key:
        raise ValueError("AWS_SECRET_ACCESS_KEY environment variable is not set")

    return access_key, secret_key


class AWSSecretsManager:
    """
    Wrapper around boto3 Secrets Manager client.

    Credentials are read from environment variables:
        AWS_ACCESS_KEY_ID      — required
        AWS_SECRET_ACCESS_KEY  — required
    """

    def __init__(self, region_name: str):
        access_key, secret_key = _get_aws_credentials()
        self.client = boto3.client(
            "secretsmanager",
            region_name=region_name,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

    def fetch_secret(self, secret_name: str, key: str | None = None) -> str:
        """Fetch a secret value. Runs inside a thread-pool executor."""
        try:
            response = self.client.get_secret_value(SecretId=secret_name)
        except Exception:
            raise

        raw = (
            response["SecretString"]
            if "SecretString" in response
            else response["SecretBinary"].decode("utf-8")
        )

        if key is None:
            return raw

        try:
            data = json.loads(raw)
        except Exception:
            raise

        if key not in data:
            raise KeyError(
                f"Key '{key}' not found in secret '{secret_name}'. "
                f"Available keys: {list(data.keys())}"
            )

        return str(data[key])


@alru_cache(maxsize=1000)
async def get_aws_secret_value(
    secret_name: str,
    region_name: str,
    key: str | None = None,
) -> str:
    """
    Cached module-level helper. Sync call running inside the event loop for non-blocking operation.
    """
    manager = AWSSecretsManager(region_name)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, partial(manager.fetch_secret, secret_name, key)
    )
