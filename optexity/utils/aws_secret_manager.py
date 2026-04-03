import asyncio
import json
import logging
import os
from functools import partial

import boto3
from async_lru import alru_cache

logger = logging.getLogger(__name__)


async def _resolve_aws_credentials(workspace_id: str | None) -> tuple[str, str]:
    """Resolve AWS credentials.

    Prefers fetching the 'aws_secret_manager' integration secret from the opbackend API
    when workspace_id is provided; falls back to AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
    env vars.
    """
    if workspace_id is not None:
        try:
            from optexity.utils.integration_secrets import (
                fetch_decrypted_integration_secret,
            )

            data = await fetch_decrypted_integration_secret(
                workspace_id, "aws_secret_manager"
            )
            access_key = data.get("access_key_id")
            secret_key = data.get("secret_access_key")
            if access_key and secret_key:
                return access_key, secret_key
            logger.warning(
                f"Integration secret for workspace={workspace_id} missing credentials, "
                "falling back to env vars"
            )
        except Exception as e:
            logger.warning(
                f"Failed to fetch AWS credentials from API for workspace={workspace_id}: {e}. "
                "Falling back to env vars"
            )

    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

    if access_key and secret_key:
        return access_key, secret_key

    raise ValueError(
        "AWS credentials could not be resolved: API fetch failed or workspace_id not provided, "
        "and AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY env vars are not set"
    )


class AWSSecretsManager:
    """Wrapper around boto3 Secrets Manager client."""

    def __init__(self, region_name: str, access_key: str, secret_key: str):
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
    workspace_id: str | None = None,
) -> str:
    """
    Cached helper to fetch a value from AWS Secrets Manager.
    """
    access_key, secret_key = await _resolve_aws_credentials(workspace_id)
    manager = AWSSecretsManager(region_name, access_key, secret_key)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, partial(manager.fetch_secret, secret_name, key)
    )
