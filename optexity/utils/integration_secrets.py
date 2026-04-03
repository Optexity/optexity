import logging
from urllib.parse import urljoin

import httpx

from optexity.utils.settings import settings
from optexity.utils.utils import decrypt_fernet_payload

logger = logging.getLogger(__name__)


async def fetch_decrypted_integration_secret(
    workspace_id: str, secret_type: str
) -> dict:
    """Fetch an integration secret from the opbackend API and decrypt it.

    Makes a GET request to /integration-secrets/{type}/encrypt using the configured
    API key, then decrypts the Fernet-encrypted payload with FERNET_SECRET_KEY.
    """
    url = urljoin(
        settings.SERVER_URL,
        settings.INTEGRATION_SECRETS_ENDPOINT.format(type=secret_type),
    )
    headers = {
        "x-api-key": settings.OPTEXITY_API_KEY,
        "x-workspace-id": workspace_id,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()

        body = response.json()
        if not body.get("success"):
            raise RuntimeError(
                f"Failed to fetch integration secret for workspace={workspace_id} "
                f"type={secret_type}: {body.get('error')}"
            )

        encrypted_data: str = body.get("data", {}).get("data")
        if not encrypted_data:
            raise ValueError(
                f"No data in response for workspace={workspace_id} type={secret_type}"
            )

        decrypted: dict = decrypt_fernet_payload(encrypted_data)

        logger.info(
            f"Fetched and decrypted integration secret workspace={workspace_id} type={secret_type}"
        )
        return decrypted
    except Exception as e:
        logger.error(
            f"Error fetching integration secret workspace={workspace_id} type={secret_type}: {e}"
        )
        raise
