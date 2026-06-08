import asyncio
import json
import logging
from pathlib import Path

from playwright.async_api import ProxySettings

from optexity.utils.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_proxy_settings(
    use_proxy: bool, proxy_session_id: str | None
) -> ProxySettings | None:
    """Resolve the proxy server + provider-formatted credentials.

    Shared by ActualBrowser (Chrome launch) and Browser (CDP auth interception)
    so the Oxylabs/Brightdata username format stays in one place.
    """
    if not use_proxy:
        return None

    if settings.PROXY_URL is None:
        raise ValueError("PROXY_URL is not set")

    proxy: dict = {"server": settings.PROXY_URL}
    if settings.PROXY_USERNAME is not None:
        if settings.PROXY_PROVIDER == "oxylabs":
            assert settings.PROXY_USERNAME, "PROXY_USERNAME is not set"
            assert settings.PROXY_PASSWORD, "PROXY_PASSWORD is not set"
            proxy["username"] = (
                f"customer-{settings.PROXY_USERNAME}-cc-{settings.PROXY_COUNTRY}-sessid-{proxy_session_id}-sesstime-10"
            )
        elif settings.PROXY_PROVIDER == "brightdata":
            proxy["username"] = f"{settings.PROXY_USERNAME}-session-{proxy_session_id}"
        else:
            proxy["username"] = settings.PROXY_USERNAME

    if settings.PROXY_PASSWORD is not None:
        proxy["password"] = settings.PROXY_PASSWORD

    return ProxySettings(**proxy)


async def setup_proxy_auth_cdp(context, username: str, password: str, sessions: list):
    """Intercept proxy 407 challenges via CDP so Chrome never shows the auth popup.

    Must be installed on the CDP connection whose pages actually navigate.
    Appends each created CDP session to ``sessions`` for later detach.
    """
    if context is None:
        return

    async def setup_page(page):
        try:
            cdp = await context.new_cdp_session(page)
            sessions.append(cdp)
            # patterns=[{"urlPattern":"*"}] mirrors Playwright's own proxy-auth implementation.
            # An empty patterns list leaves the Fetch domain inactive in some Chrome builds,
            # causing authRequired to never fire.
            await cdp.send(
                "Fetch.enable",
                {"handleAuthRequests": True, "patterns": [{"urlPattern": "*"}]},
            )

            async def on_request_paused(event):
                # Pass all non-auth requests through immediately.
                try:
                    await cdp.send(
                        "Fetch.continueRequest", {"requestId": event["requestId"]}
                    )
                except Exception:
                    pass

            async def on_auth_required(event):
                source = event.get("authChallenge", {}).get("source", "")
                logger.debug(f"Proxy auth challenge: source={source}")
                if source == "Proxy":
                    try:
                        await cdp.send(
                            "Fetch.continueWithAuth",
                            {
                                "requestId": event["requestId"],
                                "authChallengeResponse": {
                                    "response": "ProvideCredentials",
                                    "username": username,
                                    "password": password,
                                },
                            },
                        )
                    except Exception as e:
                        logger.debug(f"Proxy auth CDP error: {e}")
                else:
                    try:
                        await cdp.send(
                            "Fetch.continueWithAuth",
                            {
                                "requestId": event["requestId"],
                                "authChallengeResponse": {"response": "Default"},
                            },
                        )
                    except Exception:
                        pass

            cdp.on(
                "Fetch.requestPaused",
                lambda e: asyncio.create_task(on_request_paused(e)),
            )
            cdp.on(
                "Fetch.authRequired",
                lambda e: asyncio.create_task(on_auth_required(e)),
            )
        except Exception as e:
            logger.warning(f"Failed to set up proxy auth CDP for page: {e}")

    for page in context.pages:
        await setup_page(page)

    context.on("page", lambda page: asyncio.create_task(setup_page(page)))


def _download_extension(url: str, output_path: Path) -> None:
    """Download extension .crx file."""
    import urllib.request

    try:
        logger.info(f"Downloading from: {url}")
        with urllib.request.urlopen(url) as response:
            content = response.read()
            logger.info(f"Downloaded {len(content)} bytes")
            with open(output_path, "wb") as f:
                f.write(content)
        logger.info(f"Saved to: {output_path}")
    except Exception as e:
        raise Exception(f"Failed to download extension: {e}")


def _extract_extension(crx_path: Path, extract_dir: Path) -> None:
    """Extract .crx file to directory."""
    import os
    import shutil
    import zipfile

    # Remove existing directory
    if extract_dir.exists():
        shutil.rmtree(extract_dir)

    extract_dir.mkdir(parents=True, exist_ok=True)

    try:
        # CRX files are ZIP files with a header, try to extract as ZIP
        with zipfile.ZipFile(crx_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        # Verify manifest exists
        if not (extract_dir / "manifest.json").exists():
            raise Exception("No manifest.json found in extension")

        logger.info("✅ Extracted as regular ZIP file")

    except zipfile.BadZipFile:
        logger.info("📦 Processing CRX header...")
        # CRX files have a header before the ZIP data
        with open(crx_path, "rb") as f:
            # Read CRX header to find ZIP start
            magic = f.read(4)
            if magic != b"Cr24":
                raise Exception(f"Invalid CRX file format. Magic: {magic}")

            version = int.from_bytes(f.read(4), "little")
            logger.info(f"CRX version: {version}")

            if version == 2:
                pubkey_len = int.from_bytes(f.read(4), "little")
                sig_len = int.from_bytes(f.read(4), "little")
                f.seek(16 + pubkey_len + sig_len)
            elif version == 3:
                header_len = int.from_bytes(f.read(4), "little")
                f.seek(12 + header_len)
            else:
                raise Exception(f"Unsupported CRX version: {version}")

            # Extract ZIP data
            zip_data = f.read()
            logger.info(f"ZIP data size: {len(zip_data)} bytes")

        # Write ZIP data to temp file and extract
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_zip:
            temp_zip.write(zip_data)
            temp_zip.flush()

            with zipfile.ZipFile(temp_zip.name, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

            os.unlink(temp_zip.name)

    # Remove 'key' from manifest if present (can cause issues)
    manifest_path = extract_dir / "manifest.json"
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text())
        logger.info(f"Manifest version: {data.get('manifest_version')}")
        logger.info(f"Extension name: {data.get('name')}")

        if "key" in data:
            logger.info("Removing 'key' field from manifest")
            del data["key"]
            manifest_path.write_text(json.dumps(data, indent=2))
    else:
        raise Exception("manifest.json not found after extraction")
