import asyncio
import logging
import os
import pathlib
import platform
import shutil
import signal
import time
from typing import Literal

import aiohttp

from optexity.inference.infra.utils import _download_extension, _extract_extension
from optexity.utils.settings import settings

logger = logging.getLogger(__name__)


def find_chrome_binary(channel: Literal["chrome", "chromium"]) -> str:
    system = platform.system()

    # ---- macOS
    if system == "Darwin":
        chrome_paths = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary",
            "/Applications/Google Chrome Beta.app/Contents/MacOS/Google Chrome Beta",
            "/Applications/Google Chrome Dev.app/Contents/MacOS/Google Chrome Dev",
        ]

        chromium_paths = ["/Applications/Chromium.app/Contents/MacOS/Chromium"]

        paths = (
            chrome_paths + chromium_paths
            if channel == "chrome"
            else chromium_paths + chrome_paths
        )

        for path in paths:
            if os.path.exists(path):
                return path

        raise RuntimeError("Chrome/Chromium not found on macOS")

    # ---- Linux
    if system == "Linux":
        chrome_bins = ["google-chrome", "google-chrome-stable"]

        chromium_bins = ["chromium", "chromium-browser"]

        bins = (
            chrome_bins + chromium_bins
            if channel == "chrome"
            else chromium_bins + chrome_bins
        )

        for name in bins:
            path = shutil.which(name)
            if path:
                return path

        raise RuntimeError("Chrome/Chromium not found on Linux")

    raise RuntimeError(f"Unsupported OS: {system}")


class ActualBrowser:
    def __init__(
        self,
        channel: Literal["chrome", "chromium"],
        unique_child_arn: str,
        port: int = 9222,
        headless: bool = False,
        is_dedicated: bool = False,
        use_proxy: bool = False,
        proxy_session_id: str | None = None,
    ):
        self.chrome_path = find_chrome_binary(channel)
        self.user_data_dir = f"/tmp/userdata_{unique_child_arn}"
        self.port = port
        self.headless = headless
        self.is_dedicated = is_dedicated
        self.proc: asyncio.subprocess.Process | None = None
        self.use_proxy = use_proxy
        self.proxy_session_id = proxy_session_id

        self.extensions = [
            # {
            #     "name": "optexity recorder",
            #     "id": "pbaganbicadeoacahamnbgohafchgakp",
            #     "url": "https://clients2.google.com/service/update2/crx?response=redirect&prodversion=133&acceptformat=crx3&x=id%3Dpbaganbicadeoacahamnbgohafchgakp%26uc",
            # },
            {
                "name": "I still don't care about cookies",
                "id": "edibdbjcniadpccecjdfdjjppcpchdlm",
                "url": "https://clients2.google.com/service/update2/crx?response=redirect&prodversion=133&acceptformat=crx3&x=id%3Dedibdbjcniadpccecjdfdjjppcpchdlm%26uc",
            },
            # {
            #     "name": "popupoff",
            #     "id": "kiodaajmphnkcajieajajinghpejdjai",
            #     "url": "https://clients2.google.com/service/update2/crx?response=redirect&prodversion=133&acceptformat=crx3&x=id%3Dkiodaajmphnkcajieajajinghpejdjai%26uc",
            # },
            {
                "name": "ublock origin",
                "id": "ddkjiahejlhfcafbddmgiahcphecmpfh",
                "url": "https://clients2.google.com/service/update2/crx?response=redirect&prodversion=133&acceptformat=crx3&x=id%3Dddkjiahejlhfcafbddmgiahcphecmpfh%26uc",
            },
        ]

    async def start(self):
        try:
            logger.debug("Starting actual browser")
            if self.proc and self.proc.returncode is None:
                return

            args = [
                # ---- security / isolation (Playwright parity)
                "--disable-site-isolation-trials",
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
                "--allow-running-insecure-content",
                "--ignore-certificate-errors",
                "--ignore-ssl-errors",
                "--ignore-certificate-errors-spki-list",
                # ---- extensions
                "--enable-extensions",
                "--disable-extensions-file-access-check",
                "--disable-extensions-http-throttling",
                # ---- window / ui
                "--disable-popup-blocking",
                "--window-size=1920,1080",
                # ---- performance / stability
                "--disable-gpu",
                "--disable-background-networking",
                "--disable-sync",
                "--disable-translate",
                # ---- automation hygiene
                f"--remote-debugging-port={self.port}",
                f"--user-data-dir={self.user_data_dir}",
                "--no-first-run",
                "--no-default-browser-check",
                "--no-sandbox",
                # ---- privacy / security
                "--disable-save-password-bubble",
                "--disable-autofill-keyboard-accessory-view",
                "--disable-autofill",
                "--password-store=basic",
                "--disable-notifications",
                "--disable-credential-manager-api",
            ]

            if self.headless:
                args.append("--headless=new")

            extension_paths = self.get_extension_paths()
            if extension_paths:
                args.append(f"--disable-extensions-except={','.join(extension_paths)}")
                args.append(f"--load-extension={','.join(extension_paths)}")

            # # 👇 ADD PROXY FLAGS
            # args += self.get_proxy_args()

            if not self.is_dedicated:
                shutil.rmtree(self.user_data_dir, ignore_errors=True)

            self.proc = await asyncio.create_subprocess_exec(
                self.chrome_path,
                *args,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                preexec_fn=os.setsid,  # critical: isolate process group
            )

            await self._wait_for_cdp()
            logger.debug("CDP ready")
        except Exception as e:
            logger.error(f"Error starting actual browser: {e}")
            raise e

    async def _wait_for_cdp(self, timeout=10):
        logger.debug("Waiting for CDP")
        url = f"http://localhost:{self.port}/json/version"
        start = time.monotonic()

        async with aiohttp.ClientSession() as session:
            while time.monotonic() - start < timeout:
                try:
                    async with session.get(url, timeout=0.5) as r:
                        if r.status == 200:
                            return
                except Exception:
                    pass
                await asyncio.sleep(0.2)

        raise RuntimeError("Chrome CDP not reachable")

    async def stop(self, graceful=True):
        if not self.proc or self.proc.returncode is not None:
            return

        pgid = os.getpgid(self.proc.pid)

        if graceful:
            os.killpg(pgid, signal.SIGTERM)
            try:
                await asyncio.wait_for(self.proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                os.killpg(pgid, signal.SIGKILL)
        else:
            os.killpg(pgid, signal.SIGKILL)

        self.proc = None

        if not self.is_dedicated:
            shutil.rmtree(self.user_data_dir, ignore_errors=True)

    def get_extension_paths(self) -> list[str]:
        cache_dir = pathlib.Path("/tmp/extensions")
        cache_dir.mkdir(parents=True, exist_ok=True)
        extension_paths = []
        loaded_extension_names = []
        for ext in self.extensions:
            ext_dir = cache_dir / ext["id"]
            crx_file = cache_dir / f'{ext["id"]}.crx'

            # Check if extension is already extracted
            if ext_dir.exists() and (ext_dir / "manifest.json").exists():
                logger.info(f'✅ Using cached {ext["name"]} extension from {ext_dir}')
                extension_paths.append(str(ext_dir))
                loaded_extension_names.append(ext["name"])
                continue

            try:
                # Download extension if not cached
                if not crx_file.exists():
                    logger.info(f'📦 Downloading {ext["name"]} extension...')
                    _download_extension(ext["url"], crx_file)
                else:
                    logger.info(f'📦 Found cached {ext["name"]} .crx file')

                # Extract extension
                logger.info(f'📂 Extracting {ext["name"]} extension...')
                _extract_extension(crx_file, ext_dir)

                extension_paths.append(str(ext_dir))
                loaded_extension_names.append(ext["name"])
                logger.info(f'✅ Successfully loaded {ext["name"]}')

            except Exception as e:
                logger.error(
                    f'❌ Failed to setup {ext["name"]} extension: {e}',
                    exc_info=True,
                )
                continue

        if not extension_paths:
            logger.error("⚠️ No extensions were loaded successfully!")

        logger.info(f"Loaded extensions: {', '.join(loaded_extension_names)}")

        return extension_paths

    def _proxy_args(self) -> list[str]:
        if not self.use_proxy:
            return []

        if settings.PROXY_URL is None:
            raise ValueError("PROXY_URL is not set")

        server = self.proxy["server"]  # e.g. http://host:port
        parsed = urlparse(settings.PROXY_URL)

        if "username" in self.proxy and "password" in self.proxy:
            proxy_url = (
                f"{parsed.scheme}://"
                f"{self.proxy['username']}:{self.proxy['password']}@"
                f"{parsed.hostname}:{parsed.port}"
            )
        else:
            proxy_url = settings.PROXY_URL

        return [f"--proxy-server={proxy_url}"]

    def get_proxy(self):
        proxy = None
        if self.use_proxy:
            if settings.PROXY_URL is None:
                raise ValueError("PROXY_URL is not set")
            proxy = {"server": settings.PROXY_URL}
            if settings.PROXY_USERNAME is not None:
                if settings.PROXY_PROVIDER == "oxylabs":
                    assert settings.PROXY_COUNTRY, "PROXY_COUNTRY is not set"
                    assert settings.PROXY_USERNAME, "PROXY_USERNAME is not set"
                    assert settings.PROXY_PASSWORD, "PROXY_PASSWORD is not set"

                    proxy["username"] = (
                        f"customer-{settings.PROXY_USERNAME}-cc-{settings.PROXY_COUNTRY}-sessid-{self.proxy_session_id}-sesstime-20"
                    )
                elif settings.PROXY_PROVIDER == "brightdata":

                    proxy["username"] = (
                        f"{settings.PROXY_USERNAME}-session-{self.proxy_session_id}"
                    )

                else:
                    proxy["username"] = settings.PROXY_USERNAME

            if settings.PROXY_PASSWORD is not None:
                proxy["password"] = settings.PROXY_PASSWORD
        return proxy
