import asyncio
import base64
import json
import logging
import pathlib
import re
import shutil
from typing import Literal
from uuid import uuid4

import patchright.async_api
import playwright.async_api
from browser_use import Agent, BrowserSession, ChatGoogle
from browser_use.browser.views import BrowserStateSummary
from patchright._impl._errors import TimeoutError as PatchrightTimeoutError
from playwright._impl._errors import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import Download, Locator, Page, Request, Response

from optexity.inference.infra.utils import _download_extension, _extract_extension
from optexity.schema.memory import Memory, NetworkRequest, NetworkResponse
from optexity.utils.settings import settings

logger = logging.getLogger(__name__)

_global_playwright: (
    playwright.async_api.Playwright | patchright.async_api.Playwright | None
) = None
_global_context: (
    playwright.async_api.BrowserContext | patchright.async_api.BrowserContext | None
) = None


class Browser:
    def __init__(
        self,
        memory: Memory,
        user_data_dir: str,
        headless: bool = False,
        proxy: str | None = None,
        stealth: bool = True,
        backend: Literal["browser-use", "browserbase"] = "browser-use",
        debug_port: int = 9222,
        channel: Literal["chromium", "chrome"] = "chromium",
        use_proxy: bool = False,
        proxy_session_id: str | None = None,
        is_dedicated: bool = False,
    ):

        if proxy:
            proxy = proxy.removeprefix("http://").removeprefix("https://")
            self.proxy = "http://" + proxy

        self.headless = headless
        self.stealth = stealth
        self.user_data_dir = user_data_dir
        self.backend = backend
        self.debug_port = debug_port
        self.use_proxy = use_proxy
        self.proxy_session_id = proxy_session_id
        self.playwright: (
            playwright.async_api.Playwright | patchright.async_api.Playwright | None
        ) = None
        self.browser = None
        self.context: (
            playwright.async_api.BrowserContext
            | patchright.async_api.BrowserContext
            | None
        ) = None
        self.page = None
        self.cdp_url = f"http://localhost:{self.debug_port}"
        self.backend_agent = None
        self.channel = channel
        self.memory = memory
        self.page_to_target_id = []
        self.previous_total_pages = 0
        self.is_dedicated = is_dedicated
        self.active_downloads = 0
        self.all_active_downloads_done = asyncio.Event()
        self.all_active_downloads_done.set()

        self.network_calls: list[NetworkResponse | NetworkRequest] = []

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

        if not self.is_dedicated:
            global _global_playwright, _global_context
            _global_playwright = None
            _global_context = None

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

    async def start(self):
        global _global_playwright, _global_context
        logger.debug("Starting browser")
        try:
            if (
                _global_playwright is None
                or _global_context is None
                or not self.is_dedicated
            ):
                if _global_context is not None:
                    await _global_context.close()
                    _global_context = None
                    self.context = None
                if _global_playwright is not None:
                    await _global_playwright.stop()
                    _global_playwright = None
                    self.playwright = None

                if self.stealth:
                    from patchright.async_api import async_playwright
                else:
                    from playwright.async_api import async_playwright

                extension_paths = self.get_extension_paths()
                proxy = self.get_proxy()

                args = [
                    "--disable-site-isolation-trials",
                    "--disable-web-security",
                    "--disable-features=IsolateOrigins,site-per-process",
                    "--allow-running-insecure-content",
                    "--ignore-certificate-errors",
                    "--ignore-ssl-errors",
                    "--ignore-certificate-errors-spki-list",
                    "--enable-extensions",
                    "--disable-extensions-file-access-check",
                    "--disable-extensions-http-throttling",
                ]

                if extension_paths:
                    disable_except = (
                        f'--disable-extensions-except={",".join(extension_paths)}'
                    )
                    load_extension = f'--load-extension={",".join(extension_paths)}'
                    args.append(disable_except)
                    args.append(load_extension)
                    logger.info(f"Extension args: {disable_except}")
                    logger.info(f"Extension args: {load_extension}")

                self.playwright = await async_playwright().start()
                self.context = await self.playwright.chromium.launch_persistent_context(
                    channel=self.channel,
                    user_data_dir=self.user_data_dir,
                    headless=self.headless,
                    proxy=proxy,
                    args=[
                        # "--start-fullscreen",
                        "--disable-popup-blocking",
                        "--window-size=1920,1080",
                        f"--remote-debugging-port={self.debug_port}",
                        "--disable-gpu",
                        "--disable-background-networking",
                        "--disable-sync",
                        "--disable-translate",
                        "--disable-features=site-per-process",
                    ]
                    + args,
                    chromium_sandbox=False,
                    no_viewport=True,
                )

                _global_playwright = self.playwright
                _global_context = self.context

                self.context.on("request", lambda req: self.log_request(req))
                self.context.on("response", lambda resp: self.log_response(resp))
                self.context.on(
                    "response", lambda resp: self.handle_random_url_downloads(resp)
                )
                self.context.on(
                    "page",
                    lambda p: (
                        p.on(
                            "download",
                            lambda download: self.handle_random_download(download),
                        )
                    ),
                )

            elif self.is_dedicated:
                self.context = _global_context
                self.playwright = _global_playwright
                for i in range(len(self.context.pages) - 1, 0, -1):
                    await self.context.pages[i].close()
            else:
                raise ValueError(
                    "Browser is not dedicated and global playwright and context are not set"
                )

            # self.context = await self.browser.new_context(
            #     no_viewport=True, ignore_https_errors=True
            # )

            # self.page = await self.context.new_page()
            if len(self.context.pages) == 0:
                await self.context.new_page()

            self.page = self.context.pages[0]

            browser_session = BrowserSession(cdp_url=self.cdp_url, keep_alive=True)

            self.backend_agent = Agent(
                task="",
                llm=ChatGoogle(model="gemini-flash-latest"),
                browser_session=browser_session,
                use_vision=False,
            )

            await self.backend_agent.browser_session.start()

            tabs = await self.backend_agent.browser_session.get_tabs()

            for tab in tabs[::-1]:
                if tab.target_id not in self.page_to_target_id:
                    self.page_to_target_id.append(tab.target_id)
            self.previous_total_pages = len(self.context.pages)

            logger.debug("Browser started successfully")

        except Exception as e:
            logger.error(f"Error starting playwright: {e}")
            raise e

    async def stop(self, force: bool = False):
        logger.debug("Stopping backend agent")
        if self.backend_agent is not None:
            logger.debug("Stopping backend agent")
            self.backend_agent.stop()
            if self.backend_agent.browser_session:
                logger.debug("Resetting browser session")
                await self.backend_agent.browser_session.stop()
                await self.backend_agent.close()
                # await self.backend_agent.browser_session._storage_state_watchdog._stop_monitoring()
                # await self.backend_agent.browser_session.reset()
                logger.debug("Browser session reset")
            self.backend_agent = None

        if not self.is_dedicated or force:
            logger.debug("Stopping context and playwright and browser as not dedicated")
            if self.context is not None:
                logger.debug("Stopping context")
                await self.context.close()
                self.context = None

            if self.browser is not None:
                logger.debug("Stopping browser")
                await self.browser.close()
                self.browser = None

            if self.playwright is not None:
                logger.debug("Stopping playwright")
                await self.playwright.stop()
                self.playwright = None
            shutil.rmtree(self.user_data_dir, ignore_errors=True)

            global _global_playwright, _global_context
            _global_playwright = None
            _global_context = None
            self.context = None
            self.playwright = None
            logger.debug("Global playwright and context reset")

        else:
            logger.debug("browser not stopped as dedicated")

    async def get_current_page(
        self,
    ) -> playwright.async_api.Page | patchright.async_api.Page:
        if self.context is None:
            raise ValueError("Context is not set")
        pages = self.context.pages
        if len(pages) == 0:
            self.page = await self.context.new_page()
        else:
            self.page = pages[-1]

        return self.page

    async def handle_new_tabs(self, max_wait_time: float) -> bool:

        total_time = 0
        while total_time < max_wait_time:
            pages = self.context.pages
            if len(pages) > self.previous_total_pages:
                break
            await asyncio.sleep(1)
            total_time += 1

        pages = self.context.pages
        if len(pages) == self.previous_total_pages:
            return False, total_time

        tabs = await self.backend_agent.browser_session.get_tabs()

        for tab in tabs[::-1]:
            if tab.target_id not in self.page_to_target_id:
                self.page_to_target_id.append(tab.target_id)
        self.previous_total_pages = len(pages)

        tab_id = self.page_to_target_id[-1][-4:]
        action_model = self.backend_agent.ActionModel(**{"switch": {"tab_id": tab_id}})
        await self.backend_agent.multi_act([action_model])
        return True, total_time

    async def close_current_tab(self):
        if self.context is None:
            return None

        pages = self.context.pages

        if len(pages) == 1:
            logger.warning("Atleast one tab should be open, skipping close current tab")
            return False

        if len(self.page_to_target_id) > 1:
            tab_id_after_close = self.page_to_target_id[-2][-4:]
            action_model = self.backend_agent.ActionModel(
                **{"switch": {"tab_id": tab_id_after_close}}
            )
            await self.backend_agent.multi_act([action_model])
            self.page_to_target_id.pop()

        last_page = pages[-1]
        await last_page.close()

    async def switch_tab(self, tab_index: int):
        if self.context is None:
            return None

        pages = self.context.pages

        if len(pages) == 1:
            logger.warning("Atleast one tab should be open, skipping close current tab")
            return False

        tab_id = self.page_to_target_id[tab_index][-4:]
        page = pages[tab_index]

        await page.bring_to_front()

        action_model = self.backend_agent.ActionModel(**{"switch": {"tab_id": tab_id}})
        await self.backend_agent.multi_act([action_model])

    async def get_locator_from_command(self, command: str) -> Locator:
        page = await self.get_current_page()
        if page is None:
            return None
        locator: Locator = eval(f"page.{command}")
        return locator

    def get_xpath_from_index(self, index: int) -> str:
        raise NotImplementedError("Not implemented")

    async def go_to_url(self, url: str):
        try:
            page = await self.get_current_page()
            if page is None:
                logger.error(f"Cannot navigate to {url}: No page available")
                return None
            await page.goto(url, timeout=10000)
        except (TimeoutError, PatchrightTimeoutError, PlaywrightTimeoutError) as e:
            logger.warning(
                f"Navigation timeout for {url}: {type(e).__name__}",
                extra={"url": url, "timeout_ms": 10000, "error_type": type(e).__name__},
            )

            # For non-critical navigation, continue with warning
            return None
        except Exception as e:
            logger.error(
                f"Unexpected error navigating to {url}: {e}",
                exc_info=True,
                extra={"url": url},
            )
            return None

    async def get_browser_state_summary(self) -> BrowserStateSummary:
        browser_state_summary = await self.backend_agent.browser_session.get_browser_state_summary(
            include_screenshot=True,  # always capture even if use_vision=False so that cloud sync is useful (it's fast now anyway)
            include_recent_events=False,
            cached=False,
        )

        return browser_state_summary

    async def get_current_page_url(self) -> str:
        try:
            page = await self.get_current_page()
            return page.url
        except Exception as e:
            logger.error(f"Error getting current page URL: {e}")
            return ""

    async def get_current_page_title(self) -> str:
        try:
            page = await self.get_current_page()
            return await page.title()
        except Exception as e:
            logger.error(f"Error getting current page title: {e}")
            return ""

    async def handle_random_download(self, download: Download):
        self.active_downloads += 1
        self.all_active_downloads_done.clear()

        temp_path = await download.path()
        async with self.memory.download_lock:
            if temp_path not in self.memory.raw_downloads:
                self.memory.raw_downloads[temp_path] = (False, download)
        self.active_downloads -= 1

        if self.active_downloads == 0:
            self.all_active_downloads_done.set()

    async def handle_random_url_downloads(self, resp: Response):
        try:

            if "application/pdf" in resp.headers.get("content-type", ""):
                self.active_downloads += 1
                self.all_active_downloads_done.clear()

                # Default filename fallback
                filename = f"{uuid4()}.pdf"

                # Try to get suggested filename from headers
                content_disposition = resp.headers.get("content-disposition")
                if content_disposition:
                    match = re.search(
                        r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?',
                        content_disposition,
                    )
                    if match:
                        filename = match.group(1)

                self.memory.urls_to_downloads.append((resp.url, filename))
                logger.info(f"Added URL to downloads: {resp.url}, {filename}")
                self.active_downloads -= 1
        except Exception as e:
            logger.error(f"Error handling random responses: {e}")

        if self.active_downloads == 0:
            self.all_active_downloads_done.set()

    async def log_request(self, req: Request):
        try:
            body = req.post_data  # this is None for GET/HEAD
            # Rebuild cookies exactly like curl -b
            cookies = await req.frame.page.context.cookies()
            cookie_header = "; ".join(f"{c['name']}={c['value']}" for c in cookies)

            # Rebuild headers
            headers = dict(req.headers)
            headers["cookie"] = cookie_header

            # Body as raw bytes
            body = req.post_data

            self.network_calls.append(
                NetworkRequest(
                    url=req.url, method=req.method, headers=headers, body=body
                )
            )

        except Exception as e:
            # logger.error(f"Could not get body: {e}")
            pass

    async def log_response(self, response: Response):
        try:
            body = await response.json()
        except Exception:
            try:
                body = await response.text()
            except Exception:
                body = None

        # Try to enrich response with request method and content length
        method = None
        try:
            # Playwright provides request object for a response
            method = response.request.method
        except Exception:
            pass

        content_length = 0
        try:
            if body is not None:
                if isinstance(body, (str, bytes)):
                    content_length = len(body)
                elif isinstance(body, dict):
                    content_length = len(json.dumps(body))
        except Exception:
            pass

        self.network_calls.append(
            NetworkResponse(
                url=response.url,
                method=method,
                status=response.status,
                headers=response.headers,
                body=body,
                content_length=content_length,
            )
        )

    async def clear_network_calls(self):
        self.network_calls.clear()

    async def get_screenshot(self, full_page: bool = False) -> str | None:
        page = await self.get_current_page()
        if page is None:
            return None
        screenshot_bytes = await page.screenshot(full_page=full_page)
        screenshot_base64 = base64.b64encode(screenshot_bytes).decode("utf-8")

        return screenshot_base64
