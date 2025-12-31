import asyncio
import base64
import json
import logging
import re
from typing import Literal
from uuid import uuid4

from browser_use import Agent, BrowserSession, ChatGoogle
from browser_use.browser.views import BrowserStateSummary
from patchright._impl._errors import TimeoutError as PatchrightTimeoutError
from playwright._impl._errors import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import Download, Locator, Page, Request, Response

from optexity.schema.memory import Memory, NetworkRequest, NetworkResponse
from optexity.utils.settings import settings

logger = logging.getLogger(__name__)


class Browser:
    def __init__(
        self,
        memory: Memory,
        user_data_dir: str = None,
        headless: bool = False,
        proxy: str = None,
        stealth: bool = True,
        backend: Literal["browser-use", "browserbase"] = "browser-use",
        debug_port: int = 9222,
        channel: Literal["chromium", "chrome"] = "chromium",
        use_proxy: bool = False,
        proxy_session_id: str | None = None,
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
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.cdp_url = f"http://localhost:{self.debug_port}"
        self.backend_agent = None
        self.channel = channel
        self.memory = memory
        self.page_to_target_id = []
        self.previous_total_pages = 0

        self.active_downloads = 0
        self.all_active_downloads_done = asyncio.Event()
        self.all_active_downloads_done.set()

        self.network_calls: list[NetworkResponse | NetworkRequest] = []

    async def start(self):
        logger.debug("Starting browser")
        try:
            if self.playwright is not None:
                await self.playwright.stop()

            if self.stealth:
                from patchright.async_api import async_playwright
            else:
                from playwright.async_api import async_playwright

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

            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                channel=self.channel,
                headless=self.headless,
                proxy=proxy,
                args=[
                    "--start-fullscreen",
                    "--disable-popup-blocking",
                    "--window-size=1920,1080",
                    f"--remote-debugging-port={self.debug_port}",
                    "--disable-gpu",
                    "--disable-extensions",
                    "--disable-background-networking",
                ],
                chromium_sandbox=False,
            )

            self.context = await self.browser.new_context(
                no_viewport=True, ignore_https_errors=True
            )

            async def log_request(req: Request):
                await self.log_request(req)

            async def handle_random_download(download: Download):
                await self.handle_random_download(download)

            async def handle_random_url_downloads(resp: Response):
                await self.handle_random_url_downloads(resp)

            self.context.on("request", log_request)
            self.context.on("response", handle_random_url_downloads)

            self.context.on(
                "page", lambda p: (p.on("download", handle_random_download))
            )

            self.page = await self.context.new_page()

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

    async def stop(self):
        logger.debug("Stopping full system")
        if self.backend_agent is not None:
            logger.debug("Stopping backend agent")
            self.backend_agent.stop()
            if self.backend_agent.browser_session:
                logger.debug("Resetting browser session")
                await self.backend_agent.browser_session.stop()
                # await self.backend_agent.browser_session._storage_state_watchdog._stop_monitoring()
                # await self.backend_agent.browser_session.reset()
                logger.debug("Browser session reset")
            self.backend_agent = None

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
        logger.debug("Full system stopped")

    async def get_current_page(self) -> Page | None:
        if self.context is None:
            return None
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
            if url == "about:blank":
                return
            page = await self.get_current_page()
            if page is None:
                return None
            await page.goto(url, timeout=10000)
        except TimeoutError as e:
            pass
        except PatchrightTimeoutError as e:
            pass
        except PlaywrightTimeoutError as e:
            pass

    async def get_browser_state_summary(self) -> BrowserStateSummary:
        browser_state_summary = await self.backend_agent.browser_session.get_browser_state_summary(
            include_screenshot=True,  # always capture even if use_vision=False so that cloud sync is useful (it's fast now anyway)
            include_recent_events=self.backend_agent.include_recent_events,
            cached=False,
        )

        return browser_state_summary

    async def get_current_page_url(self) -> str:
        try:
            page = await self.get_current_page()
            if page is None:
                return None
            return page.url
        except Exception as e:
            logger.error(f"Error getting current page URL: {e}")
            return None

    async def get_current_page_title(self) -> str:
        try:
            page = await self.get_current_page()
            if page is None:
                return None
            return await page.title()
        except Exception as e:
            logger.error(f"Error getting current page title: {e}")
            return None

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

    async def attach_network_listeners(self):
        page = await self.get_current_page()

        # remove old listeners first
        try:
            page.remove_listener("response", self._on_response)
        except Exception:
            pass

        page.on("response", self._on_response)

    async def detach_network_listeners(self):
        page = await self.get_current_page()
        try:
            page.remove_listener("response", self._on_response)
        except Exception:
            pass

    async def _on_response(self, response: Response):
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
