import asyncio
import base64
import json
import logging
import re
from typing import Literal
from uuid import uuid4

import patchright.async_api
import playwright.async_api
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
        headless: bool = False,
        stealth: bool = True,
        backend: Literal["browser-use", "browserbase"] = "browser-use",
        debug_port: int = 9222,
        channel: Literal["chromium", "chrome"] = "chromium",
        use_proxy: bool = False,
        proxy_session_id: str | None = None,
    ):

        self.headless = headless
        self.stealth = stealth
        self.backend = backend
        self.debug_port = debug_port

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
        self.channel: Literal["chrome", "chromium"] = channel
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
            await self.stop()

            if self.stealth:
                from patchright.async_api import async_playwright
            else:
                from playwright.async_api import async_playwright

            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.connect_over_cdp(self.cdp_url)
            self.context = self.browser.contexts[0]
            if self.context is None:
                raise ValueError("Context is not set")
            if len(self.context.pages) == 0:
                self.page = await self.context.new_page()
            else:
                for i in range(len(self.context.pages) - 1, 0, -1):
                    await self.context.pages[i].close()

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

        if self.browser is not None:
            logger.debug("Stopping browser")
            await self.browser.close()
            self.browser = None

        if self.playwright is not None:
            logger.debug("Stopping playwright")
            await self.playwright.stop()
            self.playwright = None

        self.context = None

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

    async def handle_new_tabs(self, max_wait_time: float) -> tuple[bool, float]:

        if self.context is None or self.backend_agent is None:
            return False, 0

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
        if self.context is None or self.backend_agent is None:
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
        if self.context is None or self.backend_agent is None:
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

    async def get_locator_from_command(self, command: str) -> Locator | None:
        if self.context is None or self.backend_agent is None:
            return None
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
                return None
            await page.goto(url, timeout=10000)
        except (TimeoutError, PatchrightTimeoutError, PlaywrightTimeoutError):
            pass

    async def get_browser_state_summary(self) -> BrowserStateSummary:
        if self.backend_agent is None:
            raise ValueError("Backend agent is not set")

        browser_state_summary = await self.backend_agent.browser_session.get_browser_state_summary(
            include_screenshot=True,  # always capture even if use_vision=False so that cloud sync is useful (it's fast now anyway)
            include_recent_events=False,
            cached=False,
        )

        return browser_state_summary

    async def get_current_page_url(self) -> str:
        try:
            page = await self.get_current_page()
            if page is None:
                return "about:blank"
            return page.url
        except Exception as e:
            logger.error(f"Error getting current page URL: {e}")
            return "about:blank"

    async def get_current_page_title(self) -> str:
        try:
            page = await self.get_current_page()
            if page is None:
                return "Unknown page title"
            return await page.title()
        except Exception as e:
            logger.error(f"Error getting current page title: {e}")
            return "Unknown page title"

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
            content_type = (resp.headers.get("content-type") or "").lower()
            content_disposition = (
                resp.headers.get("content-disposition") or ""
            ).lower()

            # PDF: either content-type is application/pdf, or attachment with .pdf filename
            # (many servers use application/octet-stream + content-disposition for PDFs)
            is_pdf_content = "application/pdf" in content_type
            is_pdf_attachment = (
                "attachment" in content_disposition and ".pdf" in content_disposition
            )
            if not (is_pdf_content or is_pdf_attachment):
                if self.active_downloads == 0:
                    self.all_active_downloads_done.set()
                return

            self.active_downloads += 1
            self.all_active_downloads_done.clear()

            filename = f"{uuid4()}.pdf"
            if content_disposition:
                match = re.search(
                    r'filename\*?=(?:utf-8\'\')?"?([^";]+)"?',
                    content_disposition,
                    re.IGNORECASE,
                )
                if match:
                    filename = match.group(1).strip()

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
