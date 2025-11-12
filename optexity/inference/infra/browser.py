import asyncio
import json
import logging
from typing import Literal

from browser_use import Agent, BrowserSession, ChatGoogle
from browser_use.browser.views import BrowserStateSummary
from playwright.async_api import Locator, Response

from optexity.schema.memory import NetworkResponse

logger = logging.getLogger(__name__)


class Browser:
    def __init__(
        self,
        user_data_dir: str = None,
        headless: bool = False,
        proxy: str = None,
        stealth: bool = True,
        backend: Literal["browser-use", "browserbase"] = "browser-use",
        debug_port: int = 9222,
    ):

        if proxy:
            proxy = proxy.removeprefix("http://").removeprefix("https://")
            self.proxy = "http://" + proxy

        self.headless = headless
        self.stealth = stealth
        self.user_data_dir = user_data_dir
        self.backend = backend
        self.debug_port = debug_port

        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.cdp_url = f"http://localhost:{self.debug_port}"
        self.backend_agent = None

        self.page_to_target_id = []
        self.previous_total_pages = 0

        self.network_calls: list[NetworkResponse] = []

    async def start(self):
        logger.debug("Starting browser")
        try:
            if self.playwright is not None:
                await self.playwright.stop()

            if self.stealth:
                from patchright.async_api import async_playwright
            else:
                from playwright.async_api import async_playwright

            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                channel="chromium",
                headless=self.headless,
                args=[
                    "--start-fullscreen",
                    f"--remote-debugging-port={self.debug_port}",
                ],
                chromium_sandbox=False,
            )

            self.context = await self.browser.new_context(no_viewport=True)
            self.page = await self.context.new_page()

            browser_session = BrowserSession(cdp_url=self.cdp_url)

            llm = ChatGoogle(model="gemini-flash-latest")

            self.backend_agent = Agent(
                task="",
                llm=llm,
                browser_session=browser_session,
                use_vision=False,
            )

            await self.backend_agent.browser_session.start()

            logger.debug("Browser started successfully")

        except Exception as e:
            logger.error(f"Error starting playwright: {e}")
            raise e

    async def stop(self):
        if self.playwright is not None:
            await self.backend_agent.browser_session.stop()
            await self.context.close()
            await self.browser.close()
            await self.playwright.stop()
            self.playwright = None
            self.browser = None
            self.context = None
            self.page = None

    async def get_current_page(self):
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

    async def get_locator_from_command(self, command: str) -> Locator:
        page = await self.get_current_page()
        if page is None:
            return None
        locator: Locator = eval(f"page.{command}")
        return locator

    # async def get_axtree(self) -> str:
    #     browser_state_summary = await self.backend_agent.browser_session.get_browser_state_summary(
    #         include_screenshot=True,  # always capture even if use_vision=False so that cloud sync is useful (it's fast now anyway)
    #         include_recent_events=self.backend_agent.include_recent_events,
    #     )

    #     llm_representation = browser_state_summary.dom_state.llm_representation()

    #     return llm_representation

    def get_xpath_from_index(self, index: int) -> str:
        raise NotImplementedError("Not implemented")

    async def click_index(self, index: int):
        action_model = self.backend_agent.ActionModel(
            **{"click": {"index": int(index)}}
        )
        results = await self.backend_agent.multi_act([action_model])

    async def input_text_index(self, index: int, text: str):
        action_model = self.backend_agent.ActionModel(
            **{"input": {"index": int(index), "text": text, "clear": True}}
        )
        results = await self.backend_agent.multi_act([action_model])

    async def go_to_url(self, url: str):
        page = await self.get_current_page()
        if page is None:
            return None
        await page.goto(url)

    async def get_browser_state_summary(self) -> BrowserStateSummary:
        browser_state_summary = await self.backend_agent.browser_session.get_browser_state_summary(
            include_screenshot=True,  # always capture even if use_vision=False so that cloud sync is useful (it's fast now anyway)
            include_recent_events=self.backend_agent.include_recent_events,
            cached=False,
        )

        return browser_state_summary

    async def get_current_page_url(self) -> str:
        page = await self.get_current_page()
        if page is None:
            return None
        return page.url

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
