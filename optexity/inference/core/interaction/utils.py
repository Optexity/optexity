import asyncio
import logging
import math
import os
import shutil
import uuid
from pathlib import Path
from typing import Callable, Union

import aiofiles
import patchright.async_api
import playwright.async_api

from optexity.exceptions import (
    ElementNotFoundInAxtreeException,
)
from optexity.inference.agents.index_prediction.action_prediction_locator_axtree import (
    ActionPredictionLocatorAxtree,
)
from optexity.inference.infra.browser import Browser
from optexity.inference.models import get_llm_model_with_fallback
from optexity.schema.memory import BrowserState, Memory
from optexity.schema.task import Task

logger = logging.getLogger(__name__)

Page = Union[playwright.async_api.Page, patchright.async_api.Page]

_HIGHLIGHT_JS_INJECT = """
(bbox) => {
    const el = document.createElement('div');
    el.id = '__optexity_element_highlight';
    el.style.position = 'fixed';
    el.style.left = `${bbox.x}px`;
    el.style.top = `${bbox.y}px`;
    el.style.width = `${bbox.width}px`;
    el.style.height = `${bbox.height}px`;
    el.style.border = '3px solid red';
    el.style.background = 'rgba(255, 0, 0, 0.15)';
    el.style.zIndex = '2147483647';
    el.style.pointerEvents = 'none';
    el.style.boxSizing = 'border-box';
    document.body.appendChild(el);
    return el.id;
}
"""

_HIGHLIGHT_JS_REMOVE = """
(id) => {
    const el = document.getElementById(id);
    if (el) el.remove();
}
"""


async def highlight_element_and_screenshot(
    page: Page, browser: Browser, bbox: dict
) -> str | None:
    """Inject a bounding-box highlight overlay, take a screenshot, then remove
    the overlay.  Returns the base64 screenshot or ``None`` on failure."""
    highlight_id: str | None = None
    try:
        logger.debug(f"Injecting highlight overlay for bbox: {bbox}")
        highlight_id = await page.evaluate(_HIGHLIGHT_JS_INJECT, bbox)
        screenshot = await browser.get_screenshot()
        logger.debug(f"Screenshot captured successfully")
        return screenshot
    except Exception as e:
        logger.error(f"highlight_element_and_screenshot failed: {e}")
        return None
    finally:
        if highlight_id is not None:
            try:
                await page.evaluate(_HIGHLIGHT_JS_REMOVE, highlight_id)
                logger.debug(f"Highlight removed successfully")
            except Exception as e:
                logger.warning(f"Failed to remove highlight {highlight_id}: {e}")


async def get_element_viewport_bbox_by_index(
    browser: Browser, index: int
) -> dict | None:
    """Resolve an element *index* (backend_node_id) to a viewport-coordinate
    bounding box ``{x, y, width, height}``.  Returns ``None`` when the
    position cannot be determined."""
    logger.debug(f"Getting viewport bbox for element index: {index}")

    def _rect_to_bbox(rect) -> dict | None:
        if rect is None:
            return None
        try:
            x = float(getattr(rect, "x"))
            y = float(getattr(rect, "y"))
            width = float(getattr(rect, "width"))
            height = float(getattr(rect, "height"))
        except Exception:
            return None

        if not all(math.isfinite(v) for v in (x, y, width, height)):
            return None
        if width <= 0 or height <= 0:
            return None

        return {"x": x, "y": y, "width": width, "height": height}

    try:
        backend_agent = browser.backend_agent
        if backend_agent is None or backend_agent.browser_session is None:
            return None

        element = await backend_agent.browser_session.get_dom_element_by_index(index)
        if element is None:
            return None

        client_bbox = None
        if element.snapshot_node and element.snapshot_node.clientRects:
            client_bbox = _rect_to_bbox(element.snapshot_node.clientRects)

        abs_doc_bbox = _rect_to_bbox(element.absolute_position)
        abs_viewport_bbox = None

        page = await browser.get_current_page()
        if abs_doc_bbox and page is not None:
            scroll = await page.evaluate("({x: window.scrollX, y: window.scrollY})")
            abs_viewport_bbox = {
                "x": abs_doc_bbox["x"] - float(scroll["x"]),
                "y": abs_doc_bbox["y"] - float(scroll["y"]),
                "width": abs_doc_bbox["width"],
                "height": abs_doc_bbox["height"],
            }

        if client_bbox and abs_viewport_bbox:
            # In practice, some snapshot client rects resolve to (0,0) for nodes
            # that are not actually at the viewport origin (e.g. frame/local coords).
            client_near_origin = (
                abs(client_bbox["x"]) <= 1 and abs(client_bbox["y"]) <= 1
            )
            abs_not_near_origin = (
                abs(abs_viewport_bbox["x"]) > 5 or abs(abs_viewport_bbox["y"]) > 5
            )
            if client_near_origin and abs_not_near_origin:
                return abs_viewport_bbox

            # Prefer absolute coordinates when both are available because they are
            # translated to top-page coordinates (better for iframes/shadow contexts).
            return abs_viewport_bbox

        if abs_viewport_bbox:
            logger.debug(
                f"Using absolute viewport bbox for index {index}: {abs_viewport_bbox}"
            )
            return abs_viewport_bbox

        if client_bbox:
            logger.debug(f"Using client bbox for index {index}: {client_bbox}")
            return client_bbox
    except Exception as e:
        logger.error(
            f"get_element_viewport_bbox_by_index failed for index {index}: {e}"
        )
    logger.warning(f"Could not determine viewport bbox for element index {index}")
    return None


async def update_screenshot_with_highlight(
    browser: Browser, memory: Memory, index: int
) -> None:
    """Highlight the element at *index* and update the last browser-state screenshot."""
    logger.info(f"Updating screenshot with highlight for element index: {index}")
    page = await browser.get_current_page()
    if page is None:
        logger.warning(f"Cannot update screenshot highlight - current page is None")
        return
    bbox = await get_element_viewport_bbox_by_index(browser, index)
    if bbox is None:
        return
    highlighted = await highlight_element_and_screenshot(page, browser, bbox)
    if highlighted:
        memory.browser_states[-1].screenshot = highlighted
        logger.info(
            f"Successfully updated screenshot with highlight for element index {index}"
        )
    else:
        logger.warning(
            f"Failed to capture highlighted screenshot for element index {index}"
        )


_index_prediction_cache: dict[tuple, ActionPredictionLocatorAxtree] = {}


def _get_index_prediction_agent(task: "Task") -> ActionPredictionLocatorAxtree:
    cache_key = (task.llm_provider, task.llm_model_name)
    if cache_key not in _index_prediction_cache:
        model = get_llm_model_with_fallback(
            task.llm_provider, task.llm_model_name, True
        )
        _index_prediction_cache[cache_key] = ActionPredictionLocatorAxtree(model)
    return _index_prediction_cache[cache_key]


async def get_index_from_prompt(
    memory: Memory, prompt_instructions: str, browser: Browser, task: Task
):
    browser_state_summary = await browser.get_browser_state_summary()
    memory.browser_states[-1] = BrowserState(
        url=browser_state_summary.url,
        screenshot=browser_state_summary.screenshot,
        title=browser_state_summary.title,
        axtree=browser_state_summary.dom_state.llm_representation(
            remove_empty_nodes=task.automation.remove_empty_nodes_in_axtree
        ),
    )

    try:
        if memory.browser_states[-1].axtree is None:
            logger.error("Axtree is None, cannot predict action")
            return None
        final_prompt, response, token_usage = _get_index_prediction_agent(
            task
        ).predict_action(
            prompt_instructions,
            memory.browser_states[-1].axtree,
            can_return_negative_index=task.version == "v2",
        )
        memory.token_usage += token_usage
        memory.browser_states[-1].final_prompt = final_prompt
        memory.browser_states[-1].llm_response = response.model_dump()

        if response.index == -1:
            raise ElementNotFoundInAxtreeException(
                message=f"Element not found in the axtree: {prompt_instructions}",
                original_error=Exception(
                    f"Element not found in the axtree: {prompt_instructions}"
                ),
                command=prompt_instructions,
            )

        return response.index
    except ElementNotFoundInAxtreeException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in get_index_from_prompt: {e}")


def _snapshot_dir(directory: str) -> dict[str, float]:
    """Return {filename: mtime} for all files in directory."""
    result = {}
    try:
        for entry in os.scandir(directory):
            if entry.is_file():
                result[entry.name] = entry.stat().st_mtime
    except FileNotFoundError:
        pass
    return result


async def _wait_for_file_stable(
    path: Path, timeout: float = 5.0, interval: float = 0.3
) -> bool:
    """Wait until a file's size stops changing (download finished writing)."""
    prev_size = -1
    elapsed = 0.0
    while elapsed < timeout:
        try:
            size = path.stat().st_size
        except OSError:
            await asyncio.sleep(interval)
            elapsed += interval
            continue
        if size > 0 and size == prev_size:
            return True
        prev_size = size
        await asyncio.sleep(interval)
        elapsed += interval
    return prev_size > 0


async def handle_download(
    func: Callable, memory: Memory, browser: Browser, task: Task, download_filename: str
):
    download_path: Path = task.downloads_directory / download_filename

    before = _snapshot_dir(browser.temp_downloads_dir)

    # ---- Fallback-only signal collection (does not affect the main path) ----
    # Some sites (e.g. ASP.NET reports) open a popup that performs the
    # download. The popup may take longer than the primary 30s window to
    # produce the file, or the download event may fire on the new tab rather
    # than the current page. We attach lightweight observers here purely so
    # the fallback below can decide whether a download is genuinely in
    # flight, in which case we extend the wait. If none of these signals
    # fire we behave exactly like before (timeout + error log).
    download_event = asyncio.Event()
    new_popup_pages: list = []
    listener_cleanup: list = []

    def _on_download_event(_dl):
        download_event.set()

    def _on_context_page(p):
        new_popup_pages.append(p)
        try:
            p.on("download", _on_download_event)
            listener_cleanup.append((p, "download", _on_download_event))
        except Exception as e:
            logger.debug(
                f"handle_download: could not attach popup download listener: {e}"
            )

    current_page = None
    try:
        current_page = await browser.get_current_page()
    except Exception as e:
        logger.debug(
            f"handle_download: could not get current page for fallback listener: {e}"
        )

    if current_page is not None:
        try:
            current_page.on("download", _on_download_event)
            listener_cleanup.append((current_page, "download", _on_download_event))
        except Exception as e:
            logger.debug(
                f"handle_download: could not attach page download listener: {e}"
            )

    if browser.context is not None:
        try:
            browser.context.on("page", _on_context_page)
            listener_cleanup.append((browser.context, "page", _on_context_page))
        except Exception as e:
            logger.debug(
                f"handle_download: could not attach context page listener: {e}"
            )

    try:
        # page = await browser.get_current_page()
        # async with page.expect_download() as download_info:
        await func()
        # download = await download_info.value
        # logger.info(f"Suggested filename: {download.suggested_filename}")

        timeout = 30.0
        poll_interval = 0.5
        elapsed = 0.0
        new_file: str | None = None

        while elapsed < timeout:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
            after = _snapshot_dir(browser.temp_downloads_dir)
            new_files = [
                name
                for name in after
                if name not in before
                and not name.endswith(".crdownload")
                and not name.endswith(".tmp")
            ]
            if new_files:
                new_file = max(new_files, key=lambda n: after[n])
                break

        # ---- Fallback: extend the wait only if we have evidence a download
        # is actually in flight. This keeps the working path untouched while
        # rescuing slow popups / slow servers that today fail at 30s.
        if new_file is None:
            after = _snapshot_dir(browser.temp_downloads_dir)
            in_progress_files = [
                name
                for name in after
                if name not in before
                and (name.endswith(".crdownload") or name.endswith(".tmp"))
            ]
            has_signal = (
                download_event.is_set()
                or len(new_popup_pages) > 0
                or len(in_progress_files) > 0
            )
            if has_signal:
                extra_timeout = 60.0
                logger.warning(
                    f"handle_download: primary {timeout}s window elapsed without a "
                    f"finalized file; extending by {extra_timeout}s "
                    f"(download_event={download_event.is_set()}, "
                    f"popups={len(new_popup_pages)}, "
                    f"in_progress={in_progress_files})"
                )
                extra_elapsed = 0.0
                while extra_elapsed < extra_timeout:
                    await asyncio.sleep(poll_interval)
                    extra_elapsed += poll_interval
                    after = _snapshot_dir(browser.temp_downloads_dir)
                    new_files = [
                        name
                        for name in after
                        if name not in before
                        and not name.endswith(".crdownload")
                        and not name.endswith(".tmp")
                    ]
                    if new_files:
                        new_file = max(new_files, key=lambda n: after[n])
                        logger.info(
                            f"handle_download: recovered download via extended wait after "
                            f"{timeout + extra_elapsed:.1f}s total"
                        )
                        break

        if new_file is None:
            logger.error(
                f"No new file appeared in {browser.temp_downloads_dir} within {timeout}s after download action"
            )
            return
    finally:
        for target, event_name, handler in listener_cleanup:
            try:
                target.remove_listener(event_name, handler)
            except Exception:
                pass

    src_path = Path(browser.temp_downloads_dir) / new_file

    if not await _wait_for_file_stable(src_path):
        logger.warning(f"Downloaded file {src_path} may be incomplete")

    try:
        uuid.UUID(download_path.stem)
        is_uuid_filename = True
    except Exception:
        is_uuid_filename = False

    if is_uuid_filename:
        download_path = task.downloads_directory / new_file
    elif not download_path.suffix:
        suffix = Path(new_file).suffix
        if suffix:
            download_path = download_path.with_suffix(suffix)

    shutil.move(str(src_path), str(download_path))
    logger.info(f"Moved download {src_path} -> {download_path}")

    # await clean_download(download_path)

    if download_path.exists() and download_path.stat().st_size > 0:
        memory.downloads.append(download_path)
    else:
        logger.error(f"Download file is empty or missing: {download_path}")


async def clean_download(download_path: Path):
    return

    if download_path.suffix == ".csv":
        # Read full file
        async with aiofiles.open(download_path, "r", encoding="utf-8") as f:
            content = await f.read()
        # Remove everything between <script>...</script> (multiline safe)

        if "</script>" in content:
            clean_content = content.split("</script>")[-1]

            # Write cleaned CSV back
            async with aiofiles.open(download_path, "w", encoding="utf-8") as f:
                await f.write(clean_content)
