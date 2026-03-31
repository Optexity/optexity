import asyncio
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Callable
from urllib.parse import urljoin

import aiofiles
import httpx

from optexity.exceptions import (
    ElementNotFoundInAxtreeException,
)
from optexity.inference.agents.index_prediction.action_prediction_locator_axtree import (
    ActionPredictionLocatorAxtree,
)
from optexity.inference.core.vision.ocr.aws_textract import AWSTextract
from optexity.inference.core.vision.ocr.ocr import OCRModels
from optexity.inference.core.vision.ocr.tesseract import Tesseract
from optexity.inference.infra.browser import Browser
from optexity.inference.models import GeminiModels, get_llm_model, resolve_model_name
from optexity.schema.memory import BrowserState, Memory
from optexity.schema.ocr import BoundingBox
from optexity.schema.task import Task
from optexity.utils.settings import settings

logger = logging.getLogger(__name__)

_index_prediction_cache: dict[tuple, ActionPredictionLocatorAxtree] = {}


def _get_index_prediction_agent(task: "Task") -> ActionPredictionLocatorAxtree:
    cache_key = (task.llm_provider, task.llm_model_name)
    if cache_key not in _index_prediction_cache:
        model = get_llm_model(
            resolve_model_name(task.llm_provider, task.llm_model_name), True
        )
        _index_prediction_cache[cache_key] = ActionPredictionLocatorAxtree(model)
    return _index_prediction_cache[cache_key]


async def get_index_from_prompt(
    memory: Memory, prompt_instructions: str, browser: Browser, task: Task
):
    browser_state = await browser.get_browser_state_summary(
        remove_empty_nodes=task.automation.remove_empty_nodes_in_axtree
    )
    memory.browser_states[-1] = browser_state

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

    if new_file is None:
        logger.error(
            f"No new file appeared in {browser.temp_downloads_dir} within {timeout}s after download action"
        )
        return

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


async def get_coordinates_from_prompt(
    memory: Memory, prompt_instructions: str, browser: Browser, task: Task
):
    ## call optexity api to get coordinates from prompt
    browser_state = await browser.get_browser_state_summary(
        remove_empty_nodes=task.automation.remove_empty_nodes_in_axtree
    )
    memory.browser_states[-1] = browser_state

    screenshot_base64 = browser_state.screenshot
    if screenshot_base64 is None or not isinstance(screenshot_base64, str):
        logger.error("Screenshot is None or not a string")
        return None

    model_name = resolve_model_name(task.llm_provider, task.llm_model_name)
    if not model_name.is_computer_use_model():
        # Fall back to Gemini for computer use if the task's model doesn't support it
        model_name = GeminiModels.GEMINI_3_FLASH
    model = get_llm_model(model_name, True)
    coordinates, token_usage = model.get_computer_use_model_response(
        prompt=prompt_instructions,
        screenshot=screenshot_base64,
    )

    return coordinates


async def match_text_in_screenshot(
    memory: Memory,
    keyword: str | list[str],
    region_of_interest: BoundingBox | None = None,
    screenshot: str | bytes | None = None,
) -> tuple[int, int] | None | dict[str, tuple[int, int] | None]:
    """Find text on screenshot and return center coordinates.

    Single keyword: returns (x, y) or None.
    List of keywords: returns {keyword: (x, y) or None} for each keyword (single OCR call).
    """
    ocr = AWSTextract()
    img = screenshot or (
        memory.browser_states[-1].screenshot if memory.browser_states else None
    )
    if img is None:
        logger.error("Screenshot is None or not a string")
        return None if isinstance(keyword, str) else {k: None for k in keyword}

    # Save screenshot for debugging
    try:
        import base64 as _b64

        debug_path = "/tmp/ocr_debug_screenshot.png"
        if isinstance(img, str):
            with open(debug_path, "wb") as f:
                f.write(_b64.b64decode(img))
        elif isinstance(img, bytes):
            with open(debug_path, "wb") as f:
                f.write(img)
        logger.info(f"Saved OCR debug screenshot to {debug_path}")
    except Exception as e:
        logger.error(f"Failed to save debug screenshot: {e}")

    batch_mode = isinstance(keyword, list)
    keywords = keyword if batch_mode else [keyword]

    def _find_in_results(ocr_results, keywords_to_find):
        logger.info(f"ocr_results: {ocr_results}")
        logger.info(f"keywords_to_find: {keywords_to_find}")
        found = {}
        for kw in keywords_to_find:
            kw_lower = kw.lower().strip()
            match = None
            # Exact match
            for r in ocr_results:
                if r.text.lower().strip() == kw_lower:
                    match = (
                        int(r.bounding_box.x + r.bounding_box.width / 2),
                        int(r.bounding_box.y + r.bounding_box.height / 2),
                    )
                    break
            # Substring match fallback
            if match is None:
                for r in ocr_results:
                    r_text = r.text.lower().strip()
                    if kw_lower in r_text or r_text in kw_lower:
                        match = (
                            int(r.bounding_box.x + r.bounding_box.width / 2),
                            int(r.bounding_box.y + r.bounding_box.height / 2),
                        )
                        break
            found[kw] = match
        return found

    # Try ROI first if provided
    if region_of_interest is not None:
        results = ocr.ocr(
            img, region_of_interest=region_of_interest, padding_factor=4.0
        )
        found = _find_in_results(results, keywords)
        remaining = [kw for kw in keywords if found[kw] is None]
    else:
        found = {kw: None for kw in keywords}
        remaining = keywords

    # Full screenshot fallback for any unmatched keywords
    if remaining:
        results = ocr.ocr(img)
        full_found = _find_in_results(results, remaining)
        found.update(full_found)

    if batch_mode:
        return found
    return found[keywords[0]]
