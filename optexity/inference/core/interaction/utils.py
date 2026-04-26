import asyncio
import base64
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
from pydantic import BaseModel as _PydanticBaseModel
from rapidfuzz import fuzz

from optexity.exceptions import (
    ElementNotFoundInAxtreeException,
)
from optexity.inference.agents.index_prediction.action_prediction_locator_axtree import (
    ActionPredictionLocatorAxtree,
)
from optexity.inference.core.vision.ocr.aws_textract import AWSTextract
from optexity.inference.core.vision.ocr.ocr import _cv2_to_bytes, _load_cv2
from optexity.inference.core.vision.ocr.tesseract import Tesseract
from optexity.inference.infra.browser import Browser
from optexity.inference.models import (
    GeminiModels,
    get_llm_model,
    get_llm_model_with_fallback,
    resolve_model_name,
)
from optexity.schema.memory import BrowserState, Memory
from optexity.schema.ocr import BoundingBox, OCRResult
from optexity.schema.task import Task
from optexity.utils.settings import settings

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
small_ocr = AWSTextract()
large_ocr = AWSTextract()

_KW_LLM_CANDIDATE_LIMIT = 30
_CROP_ELEMENT_WIDTH = 113
_CROP_ELEMENT_HEIGHT = 41
_CROP_PADDING_FACTOR = 2.0


class _KeywordLLMResult(_PydanticBaseModel):
    matched_index: int | None = None


def crop_screenshot_to_bbox(
    screenshot_b64: str, x1: int, y1: int, x2: int, y2: int
) -> str:
    img = _load_cv2(screenshot_b64)
    h, w = img.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    cropped = img[y1:y2, x1:x2]
    return base64.b64encode(_cv2_to_bytes(cropped)).decode("utf-8")


def _crop_for_llm(screenshot_b64: str, x: int, y: int) -> str:
    img = _load_cv2(screenshot_b64)
    h, w = img.shape[:2]
    half_w = _CROP_ELEMENT_WIDTH // 2
    half_h = _CROP_ELEMENT_HEIGHT // 2
    pad_x = int(_CROP_ELEMENT_WIDTH * _CROP_PADDING_FACTOR)
    pad_y = int(_CROP_ELEMENT_HEIGHT * _CROP_PADDING_FACTOR)
    x1 = max(0, x - half_w - pad_x)
    y1 = max(0, y - half_h - pad_y)
    x2 = min(w, x - half_w + _CROP_ELEMENT_WIDTH + pad_x)
    y2 = min(h, y - half_h + _CROP_ELEMENT_HEIGHT + pad_y)
    cropped = img[y1:y2, x1:x2]
    return base64.b64encode(_cv2_to_bytes(cropped)).decode("utf-8")


def resolve_bounding_box_variables(
    variables: list[str], memory: Memory
) -> tuple[int, int, int, int] | None:
    """Resolve bounding_box_variables [x1_var, y1_var, x2_var, y2_var] to pixel coords.

    Returns None if any variable is missing or equals -1.
    """
    import re as _re

    _VAR_RE = _re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\[(\d+)\])?$")

    def _resolve(v: str) -> int:
        key = v.strip("{}")
        m = _VAR_RE.match(key)
        if not m:
            raise KeyError(key)
        var_name, idx = m.group(1), m.group(2)
        val = memory.variables.generated_variables[var_name]
        if isinstance(val, list):
            if idx is None:
                raise ValueError(
                    f"Variable '{var_name}' is a list — use [{var_name}[0]] syntax"
                )
            return int(val[int(idx)])
        return int(val)

    try:
        vals = [_resolve(v) for v in variables]
        if any(v == -1 for v in vals):
            return None
        return (vals[0], vals[1], vals[2], vals[3])
    except (KeyError, ValueError, TypeError, IndexError):
        return None


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
        model_name = GeminiModels.GEMINI_2_5_COMPUTER_USE
    model = get_llm_model(model_name, True)
    coordinates, token_usage = model.get_computer_use_model_response(
        prompt=prompt_instructions,
        screenshot=screenshot_base64,
    )

    return coordinates


async def match_text_in_screenshot(
    memory: Memory,
    keyword: str,
    region_of_interest: BoundingBox | None = None,
    screenshot: str | bytes | None = None,
) -> OCRResult | None:
    """Find text on screenshot and return center coordinates.

    Single keyword: returns (x, y) or None.
    List of keywords: returns {keyword: (x, y) or None} for each keyword (single OCR call).
    """
    img = screenshot or (
        memory.browser_states[-1].screenshot if memory.browser_states else None
    )
    if img is None:
        logger.error("Screenshot is None or not a string")
        return None

    # Try ROI first if provided
    if region_of_interest is not None:
        results, base64_image_sent_to_ocr = small_ocr.ocr(
            img, region_of_interest=region_of_interest, padding_factor=4.0
        )
        logger.info(
            f"Found {len(results)} results using small OCR for keyword '{keyword}' in ROI"
        )
        for result in results:
            logger.info(f"Result: {result.text} with confidence {result.confidence}")
        result = find_keyword_in_results(results, keyword)
        memory.browser_states[-1].ocr_image_sent_to_ocr.append(base64_image_sent_to_ocr)
        if result is not None:
            logger.info(f"Found keyword using small OCR {keyword} in ROI: {result}")
            return result

    # Full screenshot fallback for any unmatched keywords

    logger.info("Could not find keyword using small OCR, trying large OCR...")
    results, base64_image_sent_to_ocr = large_ocr.ocr(img)
    memory.browser_states[-1].ocr_image_sent_to_ocr.append(base64_image_sent_to_ocr)
    result = find_keyword_in_results(results, keyword)

    if result is not None:
        logger.info(f"Found keyword using large OCR {keyword}: {result}")
        return result
    return None


async def match_all_text_in_screenshot(
    memory: Memory,
    keyword: str,
    region_of_interest: BoundingBox | None = None,
    screenshot: str | bytes | None = None,
) -> list[OCRResult]:
    """Find all matching text on screenshot for the keyword."""
    img = screenshot or (
        memory.browser_states[-1].screenshot if memory.browser_states else None
    )
    if img is None:
        logger.error("Screenshot is None or not a string")
        return []

    # Try ROI first if provided
    if region_of_interest is not None:
        results, base64_image_sent_to_ocr = small_ocr.ocr(
            img, region_of_interest=region_of_interest, padding_factor=1.0
        )
        for result in results:
            logger.info(f"Result: {result.text} with confidence {result.confidence}")

        matches = find_all_keyword_matches(results, keyword)
        for match in matches:
            logger.info(f"Match: {match.text} with confidence {match.confidence}")

        memory.browser_states[-1].ocr_image_sent_to_ocr.append(base64_image_sent_to_ocr)
        if matches:
            logger.info(
                f"Found {len(matches)} match(es) using small OCR for keyword '{keyword}' in ROI"
            )
            return matches

    # Full screenshot fallback
    logger.info("Could not find keyword using small OCR, trying large OCR...")
    results, base64_image_sent_to_ocr = large_ocr.ocr(img)
    memory.browser_states[-1].ocr_image_sent_to_ocr.append(base64_image_sent_to_ocr)
    matches = find_all_keyword_matches(results, keyword)

    if matches:
        logger.info(
            f"Found {len(matches)} match(es) using large OCR for keyword '{keyword}'"
        )
    return matches


def _join_adjacent_ocr_results(
    ocr_results: list[OCRResult], num_words: int
) -> list[OCRResult]:
    """Join horizontally adjacent OCR words into multi-word results.

    Only joins words whose y-coordinates are within 10% of each other's height,
    ensuring words from different lines are not merged.
    """
    if num_words <= 1 or len(ocr_results) < num_words:
        return []

    # Sort by y first (group lines), then by x within each line
    sorted_results = sorted(
        ocr_results, key=lambda r: (r.bounding_box.y, r.bounding_box.x)
    )

    joined = []
    for i in range(len(sorted_results) - num_words + 1):
        group = sorted_results[i : i + num_words]

        # Check all words in group have similar y (within 10% of avg height)
        avg_height = sum(r.bounding_box.height for r in group) / len(group)
        y_threshold = max(avg_height * 0.1, 5)
        base_y = group[0].bounding_box.y
        if any(abs(r.bounding_box.y - base_y) > y_threshold for r in group):
            continue

        # Sort group by x to ensure left-to-right order
        group = sorted(group, key=lambda r: r.bounding_box.x)

        # Build combined result
        combined_text = " ".join(r.text for r in group)
        min_x = group[0].bounding_box.x
        min_y = min(r.bounding_box.y for r in group)
        max_x_end = max(r.bounding_box.x + r.bounding_box.width for r in group)
        max_y_end = max(r.bounding_box.y + r.bounding_box.height for r in group)
        avg_confidence = sum(r.confidence for r in group) / len(group)

        logger.debug(
            f"Joined adjacent OCR words: '{combined_text}' at ({min_x}, {min_y})"
        )

        joined.append(
            OCRResult(
                text=combined_text,
                confidence=avg_confidence,
                bounding_box=BoundingBox(
                    x=min_x,
                    y=min_y,
                    width=max_x_end - min_x,
                    height=max_y_end - min_y,
                ),
            )
        )

    if joined:
        logger.info(
            f"Created {len(joined)} joined multi-word candidates from {len(ocr_results)} OCR results"
        )

    return joined


def _score(entry):
    return (entry[2] + entry[3] * 1.5) / 2.5


def find_keyword_in_results(
    ocr_results: list[OCRResult],
    keyword: str,
    score_threshold: float = 0.0,
) -> OCRResult | None:
    if not ocr_results:
        return None

    # Build candidate list: original results + joined multi-word results
    num_words = len(keyword.strip().split())
    candidates = list(ocr_results)
    if num_words > 1:
        joined = _join_adjacent_ocr_results(ocr_results, num_words)
        candidates.extend(joined)

    scored = []
    target = keyword.lower().strip()
    for i, candidate in enumerate(candidates):
        candidate_text = candidate.text.lower().strip()
        if target == candidate_text:
            scored.append((i, candidate, 100, 100, 100))
            continue
        scored.append(
            (
                i,
                candidate,
                fuzz.ratio(target, candidate_text),
                fuzz.partial_ratio(target, candidate_text),
            )
        )
    # combined score: weight partial higher
    scored.sort(key=_score, reverse=True)
    best = scored[0]
    best_score = _score(best)
    logger.info(
        f"Best match for keyword '{keyword}': '{best[1].text}' with score {best_score:.1f}"
    )
    if score_threshold > 0.0 and best_score < score_threshold:
        logger.info(
            f"Score {best_score:.1f} below threshold {score_threshold} for keyword '{keyword}'"
        )
        return None
    return best[1]


def find_all_keyword_matches(
    ocr_results: list[OCRResult], keyword: str
) -> list[OCRResult]:
    """Return all OCR results that share the same top fuzzy score for the keyword."""

    num_words = len(keyword.strip().split())
    candidates = list(ocr_results)
    if num_words > 1:
        joined = _join_adjacent_ocr_results(ocr_results, num_words)
        candidates.extend(joined)

    scored = []
    target = keyword.lower().strip()
    for i, candidate in enumerate(candidates):
        candidate_text = candidate.text.lower().strip()
        if target == candidate_text:
            scored.append((i, candidate, 100, 100))
            continue
        scored.append(
            (
                i,
                candidate,
                fuzz.ratio(target, candidate_text),
                fuzz.partial_ratio(target, candidate_text),
            )
        )

    if not scored:
        return []

    scores = [(s, (s[2] + s[3] * 1.5) / 2.5) for s in scored]
    best_score = max(sc for _, sc in scores)
    matches = [s[1] for s, sc in scores if sc == best_score]
    logger.info(
        f"find_all_keyword_matches for '{keyword}': best score {best_score:.1f}, "
        f"{len(matches)} match(es): {[m.text for m in matches]}"
    )
    return matches


def get_coordinates_from_ocr_result(ocr_result: OCRResult):
    return (
        int(ocr_result.bounding_box.x + ocr_result.bounding_box.width / 2),
        int(ocr_result.bounding_box.y + ocr_result.bounding_box.height / 2),
    )


async def resolve_keyword_with_llm_fallback(
    keyword: str,
    recording_x: int,
    recording_y: int,
    prompt_instructions: str,
    memory: Memory,
    task: Task,
    bounding_box: tuple[int, int, int, int] | None = None,
) -> tuple[int, int] | None:
    """OCR keyword search with LLM fallback.

    Flow:
    1. Run OCR (on bounding_box crop, ROI around recording coords, or full screenshot)
    2. Exact 100% match → return coords
    3. No exact match → top 30 by proximity → cropped screenshot + OCR list → LLM index pick
    """
    screenshot = memory.browser_states[-1].screenshot if memory.browser_states else None
    if screenshot is None:
        logger.error("[keyword_fallback] no screenshot in memory")
        return None

    is_negative_coords = recording_x == -1 and recording_y == -1

    # --- OCR phase ---
    if bounding_box is not None:
        x1, y1, x2, y2 = bounding_box
        cropped = crop_screenshot_to_bbox(screenshot, x1, y1, x2, y2)
        results, base64_sent = large_ocr.ocr(cropped)
        # offset bounding boxes back to full-screen coords
        for r in results:
            r.bounding_box.x += x1
            r.bounding_box.y += y1
        llm_screenshot = cropped
    elif is_negative_coords:
        results, base64_sent = large_ocr.ocr(screenshot)
        llm_screenshot = screenshot
    else:
        roi = BoundingBox(
            x=recording_x,
            y=recording_y,
            width=_CROP_ELEMENT_WIDTH,
            height=_CROP_ELEMENT_HEIGHT,
        )
        results, base64_sent = small_ocr.ocr(
            screenshot, region_of_interest=roi, padding_factor=1.0
        )
        if not results:
            results, base64_sent = large_ocr.ocr(screenshot)
        llm_screenshot = _crop_for_llm(screenshot, recording_x, recording_y)

    if memory.browser_states:
        memory.browser_states[-1].ocr_image_sent_to_ocr.append(base64_sent)

    # --- Exact match check ---
    exact = [r for r in results if r.text.strip().lower() == keyword.strip().lower()]
    if exact:
        ref_x = recording_x if not is_negative_coords else 0
        ref_y = recording_y if not is_negative_coords else 0
        best = min(
            exact,
            key=lambda r: math.sqrt(
                (r.bounding_box.x + r.bounding_box.width / 2 - ref_x) ** 2
                + (r.bounding_box.y + r.bounding_box.height / 2 - ref_y) ** 2
            ),
        )
        cx = int(best.bounding_box.x + best.bounding_box.width / 2)
        cy = int(best.bounding_box.y + best.bounding_box.height / 2)
        logger.info(f"[keyword_fallback] exact match '{best.text}' at ({cx}, {cy})")
        return cx, cy

    # --- LLM fallback: top 30 by proximity ---
    ref_x = recording_x if not is_negative_coords else 0
    ref_y = recording_y if not is_negative_coords else 0
    candidates = sorted(
        results,
        key=lambda r: math.sqrt(
            (r.bounding_box.x + r.bounding_box.width / 2 - ref_x) ** 2
            + (r.bounding_box.y + r.bounding_box.height / 2 - ref_y) ** 2
        ),
    )[:_KW_LLM_CANDIDATE_LIMIT]

    if not candidates:
        logger.info(f"[keyword_fallback] no OCR results for '{keyword}'")
        return None

    candidates_text = "\n".join(f"{i}: '{r.text}'" for i, r in enumerate(candidates))
    prompt = (
        f"You are verifying a UI automation step. The task is: '{prompt_instructions}'. "
        f"We are looking for: '{keyword}'.\n"
        f"Text elements detected on screen by OCR:\n{candidates_text}\n\n"
        "Which index best matches the keyword? Consider OCR typos, partial text, ellipsis "
        "truncation, and case differences (e.g. 'Prince Yadav' and 'Pricne yadav' are the same). "
        "Return matched_index (0-based) or null if none match."
    )

    model = get_llm_model_with_fallback(task.llm_provider, task.llm_model_name, True)
    raw_result, token_usage = model.get_model_response_with_structured_output(
        prompt=prompt,
        response_schema=_KeywordLLMResult,
        screenshot=llm_screenshot,
    )
    result = _KeywordLLMResult.model_validate(raw_result.model_dump())
    memory.token_usage += token_usage
    logger.info(
        f"[keyword_fallback] LLM matched index={result.matched_index} for '{keyword}'"
    )

    if result.matched_index is None:
        return None

    matched = candidates[result.matched_index]
    cx = int(matched.bounding_box.x + matched.bounding_box.width / 2)
    cy = int(matched.bounding_box.y + matched.bounding_box.height / 2)
    return cx, cy
