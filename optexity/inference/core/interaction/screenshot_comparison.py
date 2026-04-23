import asyncio
import base64
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
from pydantic import BaseModel

from optexity.exceptions import KeywordNotFoundOnScreenException
from optexity.inference.core.vision.ocr.aws_textract import AWSTextract
from optexity.inference.core.vision.ocr.ocr import _cv2_to_bytes, _load_cv2
from optexity.inference.models import (
    GeminiModels,
    get_llm_model,
    get_llm_model_with_fallback,
    resolve_model_name,
)
from optexity.schema.memory import Memory
from optexity.schema.ocr import OCRResult
from optexity.schema.task import Task

logger = logging.getLogger(__name__)
_DEBUG_DIR = Path("/tmp/screenshot_comparison_debug")
_ocr = AWSTextract()

_CROP_ELEMENT_WIDTH = 113
_CROP_ELEMENT_HEIGHT = 41
_CROP_PADDING_FACTOR = 2.0
_OCR_CANDIDATE_LIMIT = 20


class ScreenshotMatchResult(BaseModel):
    matches: bool
    reason: str


class KeywordMatchResult(BaseModel):
    matched_index: int | None = None


def _crop_at(screenshot_b64: str, x: int, y: int) -> str:
    img = _load_cv2(screenshot_b64)
    img_h, img_w = img.shape[:2]
    half_w = _CROP_ELEMENT_WIDTH // 2
    half_h = _CROP_ELEMENT_HEIGHT // 2
    pad_x = int(_CROP_ELEMENT_WIDTH * _CROP_PADDING_FACTOR)
    pad_y = int(_CROP_ELEMENT_HEIGHT * _CROP_PADDING_FACTOR)
    x1 = max(0, x - half_w - pad_x)
    y1 = max(0, y - half_h - pad_y)
    x2 = min(img_w, x - half_w + _CROP_ELEMENT_WIDTH + pad_x)
    y2 = min(img_h, y - half_h + _CROP_ELEMENT_HEIGHT + pad_y)
    cropped = img[y1:y2, x1:x2]
    return base64.b64encode(_cv2_to_bytes(cropped)).decode("utf-8")


def _build_composite(left_b64: str, right_b64: str) -> str:
    left = _load_cv2(left_b64)
    right = _load_cv2(right_b64)
    target_h = max(left.shape[0], right.shape[0])

    def _pad(img: np.ndarray) -> np.ndarray:
        diff = target_h - img.shape[0]
        if diff > 0:
            pad = np.full((diff, img.shape[1], 3), 255, dtype=np.uint8)
            return np.vstack([img, pad])
        return img

    divider = np.full((target_h, 4, 3), 200, dtype=np.uint8)
    combined = np.hstack([_pad(left), divider, _pad(right)])
    return base64.b64encode(_cv2_to_bytes(combined)).decode("utf-8")


def _ocr_center(result: OCRResult) -> tuple[int, int]:
    return (
        int(result.bounding_box.x + result.bounding_box.width / 2),
        int(result.bounding_box.y + result.bounding_box.height / 2),
    )


def _distance(result: OCRResult, x: int, y: int) -> float:
    cx, cy = _ocr_center(result)
    return math.sqrt((cx - x) ** 2 + (cy - y) ** 2)


async def _fetch_screenshot(url: str) -> str:
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        return base64.b64encode(response.content).decode("utf-8")


def _save_debug(image_b64: str, suffix: str) -> Optional[Path]:
    try:
        _DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = _DEBUG_DIR / f"{ts}_{suffix}.png"
        path.write_bytes(base64.b64decode(image_b64))
        return path
    except Exception as e:
        logger.warning(f"[screenshot_comparison] could not save debug image: {e}")
        return None


async def _computer_use_coordinates(
    screenshot_b64: str,
    prompt_instructions: str,
    task: Task,
    memory: Memory,
) -> tuple[int, int] | None:
    model_name = resolve_model_name(task.llm_provider, task.llm_model_name)
    if not model_name.is_computer_use_model():
        model_name = GeminiModels.GEMINI_3_FLASH
    model = get_llm_model(model_name, True)
    coordinates, token_usage = model.get_computer_use_model_response(
        prompt=prompt_instructions,
        screenshot=screenshot_b64,
    )
    memory.token_usage += token_usage
    logger.info(f"[screenshot_comparison] computer use coordinates: {coordinates}")
    return coordinates


async def _llm_crop_comparison(
    recording_crop_b64: str,
    current_crop_b64: str,
    prompt_instructions: str,
    task: Task,
    memory: Memory,
) -> tuple[bool, str, ScreenshotMatchResult]:
    composite_b64 = _build_composite(recording_crop_b64, current_crop_b64)
    path = _save_debug(composite_b64, "composite")

    prompt = (
        f"You are verifying a UI automation step. The task is: '{prompt_instructions}'.\n\n"
        "You are shown a composite image split by a vertical divider into two halves:\n"
        "- LEFT HALF: the expected UI element from a recording (reference only)\n"
        "- RIGHT HALF: the current state of the screen (this is what you must evaluate)\n\n"
        "YOUR TASK: Does the RIGHT HALF show the same UI element as the LEFT HALF?\n\n"
        "Answer matches=true ONLY if the RIGHT HALF clearly shows the same specific element "
        "visible in the LEFT HALF (same control type, same context, same screen).\n\n"
        "Answer matches=false if ANY of these are true about the RIGHT HALF:\n"
        "- The RIGHT HALF is empty, gray, blank, or has no meaningful UI\n"
        "- The RIGHT HALF shows a completely different screen, page, or dialog\n"
        "- The expected element from the LEFT HALF is not visible in the RIGHT HALF\n"
        "- The RIGHT HALF shows a loading, error, or login screen\n\n"
        "IMPORTANT: Only the RIGHT HALF matters for your decision. "
        "Describe what you see in each half separately and explain your decision."
    )

    model = get_llm_model_with_fallback(task.llm_provider, task.llm_model_name, True)
    raw_result, token_usage = model.get_model_response_with_structured_output(
        prompt=prompt,
        response_schema=ScreenshotMatchResult,
        screenshot=composite_b64,
    )
    result = ScreenshotMatchResult.model_validate(raw_result.model_dump())
    memory.token_usage += token_usage
    logger.info(
        f"[screenshot_comparison] crop comparison matches={result.matches} "
        f"reason='{result.reason}' (debug: {path})"
    )
    return result.matches, composite_b64, result


async def _llm_keyword_fallback(
    keyword: str,
    prompt_instructions: str,
    candidates: list[OCRResult],
    screenshot_b64: str,
    task: Task,
    memory: Memory,
) -> int | None:
    candidates_text = "\n".join(f"{i}: '{r.text}'" for i, r in enumerate(candidates))
    prompt = (
        f"You are verifying a UI automation step. The task is: '{prompt_instructions}'. "
        f"We are looking for: '{keyword}'.\n"
        f"Text elements detected on screen by OCR:\n{candidates_text}\n\n"
        "Which index best matches the keyword? Consider OCR typos, partial text, ellipsis "
        "truncation, and case differences (e.g. 'Prince Yadav' and 'Pricne yadav' are the same). "
        "Return matched_index (0-based) or null if none match."
    )
    path = _save_debug(screenshot_b64, "llm_keyword_fallback")

    model = get_llm_model_with_fallback(task.llm_provider, task.llm_model_name, True)
    raw_result, token_usage = model.get_model_response_with_structured_output(
        prompt=prompt,
        response_schema=KeywordMatchResult,
        screenshot=screenshot_b64,
    )
    result = KeywordMatchResult.model_validate(raw_result.model_dump())
    memory.token_usage += token_usage
    logger.info(
        f"[screenshot_comparison] keyword fallback: '{keyword}' → matched_index={result.matched_index} "
        f"(debug: {path})"
    )
    return result.matched_index


async def _validate_keyword(
    keyword: str,
    recording_x: int,
    recording_y: int,
    prompt_instructions: str,
    browser,
    memory: Memory,
    task: Task,
    max_tries: int,
    max_timeout_seconds_per_try: float,
) -> tuple[int, int]:
    last_results: list[OCRResult] = []
    last_screenshot: str | None = None

    for attempt in range(max_tries):
        browser_state = await browser.get_browser_state_summary(
            remove_empty_nodes=task.automation.remove_empty_nodes_in_axtree
        )
        memory.browser_states[-1] = browser_state
        last_screenshot = browser_state.screenshot

        results, _ = _ocr.ocr(browser_state.screenshot)
        last_results = results
        memory.browser_states[-1].ocr_image_sent_to_ocr.append(browser_state.screenshot)
        memory.browser_states[-1].validation_ocr_results = [
            r.model_dump() for r in results
        ]

        exact = [
            r for r in results if r.text.strip().lower() == keyword.strip().lower()
        ]
        if exact:
            best = min(exact, key=lambda r: _distance(r, recording_x, recording_y))
            cx, cy = _ocr_center(best)
            logger.info(
                f"[screenshot_comparison] OCR exact match '{best.text}' at ({cx}, {cy}) "
                f"on attempt {attempt + 1}"
            )
            return cx, cy

        logger.info(
            f"[screenshot_comparison] OCR attempt {attempt + 1}/{max_tries}: "
            f"'{keyword}' not found"
        )
        if attempt < max_tries - 1:
            await asyncio.sleep(max_timeout_seconds_per_try)

    # LLM fallback — candidates sorted by proximity to recording position
    candidates = sorted(
        last_results, key=lambda r: _distance(r, recording_x, recording_y)
    )[:_OCR_CANDIDATE_LIMIT]

    if not candidates:
        raise KeywordNotFoundOnScreenException(
            message=f"No OCR results on screen when looking for '{keyword}'.",
            keyword=keyword,
        )

    matched_idx = await _llm_keyword_fallback(
        keyword, prompt_instructions, candidates, last_screenshot, task, memory
    )

    if matched_idx is None:
        raise KeywordNotFoundOnScreenException(
            message=(
                f"Keyword '{keyword}' not found after {max_tries} OCR attempt(s) "
                "and LLM fallback."
            ),
            keyword=keyword,
        )

    result = candidates[matched_idx]
    cx, cy = _ocr_center(result)
    logger.info(f"[screenshot_comparison] LLM matched '{result.text}' at ({cx}, {cy})")
    return cx, cy


async def _validate_crop(
    recording_screenshot_url: str,
    recording_x: int,
    recording_y: int,
    prompt_instructions: str,
    browser,
    memory: Memory,
    task: Task,
    max_tries: int,
    max_timeout_seconds_per_try: float,
) -> tuple[int, int]:
    recording_b64 = await _fetch_screenshot(recording_screenshot_url)
    recording_crop = _crop_at(recording_b64, recording_x, recording_y)

    for attempt in range(max_tries):
        browser_state = await browser.get_browser_state_summary(
            remove_empty_nodes=task.automation.remove_empty_nodes_in_axtree
        )
        memory.browser_states[-1] = browser_state
        current_crop = _crop_at(browser_state.screenshot, recording_x, recording_y)

        matches, composite_b64, llm_result = await _llm_crop_comparison(
            recording_crop, current_crop, prompt_instructions, task, memory
        )
        is_last_attempt = attempt == max_tries - 1
        if matches or is_last_attempt:
            memory.browser_states[-1].comparison_screenshot = composite_b64
            memory.browser_states[-1].comparison_result = llm_result.model_dump()

        if matches:
            logger.info(
                f"[screenshot_comparison] crop matched on attempt {attempt + 1}, "
                "invoking computer use model for precise coordinates"
            )
            coordinates = await _computer_use_coordinates(
                browser_state.screenshot, prompt_instructions, task, memory
            )
            if coordinates is None:
                raise KeywordNotFoundOnScreenException(
                    message=(
                        f"Screen matched recording but computer use model could not "
                        f"locate the element. Task: '{prompt_instructions}'"
                    ),
                    keyword="element",
                )
            return coordinates

        logger.info(
            f"[screenshot_comparison] crop comparison attempt {attempt + 1}/{max_tries} failed, "
            f"retrying in {max_timeout_seconds_per_try}s..."
        )
        if not is_last_attempt:
            await asyncio.sleep(max_timeout_seconds_per_try)

    raise KeywordNotFoundOnScreenException(
        message=f"Screen validation failed after {max_tries} attempt(s).",
        keyword="element",
    )


async def validate_recording_action(
    action,
    browser,
    memory: Memory,
    task: Task,
    max_tries: int,
    max_timeout_seconds_per_try: float,
) -> tuple[int, int]:
    """
    Validates the current screen matches the recording before performing an action.

    - keyword set  → exact OCR × max_tries, then LLM fallback with OCR candidates
    - keyword=None → LLM crop comparison (recording crop vs current crop) × max_tries

    Returns (x, y) coordinates to act on.
    Raises KeywordNotFoundOnScreenException on failure.
    """
    recording_x = int(action.coordinates[0])
    recording_y = int(action.coordinates[1])

    if action.keyword:
        return await _validate_keyword(
            keyword=action.keyword,
            recording_x=recording_x,
            recording_y=recording_y,
            prompt_instructions=action.prompt_instructions,
            browser=browser,
            memory=memory,
            task=task,
            max_tries=max_tries,
            max_timeout_seconds_per_try=max_timeout_seconds_per_try,
        )
    else:
        return await _validate_crop(
            recording_screenshot_url=action.recording_screenshot,
            recording_x=recording_x,
            recording_y=recording_y,
            prompt_instructions=action.prompt_instructions,
            browser=browser,
            memory=memory,
            task=task,
            max_tries=max_tries,
            max_timeout_seconds_per_try=max_timeout_seconds_per_try,
        )
