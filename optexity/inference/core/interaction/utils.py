import asyncio
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Callable

import aiofiles
from rapidfuzz import fuzz

from optexity.exceptions import (
    ElementNotFoundInAxtreeException,
)
from optexity.inference.agents.index_prediction.action_prediction_locator_axtree import (
    ActionPredictionLocatorAxtree,
)
from optexity.inference.core.vision.ocr.aws_textract import AWSTextract
from optexity.inference.core.vision.ocr.tesseract import Tesseract
from optexity.inference.infra.browser import Browser
from optexity.inference.models import GeminiModels, get_llm_model, resolve_model_name
from optexity.schema.memory import Memory
from optexity.schema.ocr import BoundingBox, OCRResult
from optexity.schema.task import Task
from optexity.utils.settings import settings

logger = logging.getLogger(__name__)

_index_prediction_cache: dict[tuple, ActionPredictionLocatorAxtree] = {}
small_ocr = Tesseract()
large_ocr = AWSTextract()


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


def find_keyword_in_results(ocr_results: list[OCRResult], keyword: str):

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


async def resolve_keyword_coordinates(
    keyword: str,
    x: int,
    y: int,
    memory: Memory,
) -> OCRResult:
    """Find keyword on screen and return the nearest match to (x, y).

    Raises KeywordNotFoundOnScreenException if not found.
    """
    from optexity.exceptions import KeywordNotFoundOnScreenException

    matches = await match_all_text_in_screenshot(
        memory,
        keyword,
        BoundingBox(x=x, y=y, width=113, height=41),
    )
    if not matches:
        raise KeywordNotFoundOnScreenException(
            message=f"Keyword '{keyword}' not found on screen.",
            keyword=keyword,
        )

    if len(matches) == 1:
        return matches[0]

    result = min(
        matches,
        key=lambda m: (
            (m.bounding_box.x + m.bounding_box.width / 2 - x) ** 2
            + (m.bounding_box.y + m.bounding_box.height / 2 - y) ** 2
        ),
    )
    logger.info(
        f"Multiple matches ({len(matches)}) for keyword '{keyword}', picked nearest: '{result.text}' at ({result.bounding_box.x}, {result.bounding_box.y})"
    )
    return result
