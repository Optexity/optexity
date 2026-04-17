import base64
from typing import Optional

import httpx
import numpy as np
from pydantic import BaseModel

from optexity.inference.core.vision.ocr.ocr import _cv2_to_bytes, _load_cv2
from optexity.inference.models import get_llm_model_with_fallback
from optexity.schema.memory import Memory
from optexity.schema.task import Task

_CROP_ELEMENT_WIDTH = 113
_CROP_ELEMENT_HEIGHT = 41
_CROP_PADDING_FACTOR = 2.0


class ScreenshotMatchResult(BaseModel):
    matches: bool


def crop_screenshot_at_coordinates(
    screenshot_b64: str,
    x: int,
    y: int,
    padding_factor: float = _CROP_PADDING_FACTOR,
) -> str:
    img = _load_cv2(screenshot_b64)
    img_h, img_w = img.shape[:2]

    half_w = _CROP_ELEMENT_WIDTH // 2
    half_h = _CROP_ELEMENT_HEIGHT // 2
    pad_x = int(_CROP_ELEMENT_WIDTH * padding_factor)
    pad_y = int(_CROP_ELEMENT_HEIGHT * padding_factor)

    x1 = max(0, x - half_w - pad_x)
    y1 = max(0, y - half_h - pad_y)
    x2 = min(img_w, x - half_w + _CROP_ELEMENT_WIDTH + pad_x)
    y2 = min(img_h, y - half_h + _CROP_ELEMENT_HEIGHT + pad_y)

    cropped = img[y1:y2, x1:x2]
    return base64.b64encode(_cv2_to_bytes(cropped)).decode("utf-8")


def _build_side_by_side_image(left_b64: str, right_b64: str) -> str:
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


async def _fetch_recording_screenshot(url: str) -> str:
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        return base64.b64encode(response.content).decode("utf-8")


async def compare_screenshots_with_llm(
    recording_crop_b64: str,
    current_crop_b64: str,
    keyword: Optional[str],
    task: "Task",
    memory: "Memory",
) -> bool:
    composite_b64 = _build_side_by_side_image(recording_crop_b64, current_crop_b64)

    if keyword:
        prompt = (
            f"You are shown two cropped screenshots side by side. "
            f"The LEFT image is from a recording; the RIGHT image is the current screen. "
            f"Does the text '{keyword}' visually appear in BOTH images? "
            f"Respond with matches=true only if the same text is clearly visible in both images. "
            f"Respond with matches=false if the text is missing or different in either image."
        )
    else:
        prompt = (
            "You are shown two cropped screenshots side by side. "
            "The LEFT image is from a recording; the RIGHT image is the current screen. "
            "Do both images show the exact same UI element or region? "
            "Respond with matches=true only if both images are showing the same element. "
            "Respond with matches=false if the content differs significantly."
        )

    model = get_llm_model_with_fallback(task.llm_provider, task.llm_model_name, True)
    result, token_usage = model.get_model_response_with_structured_output(
        prompt=prompt,
        response_schema=ScreenshotMatchResult,
        screenshot=composite_b64,
    )
    memory.token_usage += token_usage
    return result.matches
