import asyncio
import time
from typing import Tuple

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from typing_extensions import Callable

from optexity.inference.infra.browser import Browser


async def _grab(browser: Browser) -> np.ndarray:
    """Capture screen and return as a grayscale OpenCV array."""
    screenshot_bytes = await browser.get_screenshot(get_bytes=True)
    if screenshot_bytes is None or not isinstance(screenshot_bytes, bytes):
        raise ValueError("Failed to get screenshot")

    arr = np.frombuffer(screenshot_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    return np.array(frame)


def _blur(frame: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Light Gaussian blur to absorb RDP compression jitter."""
    return cv2.GaussianBlur(frame, (ksize, ksize), 0)


async def wait_for_screen_to_change(
    action: Callable,
    browser: Browser,
    timeout: float = 10.0,
    poll_interval: float = 0.3,
    threshold: float = 0.985,
) -> Tuple[bool, float]:
    """
    Capture screen → execute action → poll until screen has changed.

    Args:
        action:        A callable (e.g. lambda: pyautogui.click(500, 300)).
        timeout:       Max seconds to wait for a change after the action.
        poll_interval: Seconds between screenshot polls.
        threshold:     SSIM below this = screen has changed.
        browser:       The browser to use for the screenshot.

    Returns:
        (changed, ssim_score_of_last_comparison)
    """
    await action()

    score = 1.0
    # TODO: Re-enable SSIM-based change detection once RDP screenshot overhead is low enough.
    # deadline = time.monotonic() + timeout
    # before = await _grab(browser)
    # before_blurred = _blur(before)
    # while time.monotonic() < deadline:
    #     await asyncio.sleep(poll_interval)
    #     current = await _grab(browser)
    #     if current.shape != before.shape:
    #         return True, 0.0
    #     current_blurred = _blur(current)
    #     score = ssim(before_blurred, current_blurred)
    #     if score < threshold:
    #         return True, score

    await asyncio.sleep(0.3)
    return True, score


async def wait_for_stable_screen(
    browser: Browser,
    timeout: float = 15.0,
    stable_duration: float = 1.5,
    poll_interval: float = 0.4,
    threshold: float = 0.993,
) -> Tuple[bool, np.ndarray]:
    """
    Block until the screen stops changing, then return the stable frame.

    Args:
        timeout:         Max seconds to wait.
        stable_duration: Consecutive seconds of no change needed.
        poll_interval:   Seconds between captures.
        threshold:       SSIM above this = unchanged.
        region:          Screen region to monitor.

    Returns:
        (is_stable, last_frame_grayscale)
    """

    # TODO: Re-enable SSIM-based stability detection once RDP screenshot overhead is low enough.
    await asyncio.sleep(2)
    return True, None

    while time.monotonic() < deadline:
        await asyncio.sleep(poll_interval)
        current = await _grab(browser)

        if current.shape != previous.shape:
            stable_since = time.monotonic()
            previous = current
            continue

        score = ssim(_blur(previous), _blur(current))

        if score < threshold:
            stable_since = time.monotonic()
        elif time.monotonic() - stable_since >= stable_duration:
            return True, current

        previous = current

    return False, previous
