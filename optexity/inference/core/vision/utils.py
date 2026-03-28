import base64

import cv2
import numpy as np

_MARK_RADIUS_PX = 10


async def mark_screenshot(screenshot: str | bytes, x: int, y: int) -> str | bytes:
    """Mark a screenshot with a small circle at (x, y).
    Returns same type as input (base64 str -> base64 str, bytes -> bytes).
    """
    is_base64 = isinstance(screenshot, str)
    screenshot_bytes = base64.b64decode(screenshot) if is_base64 else screenshot

    img = cv2.imdecode(np.frombuffer(screenshot_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode screenshot as an image")

    cv2.circle(
        img,
        (x, y),
        _MARK_RADIUS_PX,
        (0, 0, 255),
        thickness=2,
        lineType=cv2.LINE_AA,
    )

    ok, encoded = cv2.imencode(
        ".jpg",
        img,
        [int(cv2.IMWRITE_JPEG_QUALITY), 90],
    )
    if not ok:
        raise ValueError("Failed to encode marked screenshot")

    encoded_bytes = encoded.tobytes()
    if is_base64:
        return base64.b64encode(encoded_bytes).decode("ascii")
    return encoded_bytes
