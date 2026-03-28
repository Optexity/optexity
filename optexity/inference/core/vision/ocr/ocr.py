import base64
from enum import Enum, unique
from typing import Optional

import cv2
import numpy as np

from optexity.schema.ocr import BoundingBox, OCRResult


@unique
class OCRModels(Enum):
    AWS_TEXTEXTRACT = "aws_textract"
    TESSERACT = "tesseract"
    RAPID_OCR = "rapid_ocr"
    PADDLE_OCR = "paddle_ocr"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_cv2(screenshot: str | bytes) -> np.ndarray:
    """
    Decode a screenshot into a BGR numpy array.
      str   → base64-encoded PNG/JPEG string  (decoded then imdecode'd)
      bytes → PNG/JPEG-encoded bytes           (imdecode'd directly)

    NOTE: raw pixel bytes (e.g. PIL.Image.tobytes()) are NOT valid here.
    The input must be a compressed image format that cv2.imdecode understands.
    Use cv2.imencode / PIL.Image.save(buf, format="PNG") to get valid bytes.
    """
    raw = base64.b64decode(screenshot) if isinstance(screenshot, str) else screenshot
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(
            "cv2.imdecode returned None — the input is not valid PNG/JPEG bytes. "
            "If you have a PIL Image, use io.BytesIO + image.save(buf, format='PNG') "
            "and pass buf.getvalue() instead of image.tobytes()."
        )
    return img


def _cv2_to_bytes(img: np.ndarray) -> bytes:
    """Encode a cv2 image to PNG bytes."""
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class OCR:
    def __init__(self, model: OCRModels):
        self.model = model

    def preprocess_screenshot(
        self,
        screenshot: str | bytes,
        convert_to_grayscale: bool = True,
        resize_to_width: Optional[int] = None,
        resize_to_height: Optional[int] = None,
    ) -> bytes:
        """
        Load, optionally grayscale-convert, and optionally resize the screenshot.
        Returns PNG-encoded bytes of the processed image.

        GPU note: if OpenCV is built with CUDA support, replace cv2.resize /
        cv2.cvtColor calls with their cv2.cuda equivalents for GPU-accelerated
        preprocessing (upload with cv2.cuda_GpuMat, download with .download()).
        """
        img = _load_cv2(screenshot)

        if convert_to_grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if resize_to_width is not None or resize_to_height is not None:
            h, w = img.shape[:2]
            if resize_to_width and resize_to_height:
                new_w, new_h = resize_to_width, resize_to_height
            elif resize_to_width:
                new_w = resize_to_width
                new_h = int(h * resize_to_width / w)
            else:
                new_h = resize_to_height
                new_w = int(w * resize_to_height / h)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        return _cv2_to_bytes(img)

    def _crop_array(
        self,
        img: np.ndarray,
        roi: BoundingBox,
        padding_factor: float = 1.0,
    ) -> tuple[np.ndarray, int, int]:
        """
        Crop a BGR/grayscale numpy array to `roi` with padding and return the
        cropped array together with the top-left offset used for the crop.

        Padding is `padding_factor × roi.width` left/right and
        `padding_factor × roi.height` top/bottom, clamped to image bounds.

        Returns:
            (cropped_img, offset_x, offset_y) — offset is the (x1, y1) corner
            of the crop in the original image coordinate space, needed to
            translate detected bounding boxes back.
        """
        img_h, img_w = img.shape[:2]
        pad_x = roi.width * padding_factor
        pad_y = roi.height * padding_factor
        x1 = max(0, int(roi.x - pad_x))
        y1 = max(0, int(roi.y - pad_y))
        x2 = min(img_w, int(roi.x + roi.width + pad_x))
        y2 = min(img_h, int(roi.y + roi.height + pad_y))
        return img[y1:y2, x1:x2], x1, y1

    # ------------------------------------------------------------------
    # Preprocessing — operates on np.ndarray; called after the entry point
    # has decoded str|bytes. GPU path requires OpenCV built with CUDA.
    # ------------------------------------------------------------------

    @staticmethod
    def _has_cuda() -> bool:
        try:
            return cv2.cuda.getCudaEnabledDeviceCount() > 0
        except AttributeError:
            return False

    @staticmethod
    def _preprocess_cpu(gray: np.ndarray) -> np.ndarray:
        """
        Fast binarisation for digital UI screenshots.

        fastNlMeansDenoising is avoided: it is O(N²) patch-matching and
        10-100× slower than GaussianBlur with no benefit on crisp screenshots.

          1. 3×3 Gaussian blur  — removes JPEG/PNG compression artefacts.
          2. Otsu threshold     — globally optimal binarisation for bimodal
                                  pixel distributions (text on uniform bg).
        """
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    @staticmethod
    def _preprocess_gpu(gray: np.ndarray) -> np.ndarray:
        """
        GPU-accelerated blur via OpenCV CUDA, then Otsu on CPU.

        Otsu has no CUDA equivalent but is negligibly fast on CPU.
        Requires OpenCV built with -DWITH_CUDA=ON (symbols are runtime-only,
        absent from type stubs, hence accessed via getattr).
        """
        h, w = gray.shape
        GpuMat = getattr(cv2, "cuda_GpuMat")
        gpu_gray = GpuMat(h, w, cv2.CV_8UC1)
        gpu_gray.upload(gray)
        gauss = getattr(cv2.cuda, "createGaussianFilter")(
            cv2.CV_8UC1, cv2.CV_8UC1, (3, 3), 0
        )
        blurred = gauss.apply(gpu_gray).download()
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def _preprocess_array(self, img: np.ndarray, use_gpu: bool = False) -> np.ndarray:
        """
        Convert a BGR image to a binarised grayscale array ready for OCR.

        Args:
            img: BGR numpy array (as returned by _load_cv2 / _crop_array).
            use_gpu: attempt GPU-accelerated Gaussian blur if CUDA is available.

        Returns:
            Single-channel binary uint8 array.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if use_gpu and self._has_cuda():
            return self._preprocess_gpu(gray)
        return self._preprocess_cpu(gray)

    def ocr(
        self, screenshot: str | bytes, region_of_interest: Optional[BoundingBox] = None
    ) -> list[OCRResult]:
        raise NotImplementedError("This method is not implemented")
