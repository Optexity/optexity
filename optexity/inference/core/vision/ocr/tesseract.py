"""
CPU-optimized Tesseract OCR with optional GPU-accelerated preprocessing.

Tesseract itself runs on CPU only. GPU acceleration applies to the image
preprocessing pipeline when OpenCV is built with CUDA support.
"""

import shutil
from typing import Optional

import pytesseract
from pytesseract import Output

from optexity.inference.core.vision.ocr.ocr import OCR, OCRModels, _load_cv2
from optexity.schema.ocr import BoundingBox, OCRResult
from optexity.utils.timeit import timeit

if shutil.which("tesseract") is None:
    raise RuntimeError(
        "Tesseract is not installed.\n"
        "Mac: brew install tesseract\n"
        "Linux: sudo apt install tesseract-ocr\n"
    )
# ---------------------------------------------------------------------------
# Tesseract config
# --oem 1  : LSTM engine only (fastest, most accurate for Tesseract 4/5)
# --psm 11 : Sparse text — finds as much text as possible in no particular order.
#            Switch to --psm 6 for dense, uniform text blocks.
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG = "--oem 1 --psm 11"
_MIN_CONFIDENCE = 10  # drop boxes below this (0–100 scale)


class Tesseract(OCR):
    def __init__(
        self,
        tesseract_cmd: Optional[str] = None,
        config: str = _DEFAULT_CONFIG,
        min_confidence: int = _MIN_CONFIDENCE,
        use_gpu_preprocessing: bool = False,
    ):
        """
        Args:
            tesseract_cmd: Path to the tesseract binary. If None, uses PATH default.
            config: Tesseract config string. Defaults to LSTM-only + sparse-text mode.
            min_confidence: Drop OCR results below this confidence (0–100).
            use_gpu_preprocessing: Use CUDA-accelerated preprocessing if available.
        """
        super().__init__(OCRModels.TESSERACT)
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self._config = config
        self._min_conf = min_confidence
        self._use_gpu = use_gpu_preprocessing

    @timeit
    def ocr(
        self,
        screenshot: str | bytes,
        region_of_interest: Optional[BoundingBox] = None,
        padding_factor: float = 1.0,
    ) -> list[OCRResult]:
        """
        Run OCR on the screenshot (or a padded sub-region) and return all results
        in the coordinate space of the original full screenshot.

        Args:
            screenshot: base64-encoded image string, or raw encoded image bytes.
            region_of_interest: optional ROI; image is cropped with
                                `padding_factor` × ROI-dimension padding on each side.
            padding_factor: multiplier for padding around the ROI (default 1.0).

        Pipeline:
          1. Decode base64/bytes → np.ndarray (only str|bytes handling here).
          2. Crop via parent _crop_array; record (offset_x, offset_y).
          3. Preprocess via parent _preprocess_array (grayscale + blur + Otsu).
          4. Tesseract LSTM (--oem 1), sparse-text (--psm 11).
          5. Translate bboxes back to full-screenshot coordinates.
          6. Filter by min_confidence; return list[OCRResult].
        """
        # Entry point: str|bytes → np.ndarray. All subsequent ops are np.ndarray.
        img_bgr = _load_cv2(screenshot)

        offset_x, offset_y = 0, 0
        if region_of_interest is not None:
            img_bgr, offset_x, offset_y = self._crop_array(
                img_bgr, region_of_interest, padding_factor
            )

        processed = self._preprocess_array(img_bgr, use_gpu=self._use_gpu)

        data = pytesseract.image_to_data(
            processed, config=self._config, output_type=Output.DICT
        )

        results: list[OCRResult] = []
        for i in range(len(data["text"])):
            text = data["text"][i].strip()
            conf = int(data["conf"][i])
            if not text or conf < self._min_conf:
                continue

            results.append(
                OCRResult(
                    text=text,
                    confidence=conf / 100.0,
                    bounding_box=BoundingBox(
                        x=float(data["left"][i] + offset_x),
                        y=float(data["top"][i] + offset_y),
                        width=float(data["width"][i]),
                        height=float(data["height"][i]),
                    ),
                )
            )

        return results


if __name__ == "__main__":
    import io
    import time

    from PIL import Image

    buf = io.BytesIO()

    image = Image.open("/tmp/raintree.png")
    image.save(buf, format="PNG")
    screenshot_bytes = buf.getvalue()

    ocr = Tesseract()

    start_time = time.time()
    results = ocr.ocr(screenshot_bytes)
    end_time = time.time()

    # print(results)
    print(f"Time taken in full image: {end_time - start_time} seconds")

    bounding_box = BoundingBox(x=116.0, y=898.0, width=113.0, height=41.0)

    start_time = time.time()
    results = ocr.ocr(screenshot_bytes, region_of_interest=bounding_box)
    end_time = time.time()
    print(results)

    print(f"Time taken in ROI: {end_time - start_time} seconds")
