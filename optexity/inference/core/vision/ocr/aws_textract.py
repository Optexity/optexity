"""
AWS Textract OCR backend.

Calls DetectDocumentText via boto3. The region is fixed to us-west-2.
Credentials are resolved from the standard AWS credential chain
(env vars, ~/.aws/credentials, IAM role, etc.).
"""

from typing import Optional

import boto3
import numpy as np

from optexity.inference.core.vision.ocr.ocr import (
    OCR,
    OCRModels,
    _cv2_to_bytes,
    _load_cv2,
)
from optexity.schema.ocr import BoundingBox, OCRResult
from optexity.utils.timeit import timeit

_AWS_REGION = "us-west-2"
_MIN_CONFIDENCE = 10.0  # drop blocks below this (0–100 scale)


class AWSTextract(OCR):
    def __init__(
        self,
        min_confidence: float = _MIN_CONFIDENCE,
        region: str = _AWS_REGION,
    ):
        """
        Args:
            min_confidence: Drop OCR results below this confidence (0–100).
            region: AWS region for Textract. Defaults to us-west-2.
        """
        super().__init__(OCRModels.AWS_TEXTEXTRACT)
        self._min_conf = min_confidence
        self._client = boto3.client("textract", region_name=region)

    def _ocr(
        self,
        processed: np.ndarray,
        offset_x: int,
        offset_y: int,
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
          1. Decode base64/bytes → np.ndarray.
          2. Crop via parent _crop_array; record (offset_x, offset_y).
          3. Re-encode cropped array to PNG bytes for Textract.
          4. Call DetectDocumentText; iterate LINE then WORD blocks.
          5. Convert Textract's fractional bounding boxes to pixel coordinates.
          6. Translate bboxes back to full-screenshot coordinates.
          7. Filter by min_confidence; return list[OCRResult] (lines first, then words).
        """
        img_h, img_w = processed.shape[:2]
        img_bytes = _cv2_to_bytes(processed)

        response = self._client.detect_document_text(Document={"Bytes": img_bytes})

        lines: list[OCRResult] = []
        words: list[OCRResult] = []
        for block in response.get("Blocks", []):
            block_type = block.get("BlockType")
            if block_type not in ("LINE", "WORD"):
                continue

            conf = block.get("Confidence", 0.0)
            if conf < self._min_conf:
                continue

            text = block.get("Text", "").strip()
            if not text:
                continue

            geo = block["Geometry"]["BoundingBox"]
            # Textract returns fractions (0–1); convert to pixel coordinates.
            x = geo["Left"] * img_w + offset_x
            y = geo["Top"] * img_h + offset_y
            width = geo["Width"] * img_w
            height = geo["Height"] * img_h

            result = OCRResult(
                text=text,
                confidence=conf / 100.0,
                bounding_box=BoundingBox(
                    x=float(x),
                    y=float(y),
                    width=float(width),
                    height=float(height),
                ),
            )
            if block_type == "LINE":
                lines.append(result)
            else:
                words.append(result)

        return lines


if __name__ == "__main__":
    import io
    import time

    from PIL import Image

    buf = io.BytesIO()

    image = Image.open("/tmp/raintree.png")
    image.save(buf, format="PNG")
    screenshot_bytes = buf.getvalue()

    ocr = AWSTextract()

    start_time = time.time()
    results = ocr.ocr(screenshot_bytes)
    end_time = time.time()

    print(f"Time taken in full image: {end_time - start_time} seconds")

    annotated, canvas = ocr.visualize(screenshot_bytes, results)
    with open("/tmp/ocr_annotated.png", "wb") as f:
        f.write(annotated)
    with open("/tmp/ocr_canvas.png", "wb") as f:
        f.write(canvas)
