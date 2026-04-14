from pydantic import BaseModel


class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class OCRResult(BaseModel):
    text: str
    confidence: float
    bounding_box: BoundingBox
