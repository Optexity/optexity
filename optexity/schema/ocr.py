from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class OCRResult(BaseModel):
    text: str
    confidence: float
    bounding_box: BoundingBox
    # IDs of original OCRResult objects that this joined candidate was built from.
    # Empty for raw (non-joined) results.
    source_ids: list[int] = Field(default_factory=list)
