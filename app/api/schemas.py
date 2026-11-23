from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BoundingBox(BaseModel):
    """Spatial bounding box representation [x0, y0, x1, y1] in points."""

    model_config = ConfigDict(frozen=True)

    x0: float
    y0: float
    x1: float
    y1: float

    @model_validator(mode="after")
    def validate_and_order_coordinates(self) -> BoundingBox:
        min_x = min(self.x0, self.x1)
        max_x = max(self.x0, self.x1)
        min_y = min(self.y0, self.y1)
        max_y = max(self.y0, self.y1)
        if (min_x, min_y, max_x, max_y) != (self.x0, self.y0, self.x1, self.y1):
            object.__setattr__(self, "x0", float(min_x))
            object.__setattr__(self, "y0", float(min_y))
            object.__setattr__(self, "x1", float(max_x))
            object.__setattr__(self, "y1", float(max_y))
        return self

    @property
    def area(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)

    def union(self, other: BoundingBox) -> BoundingBox:
        """Returns the minimal enclosing bounding box covering both boxes."""
        return BoundingBox(
            x0=min(self.x0, other.x0),
            y0=min(self.y0, other.y0),
            x1=max(self.x1, other.x1),
            y1=max(self.y1, other.y1),
        )

    def intersection_area(self, other: BoundingBox) -> float:
        """Returns the area of intersection between two bounding boxes."""
        ix0 = max(self.x0, other.x0)
        iy0 = max(self.y0, other.y0)
        ix1 = min(self.x1, other.x1)
        iy1 = min(self.y1, other.y1)
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0
        return (ix1 - ix0) * (iy1 - iy0)

    def iou(self, other: BoundingBox) -> float:
        """Computes Intersection over Union (IoU) with another bounding box."""
        inter = self.intersection_area(other)
        if inter <= 0.0:
            return 0.0
        union_area = self.area + other.area - inter
        if union_area <= 0.0:
            return 0.0
        return inter / union_area


class DocumentChunk(BaseModel):
    """Atomic multimodal document chunk with spatial and semantic metadata."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    page_num: int
    chunk_type: Literal["text", "table", "ocr_text"]
    content: str
    bbox: BoundingBox
    token_estimate: int


class Citation(BaseModel):
    """Exact source citation grounding an AI claim back to document spatial coordinates."""

    model_config = ConfigDict(frozen=True)

    source_index: int
    chunk_id: str
    page_num: int
    chunk_type: str
    bbox: BoundingBox
    relevance_score: float
    snippet: str


class QueryRequest(BaseModel):
    """Request payload for semantic querying across an ingested document."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=15)


class QueryResponse(BaseModel):
    """Citation-grounded synthesis response with execution profiling metadata."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    answer: str
    citations: list[Citation]
    retrieval_ms: float
    generation_ms: float


class HealthResponse(BaseModel):
    """Response schema for the system health check endpoint."""

    model_config = ConfigDict(frozen=True)

    status: str
    version: str
    environment: str
