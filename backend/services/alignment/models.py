"""Pydantic models for alignment API."""

from pydantic import BaseModel
from typing import List


class TextFragmentModel(BaseModel):
    text: str
    x: float
    y: float
    width: float
    height: float
    font_name: str = "Times-Roman"
    font_size: float = 12.0
    page_num: int = 0


class AlignmentRequest(BaseModel):
    donor_fragments: List[TextFragmentModel]
    sample_fragments: List[TextFragmentModel]
    use_ai: bool = False
    use_classifier: bool = False
    use_semantic: bool = False
    page_dimensions: dict = {"width": 595, "height": 842}


class AlignmentResponse(BaseModel):
    aligned_fragments: List[TextFragmentModel]
    total_pairs: int
    method: str = "sequential"
    confidence_score: float = 1.0