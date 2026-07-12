"""
Text alignment endpoint.
"""

from fastapi import APIRouter
from backend.services.alignment import (
    TextAligner, TextFragment, TextFragmentModel,
    AlignmentRequest, AlignmentResponse
)
from backend.logger import logger

router = APIRouter()


@router.post("/align", response_model=AlignmentResponse)
async def align_text(request: AlignmentRequest):
    """Align text fragments from donor and sample PDFs."""
    logger.info(f"Aligning {len(request.donor_fragments)} donor vs {len(request.sample_fragments)} sample fragments")
    
    donor_fragments = [
        TextFragment(
            text=f.text, x=f.x, y=f.y,
            width=f.width, height=f.height,
            font_name=f.font_name, font_size=f.font_size
        )
        for f in request.donor_fragments
    ]
    
    sample_fragments = [
        TextFragment(
            text=f.text, x=f.x, y=f.y,
            width=f.width, height=f.height,
            font_name=f.font_name, font_size=f.font_size
        )
        for f in request.sample_fragments
    ]
    
    aligner = TextAligner(use_ai=True)
    aligned_pairs = aligner.align(donor_fragments, sample_fragments)
    
    aligned_fragments = []
    for _, sample_frag in aligned_pairs:
        aligned_fragments.append(
            TextFragmentModel(
                text=sample_frag.text,
                x=sample_frag.x, y=sample_frag.y,
                width=sample_frag.width, height=sample_frag.height,
                font_name=sample_frag.font_name, font_size=sample_frag.font_size
            )
        )
    
    logger.info(f"Aligned {len(aligned_fragments)} pairs")
    
    return AlignmentResponse(
        aligned_fragments=aligned_fragments,
        total_pairs=len(aligned_fragments),
        method="ai"
    )