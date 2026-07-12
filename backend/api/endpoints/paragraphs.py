"""
Paragraph building and alignment endpoints.
"""

from fastapi import APIRouter
from backend.services.paragraph_builder import ParagraphBuilder
from backend.services.alignment import TextAligner, TextFragment
from backend.services.models import TextFragment
from backend.logger import logger

router = APIRouter()


@router.post("/paragraphs/build")
async def build_paragraphs(request: dict):
    """Build paragraphs from text fragments."""
    fragments_data = request.get("fragments", [])
    page_dimensions = request.get("page_dimensions", {"width": 595, "height": 842})
    
    logger.debug(f"Building paragraphs from {len(fragments_data)} fragments")
    
    fragments = []
    for f_data in fragments_data:
        fragment = TextFragment(
            text=f_data["text"],
            x=f_data.get("x", 0), y=f_data.get("y", 0),
            width=f_data.get("width", 100), height=f_data.get("height", 12),
            font_name=f_data.get("font_name", "Times-Roman"),
            font_size=f_data.get("font_size", 12),
            page_num=f_data.get("page_num", 0)
        )
        fragments.append(fragment)
    
    builder = ParagraphBuilder()
    paragraphs = builder.build_paragraphs(fragments, page_dimensions)
    
    paragraphs_data = []
    for para in paragraphs:
        paragraphs_data.append({
            "text": para.text,
            "block_type": para.block_type.name,
            "fragment_count": para.fragment_count,
            "start_page": para.start_page,
            "end_page": para.end_page,
            "spans_pages": para.spans_pages,
            "avg_font_size": para.avg_font_size
        })
    
    logger.info(f"Built {len(paragraphs)} paragraphs")
    
    return {
        "paragraphs": paragraphs_data,
        "total_paragraphs": len(paragraphs),
        "page_dimensions": page_dimensions
    }


@router.post("/paragraphs/align")
async def align_paragraphs(request: dict):
    """Align paragraphs from donor and sample PDFs."""
    donor_paragraphs = request.get("donor_paragraphs", [])
    sample_paragraphs = request.get("sample_paragraphs", [])
    use_ai = request.get("use_ai", False)
    
    logger.info(f"Aligning {len(donor_paragraphs)} vs {len(sample_paragraphs)} paragraphs")
    
    aligner = TextAligner(use_ai=use_ai)
    
    aligned_pairs = []
    
    for block_type_name in ["TITLE", "SUBTITLE", "BODY", "CAPTION"]:
        donor_of_type = [p for p in donor_paragraphs if p.get("block_type") == block_type_name]
        sample_of_type = [p for p in sample_paragraphs if p.get("block_type") == block_type_name]
        
        if not donor_of_type or not sample_of_type:
            continue
        
        donor_frags = [
            TextFragment(text=p["text"], x=0, y=0, width=450, height=12, page_num=p.get("start_page", 0))
            for p in donor_of_type
        ]
        
        sample_frags = [
            TextFragment(text=p["text"], x=0, y=0, width=450, height=12, page_num=p.get("start_page", 0))
            for p in sample_of_type
        ]
        
        aligned = aligner.align(donor_frags, sample_frags)
        
        for donor_frag, sample_frag in aligned:
            aligned_pairs.append({
                "donor_text": donor_frag.text,
                "sample_text": sample_frag.text,
                "block_type": block_type_name,
                "donor_page": donor_frag.page_num,
                "sample_page": sample_frag.page_num
            })
    
    return {
        "aligned_paragraphs": aligned_pairs,
        "total_pairs": len(aligned_pairs),
        "method": "ai" if use_ai else "sequential"
    }