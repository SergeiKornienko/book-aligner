"""
API routes for BookAligner.
Handles PDF upload, processing, and text alignment.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import os
import uuid
from pathlib import Path

from backend.services.models import TextFragment
from backend.services.pdf_parser import PDFParser
from backend.services.text_alignment import (
    TextAligner,
    TextFragment,
    TextFragmentModel,
    AlignmentRequest,
    AlignmentResponse
)
from backend.services.paragraph_builder import ParagraphBuilder, Paragraph

router = APIRouter()

# In-memory storage (replace with database later)
jobs = {}


def get_upload_dir():
    """Get the upload directory path"""
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file for processing.
    Returns a job_id for tracking.
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Save file
    upload_dir = get_upload_dir()
    file_path = upload_dir / f"{job_id}.pdf"
    
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    
    # Store job info
    jobs[job_id] = {
        "filename": file.filename,
        "file_path": str(file_path),
        "status": "uploaded"
    }
    
    return {
        "job_id": job_id,
        "filename": file.filename,
        "status": "uploaded"
    }


@router.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a processing job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return jobs[job_id]


@router.get("/job/{job_id}/text")
async def extract_text(job_id: str):
    """Extract text with coordinates from uploaded PDF"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    file_path = job["file_path"]
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    parser = PDFParser(file_path)
    text_data = parser.extract_text_with_coordinates()
    parser.close()
    
    return {
        "job_id": job_id,
        "pages": text_data
    }


@router.get("/job/{job_id}/images")
async def extract_images(job_id: str):
    """Extract images from uploaded PDF"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    file_path = job["file_path"]
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    parser = PDFParser(file_path)
    images = parser.extract_images()
    parser.close()
    
    return {
        "job_id": job_id,
        "images": images
    }


@router.post("/align", response_model=AlignmentResponse)
async def align_text(request: AlignmentRequest):
    """
    Align text fragments from donor and sample PDFs.
    
    Takes text fragments from both PDFs and returns aligned pairs
    where sample text is placed at donor coordinates.
    """
    # Convert Pydantic models to TextFragment objects
    donor_fragments = [
        TextFragment(
            text=f.text,
            x=f.x,
            y=f.y,
            width=f.width,
            height=f.height,
            font_name=f.font_name,
            font_size=f.font_size
        )
        for f in request.donor_fragments
    ]
    
    sample_fragments = [
        TextFragment(
            text=f.text,
            x=f.x,
            y=f.y,
            width=f.width,
            height=f.height,
            font_name=f.font_name,
            font_size=f.font_size
        )
        for f in request.sample_fragments
    ]
    
    # Perform alignment
    aligner = TextAligner(use_ai=True)  # Using AI mode by default
    aligned_pairs = aligner.align(donor_fragments, sample_fragments)
    
    # Convert back to response model
    aligned_fragments = []
    for _, sample_frag in aligned_pairs:
        aligned_fragments.append(
            TextFragmentModel(
                text=sample_frag.text,
                x=sample_frag.x,
                y=sample_frag.y,
                width=sample_frag.width,
                height=sample_frag.height,
                font_name=sample_frag.font_name,
                font_size=sample_frag.font_size
            )
        )
    
    return AlignmentResponse(
        aligned_fragments=aligned_fragments,
        total_pairs=len(aligned_fragments),
        method="ai"  # Updated to show AI mode
    )


# =============================================================================
# PARAGRAPH ENDPOINTS
# =============================================================================

@router.post("/paragraphs/build")
async def build_paragraphs(request: dict):
    """
    Build paragraphs from text fragments.
    
    Groups individual text fragments into semantic paragraphs
    based on layout analysis and block classification.
    """
    fragments_data = request.get("fragments", [])
    page_dimensions = request.get("page_dimensions", {"width": 595, "height": 842})
    
    # Convert dicts to TextFragment objects
    fragments = []
    for f_data in fragments_data:
        fragment = TextFragment(
            text=f_data["text"],
            x=f_data.get("x", 0),
            y=f_data.get("y", 0),
            width=f_data.get("width", 100),
            height=f_data.get("height", 12),
            font_name=f_data.get("font_name", "Times-Roman"),
            font_size=f_data.get("font_size", 12),
            page_num=f_data.get("page_num", 0)
        )
        fragments.append(fragment)
    
    # Build paragraphs
    builder = ParagraphBuilder()
    paragraphs = builder.build_paragraphs(fragments, page_dimensions)
    
    # Convert to response format
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
    
    return {
        "paragraphs": paragraphs_data,
        "total_paragraphs": len(paragraphs),
        "page_dimensions": page_dimensions
    }


@router.post("/paragraphs/align")
async def align_paragraphs(request: dict):
    """
    Align paragraphs from donor and sample PDFs.
    
    Matches paragraphs by type and content, preserving donor structure.
    """
    donor_paragraphs = request.get("donor_paragraphs", [])
    sample_paragraphs = request.get("sample_paragraphs", [])
    use_ai = request.get("use_ai", False)
    
    # Group by block type for separate alignment
    from backend.services.text_alignment import TextAligner
    
    aligner = TextAligner(use_ai=use_ai)
    
    aligned_pairs = []
    
    # Align by block type groups
    for block_type_name in ["TITLE", "SUBTITLE", "BODY", "CAPTION"]:
        donor_of_type = [p for p in donor_paragraphs if p.get("block_type") == block_type_name]
        sample_of_type = [p for p in sample_paragraphs if p.get("block_type") == block_type_name]
        
        if not donor_of_type or not sample_of_type:
            continue
        
        # Convert to fragments for alignment
        donor_frags = [
            TextFragment(
                text=p["text"],
                x=0, y=0,
                width=450, height=12,
                page_num=p.get("start_page", 0)
            )
            for p in donor_of_type
        ]
        
        sample_frags = [
            TextFragment(
                text=p["text"],
                x=0, y=0,
                width=450, height=12,
                page_num=p.get("start_page", 0)
            )
            for p in sample_of_type
        ]
        
        # Align
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