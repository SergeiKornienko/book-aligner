"""
Upload and job management endpoints.
"""

import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.services.pdf_parser import PDFParser
from backend.api.dependencies import get_upload_dir
from backend.logger import logger

router = APIRouter()

# In-memory storage (replace with database later)
jobs = {}


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF file for processing."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    job_id = str(uuid.uuid4())
    upload_dir = get_upload_dir()
    file_path = upload_dir / f"{job_id}.pdf"
    
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    
    jobs[job_id] = {
        "filename": file.filename,
        "file_path": str(file_path),
        "status": "uploaded"
    }
    
    logger.info(f"File uploaded: {file.filename} -> job {job_id}")
    
    return {
        "job_id": job_id,
        "filename": file.filename,
        "status": "uploaded"
    }


@router.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a processing job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@router.get("/job/{job_id}/text")
async def extract_text(job_id: str):
    """Extract text with coordinates from uploaded PDF."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    file_path = job["file_path"]
    
    if not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    logger.debug(f"Extracting text from {file_path}")
    
    parser = PDFParser(file_path)
    text_data = parser.extract_text_with_coordinates()
    parser.close()
    
    return {
        "job_id": job_id,
        "pages": text_data
    }


@router.get("/job/{job_id}/images")
async def extract_images(job_id: str):
    """Extract images from uploaded PDF."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    file_path = job["file_path"]
    
    if not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    logger.debug(f"Extracting images from {file_path}")
    
    parser = PDFParser(file_path)
    images = parser.extract_images()
    parser.close()
    
    return {
        "job_id": job_id,
        "images": images
    }