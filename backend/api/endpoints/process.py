"""
End-to-end processing endpoint with progress tracking.
"""

import uuid
import io
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import fitz
from PIL import Image

from backend.services.pdf_parser import PDFParser
from backend.services.alignment import TextAligner
from backend.services.models import TextFragment
from backend.services.pdf_generator import PDFGenerator
from backend.services.progress_tracker import progress_tracker
from backend.api.dependencies import get_upload_dir
from backend.logger import logger

router = APIRouter()


@router.post("/process")
async def process_pdfs(
    donor: UploadFile = File(...),
    sample: UploadFile = File(...),
    use_semantic: bool = False,
    use_ocr: bool = False
):
    """
    Full end-to-end processing pipeline with progress tracking.
    
    1. Accepts two PDFs: donor (illustrated) and sample (translation)
    2. Parses both PDFs (with optional OCR)
    3. Aligns text (with optional semantic matching)
    4. Generates final PDF with translated text
    5. Returns the result
    
    Progress is tracked via /job/{task_id}/progress SSE endpoint.
    """
    if not donor.filename.endswith('.pdf') or not sample.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Both files must be PDF")
    
    logger.info(f"Processing: donor={donor.filename}, sample={sample.filename}, "
                f"semantic={use_semantic}, ocr={use_ocr}")
    
    # Create progress task
    task_id = progress_tracker.create_task(
        name=f"Processing {donor.filename}",
        total=100
    )
    
    upload_dir = get_upload_dir()
    donor_path = upload_dir / f"donor_{uuid.uuid4()}.pdf"
    sample_path = upload_dir / f"sample_{uuid.uuid4()}.pdf"
    output_path = upload_dir / f"output_{uuid.uuid4()}.pdf"
    
    donor_content = await donor.read()
    sample_content = await sample.read()
    
    with open(donor_path, "wb") as f:
        f.write(donor_content)
    with open(sample_path, "wb") as f:
        f.write(sample_content)
    
    try:
        # ── Step 1: Parse PDFs (0-30%) ──
        progress_tracker.update(task_id, progress=5, message="Reading PDFs...")
        
        if use_ocr:
            from backend.services.ocr_processor import OCRProcessor
            
            logger.info("Using OCR mode")
            ocr = OCRProcessor()
            
            # OCR donor
            progress_tracker.update(task_id, progress=10, message="OCR processing donor PDF...")
            donor_doc = fitz.open(str(donor_path))
            donor_fragments = []
            total_donor_pages = donor_doc.page_count
            for page_num in range(total_donor_pages):
                page_progress = 10 + int((page_num / total_donor_pages) * 10)
                progress_tracker.update(
                    task_id, progress=page_progress,
                    message=f"OCR donor page {page_num + 1}/{total_donor_pages}"
                )
                pix = donor_doc[page_num].get_pixmap(dpi=200)
                img = Image.open(io.BytesIO(pix.tobytes('png')))
                regions = ocr.extract_text_regions(img, page_num=page_num, lang='eng')
                for region in regions:
                    donor_fragments.append({
                        "page": page_num + 1, "x": region.x, "y": region.y,
                        "width": region.width, "height": region.height,
                        "text": region.text, "font_name": "Times-Roman", "font_size": 11.0
                    })
            donor_doc.close()
            
            # OCR sample
            progress_tracker.update(task_id, progress=20, message="OCR processing sample PDF...")
            sample_doc = fitz.open(str(sample_path))
            sample_fragments = []
            total_sample_pages = sample_doc.page_count
            for page_num in range(total_sample_pages):
                page_progress = 20 + int((page_num / total_sample_pages) * 10)
                progress_tracker.update(
                    task_id, progress=page_progress,
                    message=f"OCR sample page {page_num + 1}/{total_sample_pages}"
                )
                pix = sample_doc[page_num].get_pixmap(dpi=200)
                img = Image.open(io.BytesIO(pix.tobytes('png')))
                regions = ocr.extract_text_regions(img, page_num=page_num, lang='rus')
                for region in regions:
                    sample_fragments.append({
                        "page": page_num + 1, "x": region.x, "y": region.y,
                        "width": region.width, "height": region.height,
                        "text": region.text, "font_name": "Times-Roman", "font_size": 11.0
                    })
            sample_doc.close()
        else:
            progress_tracker.update(task_id, progress=15, message="Parsing PDFs...")
            donor_parser = PDFParser(str(donor_path))
            sample_parser = PDFParser(str(sample_path))
            donor_fragments = donor_parser.extract_text_with_coordinates()
            sample_fragments = sample_parser.extract_text_with_coordinates()
            donor_parser.close()
            sample_parser.close()
        
        progress_tracker.update(
            task_id, progress=30,
            message=f"Extracted {len(donor_fragments)} donor and {len(sample_fragments)} sample fragments"
        )
        logger.info(f"Extracted {len(donor_fragments)} donor and {len(sample_fragments)} sample fragments")
        
        # ── Step 2: Convert to TextFragment (30-35%) ──
        progress_tracker.update(task_id, progress=32, message="Preparing fragments...")
        
        donor_frags = []
        for item in donor_fragments:
            donor_frags.append(TextFragment(
                text=item.get("text", ""), x=item.get("x", 0), y=item.get("y", 0),
                width=item.get("width", 50), height=item.get("height", 10),
                font_name=item.get("font_name", "Times-Roman"),
                font_size=item.get("font_size", 11),
                page_num=item.get("page", 1) - 1
            ))
        
        sample_frags = []
        for item in sample_fragments:
            sample_frags.append(TextFragment(
                text=item.get("text", ""), x=item.get("x", 0), y=item.get("y", 0),
                width=item.get("width", 50), height=item.get("height", 10),
                font_name=item.get("font_name", "Times-Roman"),
                font_size=item.get("font_size", 11),
                page_num=item.get("page", 1) - 1
            ))
        
        # ── Step 3: Align (35-60%) ──
        progress_tracker.update(task_id, progress=35, message="Aligning text...")
        
        if use_semantic:
            try:
                aligner = TextAligner(use_semantic=True)
                logger.info("Using semantic alignment")
                progress_tracker.update(task_id, progress=40, message="Loaded semantic model, aligning...")
            except Exception as e:
                logger.warning(f"Semantic alignment failed: {e}, falling back to AI")
                aligner = TextAligner(use_ai=True)
                progress_tracker.update(task_id, progress=40, message="Falling back to AI alignment...")
        else:
            aligner = TextAligner(use_ai=True)
        
        aligned = aligner.align(donor_frags, sample_frags)
        progress_tracker.update(
            task_id, progress=55,
            message=f"Aligned {len(aligned)} pairs"
        )
        logger.info(f"Aligned {len(aligned)} pairs")
        
        # ── Step 4: Generate PDF (60-95%) ──
        progress_tracker.update(task_id, progress=60, message="Generating output PDF...")
        
        generator = PDFGenerator(str(output_path))
        result_path = generator.generate_pdf(str(donor_path), aligned)
        
        progress_tracker.update(task_id, progress=90, message="Finalizing PDF...")
        logger.info(f"Generated PDF: {result_path}")
        
        # ── Done ──
        progress_tracker.complete(task_id)
        progress_tracker.update(task_id, progress=100, message="Complete!")
        
        return FileResponse(
            path=result_path,
            media_type="application/pdf",
            filename=f"aligned_{donor.filename}",
            headers={"X-Task-Id": task_id}
        )
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        progress_tracker.fail(task_id, str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}",
            headers={"X-Task-Id": task_id}
        )
    
    finally:
        # Cleanup input files
        for path in [donor_path, sample_path]:
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass