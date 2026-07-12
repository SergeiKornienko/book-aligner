"""
End-to-end processing endpoint.
"""

import uuid
import io
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import fitz
from PIL import Image

from backend.services.pdf_parser import PDFParser
from backend.services.text_alignment import TextAligner
from backend.services.models import TextFragment
from backend.services.pdf_generator import PDFGenerator
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
    Full end-to-end processing pipeline.
    
    1. Accepts two PDFs: donor (illustrated) and sample (translation)
    2. Parses both PDFs (with optional OCR)
    3. Aligns text (with optional semantic matching)
    4. Generates final PDF with translated text
    5. Returns the result
    """
    if not donor.filename.endswith('.pdf') or not sample.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Both files must be PDF")
    
    logger.info(f"Processing: donor={donor.filename}, sample={sample.filename}, semantic={use_semantic}, ocr={use_ocr}")
    
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
        # Step 1: Parse PDFs
        if use_ocr:
            from backend.services.ocr_processor import OCRProcessor
            
            logger.info("Using OCR mode")
            ocr = OCRProcessor()
            
            donor_doc = fitz.open(str(donor_path))
            donor_fragments = []
            for page_num in range(donor_doc.page_count):
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
            
            sample_doc = fitz.open(str(sample_path))
            sample_fragments = []
            for page_num in range(sample_doc.page_count):
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
            donor_parser = PDFParser(str(donor_path))
            sample_parser = PDFParser(str(sample_path))
            donor_fragments = donor_parser.extract_text_with_coordinates()
            sample_fragments = sample_parser.extract_text_with_coordinates()
            donor_parser.close()
            sample_parser.close()
        
        logger.info(f"Extracted {len(donor_fragments)} donor and {len(sample_fragments)} sample fragments")
        
        # Step 2: Convert to TextFragment
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
        
        # Step 3: Align
        if use_semantic:
            try:
                aligner = TextAligner(use_semantic=True)
                logger.info("Using semantic alignment")
            except Exception as e:
                logger.warning(f"Semantic alignment failed: {e}, falling back to AI")
                aligner = TextAligner(use_ai=True)
        else:
            aligner = TextAligner(use_ai=True)
        
        aligned = aligner.align(donor_frags, sample_frags)
        logger.info(f"Aligned {len(aligned)} pairs")
        
        # Step 4: Generate PDF
        generator = PDFGenerator(str(output_path))
        result_path = generator.generate_pdf(str(donor_path), aligned)
        logger.info(f"Generated PDF: {result_path}")
        
        return FileResponse(
            path=result_path,
            media_type="application/pdf",
            filename=f"aligned_{donor.filename}"
        )
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        for path in [donor_path, sample_path]:
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass