"""
PDF Generator service for BookAligner.
Creates final PDF with translated text and preserved illustrations.
"""

from pathlib import Path
from typing import List, Tuple
import fitz  # PyMuPDF
from backend.services.models import TextFragment


class PDFGenerator:
    """
    Generates the final PDF by combining:
    - Original donor PDF structure (layout, images)
    - Aligned sample text (translation)
    
    Pipeline:
    1. Create blank pages matching donor dimensions
    2. Copy images from donor at original positions
    3. Insert translated text at correct positions
    4. Save the result
    
    Does NOT copy original text — only images and layout.
    """
    
    def __init__(self, output_path: str):
        """
        Initialize PDF generator.
        
        Args:
            output_path: Path where the final PDF will be saved
        """
        self.output_path = str(Path(output_path).resolve())
        self.doc = None
    
    def _save(self):
        """Save document to temp file and replace original."""
        if self.doc is None:
            return
        temp_path = self.output_path + ".tmp"
        self.doc.save(temp_path)
        self.doc.close()
        self.doc = None
        Path(temp_path).replace(self.output_path)
    
    def _open(self):
        """Open or create document."""
        if Path(self.output_path).exists():
            self.doc = fitz.open(self.output_path)
        else:
            self.doc = fitz.open()
    
    def create_blank_pdf(self) -> str:
        """
        Create a blank PDF with 1 empty page.
        
        Returns:
            Path to the created PDF
        """
        self.doc = fitz.open()
        self.doc.new_page()
        self.doc.save(self.output_path)
        self.doc.close()
        self.doc = None
        return self.output_path
    
    def add_page(self, width: float = 595, height: float = 842) -> int:
        """
        Add a new page to the PDF.
        
        Args:
            width: Page width in points
            height: Page height in points
            
        Returns:
            Page number (0-based)
        """
        self._open()
        page = self.doc.new_page(width=width, height=height)
        page_num = self.doc.page_count - 1
        self._save()
        return page_num
    
    def insert_text_on_page(
        self,
        page_num: int,
        aligned_fragments: List[Tuple[TextFragment, TextFragment]]
    ) -> None:
        """
        Insert translated text at correct positions on a page.
        For use with blank pages — does NOT try to cover old text.
        
        Args:
            page_num: Page number to modify (0-based)
            aligned_fragments: List of (donor_fragment, sample_fragment) pairs
        """
        self._open()
        
        if page_num >= self.doc.page_count:
            self.doc.close()
            self.doc = None
            raise ValueError(f"Page {page_num} does not exist")
        
        page = self.doc[page_num]
        
        for donor_frag, sample_frag in aligned_fragments:
            # Skip if fragments are on different pages
            if donor_frag.page_num != page_num:
                continue
            
            # Calculate position from donor fragment
            x0 = donor_frag.x
            y0 = donor_frag.y
            y1 = donor_frag.y + donor_frag.height
            
            # Insert text at donor's position
            text_point = fitz.Point(x0, y1 - 2)
            
            page.insert_text(
                text_point,
                sample_frag.text,
                fontname="china-s",  # Has Cyrillic glyphs
                fontsize=sample_frag.font_size
            )
        
        self._save()
    
    def replace_text_on_page(
        self,
        page_num: int,
        aligned_fragments: List[Tuple[TextFragment, TextFragment]]
    ) -> None:
        """
        Replace text on a page with aligned translations.
        Covers original text with white rectangles, then inserts translation.
        
        Args:
            page_num: Page number to modify (0-based)
            aligned_fragments: List of (donor_fragment, sample_fragment) pairs
        """
        self._open()
        
        if page_num >= self.doc.page_count:
            self.doc.close()
            self.doc = None
            raise ValueError(f"Page {page_num} does not exist")
        
        page = self.doc[page_num]
        
        for donor_frag, sample_frag in aligned_fragments:
            if donor_frag.page_num != page_num:
                continue
            
            x0 = donor_frag.x
            y0 = donor_frag.y
            x1 = donor_frag.x + donor_frag.width
            y1 = donor_frag.y + donor_frag.height
            
            # Cover original text with white rectangle
            white_rect = fitz.Rect(x0 - 2, y0 - 1, x1 + 2, y1 + 2)
            page.draw_rect(white_rect, color=None, fill=(1, 1, 1), width=0)
            
            # Insert translation
            text_point = fitz.Point(x0, y1 - 2)
            
            page.insert_text(
                text_point,
                sample_frag.text,
                fontname="china-s",
                fontsize=sample_frag.font_size
            )
        
        self._save()
    
    def copy_images_from_donor(
        self,
        donor_pdf_path: str,
        donor_page: int = 0,
        target_page: int = 0
    ) -> None:
        """
        Copy images from donor PDF to the generated PDF.
        
        Args:
            donor_pdf_path: Path to the donor PDF
            donor_page: Page number in donor PDF (0-based)
            target_page: Page number in generated PDF (0-based)
        """
        self._open()
        donor_doc = fitz.open(donor_pdf_path)
        
        if donor_page >= donor_doc.page_count or target_page >= self.doc.page_count:
            donor_doc.close()
            self.doc.close()
            self.doc = None
            return
        
        donor_page_obj = donor_doc[donor_page]
        target_page_obj = self.doc[target_page]
        
        # Extract images from donor page
        image_list = donor_page_obj.get_images(full=True)
        
        for img_info in image_list:
            xref = img_info[0]
            
            # Extract the image
            base_image = donor_doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Get image position on donor page
            image_rects = donor_page_obj.get_image_rects(xref)
            
            for rect in image_rects:
                # Convert to fitz.Rect if needed
                if isinstance(rect, tuple):
                    rect = fitz.Rect(*rect)
                
                # Insert image at the same position
                target_page_obj.insert_image(
                    rect,
                    stream=image_bytes,
                    keep_proportion=True
                )
        
        donor_doc.close()
        self._save()
    
    def generate_pdf(
        self,
        donor_pdf_path: str,
        aligned_fragments: List[Tuple[TextFragment, TextFragment]]
    ) -> str:
        """
        Full PDF generation pipeline.
        
        1. Creates blank pages matching donor dimensions
        2. Copies images from donor at original positions
        3. Inserts translated text at correct positions
        4. Saves and returns the output path
        
        Args:
            donor_pdf_path: Path to the donor PDF
            aligned_fragments: Aligned (donor, sample) fragment pairs
            
        Returns:
            Path to the generated PDF
        """
        # Check donor exists
        if not Path(donor_pdf_path).exists():
            raise FileNotFoundError(f"Donor PDF not found: {donor_pdf_path}")
        
        # Open donor to get page structure
        donor_doc = fitz.open(donor_pdf_path)
        num_pages = donor_doc.page_count
        
        # Create blank document with same page sizes
        self.doc = fitz.open()
        for i in range(num_pages):
            donor_page = donor_doc[i]
            self.doc.new_page(
                width=donor_page.rect.width,
                height=donor_page.rect.height
            )
        
        self._save()
        donor_doc.close()
        
        # Copy images from each donor page
        for i in range(num_pages):
            try:
                self.copy_images_from_donor(donor_pdf_path, donor_page=i, target_page=i)
            except Exception:
                pass  # Continue even if image copy fails
        
        # Insert translated text on each page
        for i in range(num_pages):
            page_fragments = [
                (d, s) for d, s in aligned_fragments
                if d.page_num == i
            ]
            if page_fragments:
                self.insert_text_on_page(i, page_fragments)
        
        return self.output_path
    
    def _map_font(self, font_name: str) -> str:
        """
        Map font names to PyMuPDF compatible font names.
        
        Args:
            font_name: Original font name
            
        Returns:
            PyMuPDF-compatible font name
        """
        font_lower = font_name.lower()
        
        if "arial" in font_lower:
            if "bold" in font_lower:
                return "Helvetica-Bold"
            elif "italic" in font_lower or "oblique" in font_lower:
                return "Helvetica-Oblique"
            return "Helvetica"
        
        if "times" in font_lower:
            if "bold" in font_lower:
                return "Times-Bold"
            elif "italic" in font_lower:
                return "Times-Italic"
            return "Times-Roman"
        
        if "helvetica" in font_lower:
            if "bold" in font_lower:
                return "Helvetica-Bold"
            elif "italic" in font_lower or "oblique" in font_lower:
                return "Helvetica-Oblique"
            return "Helvetica"
        
        if "courier" in font_lower:
            if "bold" in font_lower:
                return "Courier-Bold"
            elif "italic" in font_lower or "oblique" in font_lower:
                return "Courier-Oblique"
            return "Courier"
        
        return "Helvetica"