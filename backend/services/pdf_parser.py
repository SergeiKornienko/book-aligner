"""
PDF parser service using PyMuPDF (fitz)
Extracts text with precise coordinates and images
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF


class PDFParser:
    """Extract text, coordinates, and images from PDF files"""
    
    def __init__(self, filepath: str):
        """
        Initialize PDF parser with file path
        
        Args:
            filepath: Path to the PDF file
        """
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"PDF file not found: {filepath}")
        self.doc = fitz.open(str(self.filepath))
        
    def extract_text_with_coordinates(self) -> List[Dict[str, Any]]:
        """
        Extract text with precise coordinates (x, y, width, height)
        
        Returns:
            List of dicts with keys: page, x, y, width, height, text, font_name, font_size
        """
        result = []
        
        for page_num, page in enumerate(self.doc, start=1):
            # Get text with formatting info
            text_blocks = page.get_text("dict")  # Returns dict with blocks, lines, spans
            
            for block in text_blocks.get('blocks', []):
                if block.get('type') == 0:  # Text block
                    for line in block.get('lines', []):
                        for span in line.get('spans', []):
                            text = span.get('text', '').strip()
                            if text:  # Skip empty
                                result.append({
                                    'page': page_num,
                                    'x': span.get('origin', (0, 0))[0],
                                    'y': span.get('origin', (0, 0))[1],
                                    'width': span.get('size', 0) * len(text),  # Approximate
                                    'height': span.get('size', 0),
                                    'text': text,
                                    'font_name': span.get('font', 'unknown'),
                                    'font_size': span.get('size', 0),
                                })
        
        return result
    
    def extract_text_simple(self) -> List[Dict[str, Any]]:
        """
        Simple text extraction (line by line) without coordinates
        
        Returns:
            List of dicts with keys: page, line_num, text
        """
        result = []
        
        for page_num, page in enumerate(self.doc, start=1):
            text = page.get_text()
            if text:
                lines = text.split('\n')
                for line_num, line in enumerate(lines, start=1):
                    cleaned_line = line.strip()
                    if cleaned_line:
                        result.append({
                            'page': page_num,
                            'line_num': line_num,
                            'text': cleaned_line,
                        })
        
        return result
    
    def extract_images(self, output_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Extract images from PDF
        
        Args:
            output_dir: Directory to save images (optional)
            
        Returns:
            List of dicts with image metadata
        """
        images = []
        
        for page_num, page in enumerate(self.doc, start=1):
            image_list = page.get_images(full=True)
            
            for img_idx, img in enumerate(image_list):
                xref = img[0]
                try:
                    base_image = self.doc.extract_image(xref)
                    image_data = {
                        'page': page_num,
                        'index': img_idx,
                        'width': base_image.get('width', 0),
                        'height': base_image.get('height', 0),
                        'format': base_image.get('format', 'unknown'),
                        'image_data': base_image.get('image'),  # Binary image data
                        'xref': xref,
                    }
                    
                    if output_dir:
                        output_path = Path(output_dir) / f"page_{page_num}_img_{img_idx}.{base_image.get('format', 'png')}"
                        with open(output_path, 'wb') as f:
                            f.write(base_image.get('image'))
                        image_data['saved_path'] = str(output_path)
                    
                    images.append(image_data)
                except Exception as e:
                    print(f"Error extracting image {img_idx} on page {page_num}: {e}")
        
        return images
    
    def get_page_count(self) -> int:
        """Return total number of pages"""
        return len(self.doc)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Extract PDF metadata"""
        metadata = self.doc.metadata
        return {
            'title': metadata.get('title', ''),
            'author': metadata.get('author', ''),
            'subject': metadata.get('subject', ''),
            'creator': metadata.get('creator', ''),
            'producer': metadata.get('producer', ''),
            'creation_date': metadata.get('creationDate', ''),
            'modification_date': metadata.get('modDate', ''),
        }
    
    def get_page_size(self, page_num: int) -> Dict[str, float]:
        """Get page dimensions"""
        if page_num < 1 or page_num > len(self.doc):
            raise ValueError(f"Page {page_num} out of range (1-{len(self.doc)})")
        
        page = self.doc[page_num - 1]
        rect = page.rect
        return {
            'width': rect.width,
            'height': rect.height,
        }
    
    def close(self):
        """Close the PDF document"""
        if hasattr(self, 'doc'):
            self.doc.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()