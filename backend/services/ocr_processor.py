"""
OCR Processor for BookAligner.
Extracts text regions from illustrated/scanned PDF pages.
Uses Tesseract OCR with preprocessing for better accuracy.
"""

from dataclasses import dataclass
from typing import List, Tuple
import io
import numpy as np
from PIL import Image


@dataclass
class TextRegion:
    """
    Represents a detected text region on a page.
    """
    text: str
    x: float
    y: float
    width: float
    height: float
    confidence: float = 0.0
    page_num: int = 0
    
    @property
    def area(self) -> float:
        """Area of the text region"""
        return self.width * self.height


class OCRProcessor:
    """
    Processes illustrated PDF pages to extract text regions.
    
    Capabilities:
    - Detect if page has text
    - Find text bounding boxes
    - Extract text only from text areas
    - Classify page type (text/illustration/mixed)
    """
    
    def __init__(self):
        """Initialize OCR processor."""
        self.text_confidence_threshold = 0.3
    
    def has_text(self, image: Image.Image) -> bool:
        """
        Check if an image contains text using Tesseract directly.
        
        Args:
            image: PIL Image
            
        Returns:
            True if text detected
        """
        try:
            import pytesseract
            
            text = pytesseract.image_to_string(
                image,
                config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
            )
            alpha_chars = sum(c.isalpha() for c in text)
            return alpha_chars > 10
            
        except ImportError:
            return True
    
    def find_text_areas(self, image: Image.Image) -> List[Tuple[int, int, int, int]]:
        """
        Find bounding boxes of text areas using Tesseract.
        
        Args:
            image: PIL Image
            
        Returns:
            List of (x, y, width, height) tuples
        """
        try:
            import pytesseract
            
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            areas = []
            for i, text in enumerate(data['text']):
                if text.strip() and int(data['conf'][i]) > 30:
                    x = data['left'][i]
                    y = data['top'][i]
                    w = data['width'][i]
                    h = data['height'][i]
                    if w > 10 and h > 5:
                        areas.append((int(x), int(y), int(w), int(h)))
            
            return self._merge_areas(areas)
            
        except ImportError:
            return []
    
    def _merge_areas(self, areas: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """
        Merge nearby text areas into line-level regions.
        
        Args:
            areas: List of (x, y, w, h) tuples
            
        Returns:
            Merged areas
        """
        if not areas:
            return []
        
        # Sort by Y position
        areas.sort(key=lambda a: a[1])
        
        merged = []
        current = list(areas[0])
        
        for area in areas[1:]:
            # If same line (Y within height), merge horizontally
            if abs(area[1] - current[1]) < current[3]:
                current[2] = max(current[0] + current[2], area[0] + area[2]) - current[0]
                current[3] = max(current[3], area[3])
            else:
                merged.append(tuple(current))
                current = list(area)
        
        merged.append(tuple(current))
        return merged
    
    def classify_page(self, image: Image.Image) -> str:
        """
        Classify page type using Tesseract.
        
        Args:
            image: PIL Image
            
        Returns:
            'text', 'illustration', or 'mixed'
        """
        try:
            import pytesseract
            text = pytesseract.image_to_string(image)
            alpha_chars = sum(c.isalpha() for c in text)
            
            if alpha_chars < 20:
                return "illustration"
            elif alpha_chars < 200:
                return "mixed"
            else:
                return "text"
        except ImportError:
            return "text"
    
    def _preprocess_for_ocr(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for better OCR results.
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed PIL Image
        """
        try:
            import cv2
            
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            
            # Increase contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Denoise
            denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
            
            return Image.fromarray(denoised)
            
        except ImportError:
            return image
    
    def extract_text_regions(
        self,
        image: Image.Image,
        page_num: int = 0,
        lang: str = 'eng'
    ) -> List[TextRegion]:
        """
        Extract text regions using Tesseract with detailed positioning.
        
        Args:
            image: PIL Image of the page
            page_num: Page number
            lang: Tesseract language code
            
        Returns:
            List of TextRegion objects
        """
        try:
            import pytesseract
        except ImportError:
            return []
        
        # Get detailed OCR data
        data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
        
        regions = []
        current_text = []
        current_x, current_y, current_w, current_h = 0, 0, 0, 0
        current_conf = 0
        line_num = -1
        
        for i, text in enumerate(data['text']):
            if not text.strip():
                continue
            
            conf = int(data['conf'][i]) if data['conf'][i] != '-1' else 50
            
            if data['line_num'][i] != line_num:
                # Save previous line
                if current_text:
                    regions.append(TextRegion(
                        text=' '.join(current_text),
                        x=float(current_x),
                        y=float(current_y),
                        width=float(current_w),
                        height=float(current_h),
                        confidence=current_conf / max(len(current_text), 1) / 100,
                        page_num=page_num
                    ))
                
                # Start new line
                current_text = [text]
                current_x = data['left'][i]
                current_y = data['top'][i]
                current_w = data['width'][i]
                current_h = data['height'][i]
                current_conf = conf
                line_num = data['line_num'][i]
            else:
                # Continue line
                current_text.append(text)
                current_w = max(current_x + current_w, data['left'][i] + data['width'][i]) - current_x
                current_h = max(current_h, data['height'][i])
                current_conf += conf
        
        # Save last line
        if current_text:
            regions.append(TextRegion(
                text=' '.join(current_text),
                x=float(current_x),
                y=float(current_y),
                width=float(current_w),
                height=float(current_h),
                confidence=current_conf / max(len(current_text), 1) / 100,
                page_num=page_num
            ))
        
        return regions