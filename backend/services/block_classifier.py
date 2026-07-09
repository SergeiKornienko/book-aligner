"""
Block classification service for PDF text elements.
Identifies different types of text blocks: titles, body text, headers, etc.
"""

from enum import Enum, auto
from typing import List, Tuple
from backend.services.models import TextFragment


class BlockType(Enum):
    """Types of text blocks that can appear in a book PDF"""
    TITLE = auto()        # Chapter titles (large font, centered)
    SUBTITLE = auto()     # Subtitles (medium font, often italic)
    BODY = auto()         # Regular body text paragraphs
    CAPTION = auto()      # Image captions (small font, near images)
    HEADER = auto()       # Page headers/running heads
    FOOTER = auto()       # Page footers
    PAGE_NUMBER = auto()  # Page numbers
    DROPCAP = auto()      # Drop caps (large first letter)
    UNKNOWN = auto()      # Unclassified blocks


class BlockClassifier:
    """
    Classifies text blocks based on their properties:
    - Position on page (x, y coordinates)
    - Font size and style
    - Relationship to other blocks
    - Position relative to page boundaries
    """
    
    def __init__(self):
        """Initialize BlockClassifier with default thresholds"""
        # Position thresholds
        self.header_margin = 50      # Top margin for headers (points)
        self.footer_margin = 50      # Bottom margin for footers (points)
        
        # Font size thresholds
        self.title_min_size = 18     # Minimum font size for titles
        self.subtitle_min_size = 14  # Minimum font size for subtitles
        self.subtitle_max_size = 17  # Maximum font size for subtitles
        self.body_min_size = 10      # Minimum font size for body text
        self.body_max_size = 13      # Maximum font size for body text
        self.caption_max_size = 9    # Maximum font size for captions
        self.page_number_max_size = 10  # Maximum font size for page numbers
        self.dropcap_min_size = 24   # Minimum font size for drop caps
    
    def classify_block(
        self,
        fragments: List[TextFragment],
        page_dimensions: dict
    ) -> Tuple[BlockType, float]:
        """
        Classify a group of text fragments into a block type.
        
        Args:
            fragments: List of text fragments forming a block
            page_dimensions: Dict with 'width' and 'height' of the page
            
        Returns:
            Tuple of (BlockType, confidence_score)
        """
        if not fragments:
            return BlockType.UNKNOWN, 0.0
        
        page_width = page_dimensions.get('width', 595)
        page_height = page_dimensions.get('height', 842)
        
        # Extract features from fragments
        avg_font_size = sum(f.font_size for f in fragments) / len(fragments)
        avg_y = sum(f.y for f in fragments) / len(fragments)
        avg_x = sum(f.x for f in fragments) / len(fragments)
        min_y = min(f.y for f in fragments)
        max_y = max(f.y for f in fragments) + fragments[0].height if fragments else 0
        max_font_size = max(f.font_size for f in fragments)
        text_content = ' '.join(f.text for f in fragments).strip()
        
        is_bold = any('bold' in f.font_name.lower() for f in fragments)
        is_italic = any('italic' in f.font_name.lower() for f in fragments)
        is_centered = self._is_centered(avg_x, page_width)
        is_single_char = len(fragments) == 1 and len(fragments[0].text.strip()) <= 3
        
        # Classification rules (in order of priority)
        
        # 1. Drop cap: single large character
        if is_single_char and max_font_size >= self.dropcap_min_size:
            return BlockType.DROPCAP, 0.9
        
        # 2. Page number: near bottom/top, small font, contains digits
        is_near_top = min_y <= self.header_margin
        is_near_bottom = max_y >= page_height - self.footer_margin
        
        if is_near_bottom or is_near_top:
            if avg_font_size <= self.page_number_max_size and len(fragments) == 1:
                # Check if text looks like a page number (digits, possibly with punctuation)
                if self._is_page_number(text_content):
                    return BlockType.PAGE_NUMBER, 0.95
        
        # 3. Header: near top margin, small font
        if is_near_top and avg_font_size <= self.page_number_max_size:
            return BlockType.HEADER, 0.8
        
        # 4. Footer: near bottom margin, small font
        if is_near_bottom and avg_font_size <= self.page_number_max_size:
            return BlockType.FOOTER, 0.8
        
        # 5. Title: very large font, often centered and bold
        if max_font_size >= self.title_min_size:
            confidence = 0.75
            if is_centered:
                confidence += 0.15
            if is_bold:
                confidence += 0.1
            return BlockType.TITLE, min(confidence, 1.0)
        
        # 6. Subtitle: medium-large font, often italic or centered
        if self.subtitle_min_size <= max_font_size <= self.subtitle_max_size:
            confidence = 0.65
            if is_italic:
                confidence += 0.15
            if is_centered:
                confidence += 0.1
            return BlockType.SUBTITLE, min(confidence, 1.0)
        
        # 7. Caption: very small font, often below images
        if avg_font_size <= self.caption_max_size:
            # Often positioned lower on page (near images) or centered
            if avg_y > page_height * 0.5 or is_centered:
                return BlockType.CAPTION, 0.6
        
        # 8. Body text: regular font size, normal position
        if self.body_min_size <= avg_font_size <= self.body_max_size:
            # Higher confidence if there are multiple lines
            confidence = 0.7 + min(len(fragments) * 0.05, 0.25)
            return BlockType.BODY, min(confidence, 1.0)
        
        # 9. Unknown: couldn't classify
        return BlockType.UNKNOWN, 0.3
    
    def _is_page_number(self, text: str) -> bool:
        """
        Check if text content looks like a page number.
        
        Args:
            text: Text content to check
            
        Returns:
            True if text looks like a page number
        """
        # Remove common formatting
        cleaned = text.strip().replace('-', '').replace('.', '').replace(' ', '')
        # Check if it's just digits (possibly with decorative characters)
        if cleaned.isdigit():
            return True
        # Check for patterns like "42 |" or "| 42"
        if any(c.isdigit() for c in cleaned) and len(cleaned) <= 5:
            return True
        return False
    
    def _is_centered(self, x: float, page_width: float) -> bool:
        """
        Check if a text block is centered horizontally on the page.
        
        Args:
            x: X coordinate of the block
            page_width: Width of the page
            
        Returns:
            True if centered, False otherwise
        """
        center_margin = page_width * 0.25  # 25% tolerance
        page_center = page_width / 2
        return abs(x - page_center) < center_margin
    
    def should_ignore(self, block_type: BlockType) -> bool:
        """
        Determine if a block type should be ignored during alignment.
        
        Args:
            block_type: The classified block type
            
        Returns:
            True if the block should be ignored
        """
        ignored_types = {
            BlockType.PAGE_NUMBER,
            BlockType.HEADER,
            BlockType.FOOTER,
        }
        return block_type in ignored_types
    
    def should_align_separately(self, block_type: BlockType) -> bool:
        """
        Determine if a block type needs special alignment handling.
        
        Args:
            block_type: The classified block type
            
        Returns:
            True if the block needs special alignment
        """
        special_types = {
            BlockType.TITLE,
            BlockType.SUBTITLE,
            BlockType.CAPTION,
        }
        return block_type in special_types