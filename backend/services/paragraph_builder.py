"""
Paragraph builder service for grouping text fragments into semantic paragraphs.
Handles multi-page paragraphs, block type classification, and layout analysis.
"""

from typing import List, Tuple, Optional
from backend.services.models import TextFragment
from backend.services.block_classifier import BlockClassifier, BlockType


class Paragraph:
    """
    Represents a semantic paragraph composed of multiple text fragments.
    """
    
    def __init__(self, fragments: List[TextFragment], block_type: BlockType):
        """
        Initialize a paragraph from fragments.
        
        Args:
            fragments: List of TextFragment objects forming the paragraph
            block_type: Classified block type for this paragraph
        """
        self.fragments = fragments
        self.block_type = block_type
        self.text = self._join_text()
    
    @property
    def fragment_count(self) -> int:
        """Number of fragments in this paragraph"""
        return len(self.fragments)
    
    @property
    def y_start(self) -> float:
        """Y coordinate of the first fragment"""
        if not self.fragments:
            return 0.0
        return min(f.y for f in self.fragments)
    
    @property
    def y_end(self) -> float:
        """Y coordinate of the last fragment (bottom edge)"""
        if not self.fragments:
            return 0.0
        return max(f.y + f.height for f in self.fragments)
    
    @property
    def start_page(self) -> int:
        """Page number where paragraph starts"""
        if not self.fragments:
            return 0
        return min(f.page_num for f in self.fragments)
    
    @property
    def end_page(self) -> int:
        """Page number where paragraph ends"""
        if not self.fragments:
            return 0
        return max(f.page_num for f in self.fragments)
    
    @property
    def spans_pages(self) -> bool:
        """Whether this paragraph spans multiple pages"""
        return self.start_page != self.end_page
    
    @property
    def avg_font_size(self) -> float:
        """Average font size of fragments"""
        if not self.fragments:
            return 0.0
        return sum(f.font_size for f in self.fragments) / len(self.fragments)
    
    @property
    def x_position(self) -> float:
        """Average X position (for alignment)"""
        if not self.fragments:
            return 0.0
        return sum(f.x for f in self.fragments) / len(self.fragments)
    
    def _join_text(self) -> str:
        """
        Join fragment texts with proper spacing.
        Removes hyphenation at line breaks when possible.
        """
        if not self.fragments:
            return ""
        
        texts = []
        for i, fragment in enumerate(self.fragments):
            text = fragment.text.strip()
            
            # Handle hyphenated words across lines
            if text.endswith('-') and i < len(self.fragments) - 1:
                # Remove hyphen and join with next word
                text = text[:-1]
            
            texts.append(text)
        
        return ' '.join(texts)
    
    def __repr__(self) -> str:
        return f"Paragraph(type={self.block_type.name}, fragments={self.fragment_count}, pages={self.start_page}-{self.end_page})"


class ParagraphBuilder:
    """
    Builds paragraphs from text fragments by analyzing layout and content.
    
    Algorithm:
    1. Classify each fragment by block type
    2. Group fragments by type and position
    3. Detect paragraph boundaries (gaps, font changes, page breaks)
    4. Handle cross-page paragraph continuation
    5. Filter out ignored blocks (headers, footers, page numbers)
    """
    
    def __init__(self):
        """Initialize ParagraphBuilder with default thresholds"""
        self.classifier = BlockClassifier()
        
        # Thresholds for paragraph detection
        self.line_spacing_tolerance = 1.5    # Multiplier for detecting line breaks
        self.paragraph_gap_threshold = 2.2    # Multiplier for detecting paragraph breaks
        self.font_size_tolerance = 2          # Max font size difference for same paragraph
        self.x_position_tolerance = 20        # Max X position difference for same paragraph
    
    def build_paragraphs(
        self,
        fragments: List[TextFragment],
        page_dimensions: dict
    ) -> List[Paragraph]:
        """
        Build paragraphs from text fragments.
        
        Args:
            fragments: List of text fragments from PDF
            page_dimensions: Dict with 'width' and 'height' of page
            
        Returns:
            List of Paragraph objects
        """
        if not fragments:
            return []
        
        # Step 1: Classify all fragments
        classified = self._classify_fragments(fragments, page_dimensions)
        
        # Step 2: Filter out ignored blocks
        body_fragments = self._filter_ignored(classified)
        
        if not body_fragments:
            return []
        
        # Step 3: Sort fragments by page and position
        sorted_fragments = sorted(
            body_fragments,
            key=lambda x: (x[0].page_num, x[0].y, x[0].x)
        )
        
        # Step 4: Group into paragraphs
        paragraphs = self._group_into_paragraphs(sorted_fragments, page_dimensions)
        
        return paragraphs
    
    def _classify_fragments(
        self,
        fragments: List[TextFragment],
        page_dimensions: dict
    ) -> List[Tuple[TextFragment, BlockType, float]]:
        """Classify all fragments"""
        classified = []
        for fragment in fragments:
            block_type, confidence = self.classifier.classify_block(
                [fragment], page_dimensions
            )
            classified.append((fragment, block_type, confidence))
        return classified
    
    def _filter_ignored(
        self,
        classified: List[Tuple[TextFragment, BlockType, float]]
    ) -> List[Tuple[TextFragment, BlockType, float]]:
        """Remove headers, footers, and page numbers"""
        return [
            (frag, btype, conf)
            for frag, btype, conf in classified
            if not self.classifier.should_ignore(btype)
        ]
    
    def _group_into_paragraphs(
        self,
        classified: List[Tuple[TextFragment, BlockType, float]],
        page_dimensions: dict
    ) -> List[Paragraph]:
        """
        Group classified fragments into paragraphs.
        
        Rules for paragraph boundaries:
        1. Different block types start new paragraphs
        2. Large vertical gaps start new paragraphs
        3. Significant font size changes start new paragraphs
        4. X position changes start new paragraphs (indentation)
        5. Page breaks MAY continue paragraphs (context-dependent)
        """
        if not classified:
            return []
        
        paragraphs = []
        current_group = []
        current_type = None
        
        for i, (fragment, block_type, confidence) in enumerate(classified):
            # First fragment always starts a group
            if not current_group:
                current_group = [fragment]
                current_type = block_type
                continue
            
            prev_fragment = current_group[-1]
            
            # Check if we should start a new paragraph
            if self._is_paragraph_boundary(
                prev_fragment, fragment, block_type, current_type, classified, i, page_dimensions
            ):
                # Save current paragraph
                paragraphs.append(Paragraph(current_group, current_type))
                
                # Start new paragraph
                current_group = [fragment]
                current_type = block_type
            else:
                # Continue current paragraph
                current_group.append(fragment)
        
        # Don't forget the last paragraph
        if current_group:
            paragraphs.append(Paragraph(current_group, current_type))
        
        return paragraphs
    
    def _is_paragraph_boundary(
        self,
        prev_fragment: TextFragment,
        curr_fragment: TextFragment,
        curr_block_type: BlockType,
        current_group_type: BlockType,
        all_classified: List,
        curr_index: int,
        page_dimensions: dict
    ) -> bool:
        """
        Determine if there's a paragraph boundary between two fragments.
        
        Returns True if a new paragraph should start.
        """
        # Rule 1: Different block types
        if curr_block_type != current_group_type:
            return True
        
        # Rule 2: Different pages
        if curr_fragment.page_num != prev_fragment.page_num:
            return self._is_page_boundary_paragraph_break(
                prev_fragment, curr_fragment, all_classified, curr_index
            )
        
        # Rule 3: Same page - check vertical gap
        prev_bottom = prev_fragment.y + prev_fragment.height
        curr_top = curr_fragment.y
        vertical_gap = curr_top - prev_bottom
        
        # Calculate expected line spacing based on font size
        expected_spacing = prev_fragment.font_size * 1.2
        
        # Large gap indicates paragraph break
        if vertical_gap > expected_spacing * self.paragraph_gap_threshold:
            return True
        
        # Rule 4: Significant font size change
        if abs(curr_fragment.font_size - prev_fragment.font_size) > self.font_size_tolerance:
            return True
        
        # Rule 5: Significant X position change (indentation)
        if abs(curr_fragment.x - prev_fragment.x) > self.x_position_tolerance:
            # Special case: first line indent (positive indent)
            if curr_fragment.x > prev_fragment.x + 10:
                return True
        
        return False
    
    def _is_page_boundary_paragraph_break(
        self,
        prev_fragment: TextFragment,
        curr_fragment: TextFragment,
        all_classified: List,
        curr_index: int
    ) -> bool:
        """
        Determine if a page break indicates a paragraph break.
        
        A page break CONTINUES the paragraph if:
        - Current fragment starts near top of page (continuation)
        - Same formatting (font, size, x position)
        - Previous fragment was near bottom of page
        
        A page break STARTS new paragraph if:
        - Current fragment is further down the page (not at top)
        - Different formatting
        """
        # If current fragment is near top of page (y < 100), it's likely a continuation
        if curr_fragment.y < 100:
            # Check formatting similarity
            same_font_size = abs(curr_fragment.font_size - prev_fragment.font_size) <= self.font_size_tolerance
            similar_x = abs(curr_fragment.x - prev_fragment.x) <= self.x_position_tolerance
            
            if same_font_size and similar_x:
                return False  # Continue paragraph across pages
        
        return True  # Start new paragraph
    
    def get_paragraphs_by_type(
        self,
        paragraphs: List[Paragraph],
        block_type: BlockType
    ) -> List[Paragraph]:
        """
        Filter paragraphs by block type.
        
        Args:
            paragraphs: List of all paragraphs
            block_type: Desired block type
            
        Returns:
            Filtered list of paragraphs
        """
        return [p for p in paragraphs if p.block_type == block_type]