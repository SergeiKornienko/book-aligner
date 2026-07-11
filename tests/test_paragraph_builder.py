"""
Tests for paragraph grouping functionality.
Groups text fragments into semantic paragraphs based on layout.
"""

import pytest
from backend.services.models import TextFragment
from backend.services.block_classifier import BlockType
from backend.services.paragraph_builder import ParagraphBuilder, Paragraph


class TestParagraph:
    """Tests for Paragraph data class"""
    
    def test_create_paragraph(self):
        """Test creating a paragraph"""
        fragments = [
            TextFragment("First line", 10, 10, 100, 12),
            TextFragment("Second line", 10, 25, 100, 12),
        ]
        para = Paragraph(fragments, BlockType.BODY)
        
        assert para.text == "First line Second line"
        assert para.fragment_count == 2
        assert para.y_start == 10
        assert para.y_end == 37
        assert para.block_type == BlockType.BODY
    
    def test_paragraph_full_text(self):
        """Test full text with proper spacing"""
        fragments = [
            TextFragment("Hello", 10, 10, 100, 12),
            TextFragment("world", 10, 25, 100, 12),
        ]
        para = Paragraph(fragments, BlockType.BODY)
        assert para.text == "Hello world"
    
    def test_paragraph_single_fragment(self):
        """Test paragraph with single fragment"""
        fragments = [TextFragment("Title", 100, 50, 200, 20, font_size=20)]
        para = Paragraph(fragments, BlockType.TITLE)
        assert para.text == "Title"
        assert para.fragment_count == 1
    
    def test_paragraph_with_page_numbers(self):
        """Test paragraph spanning pages"""
        fragments = [
            TextFragment("Line 1", 10, 10, 100, 12, page_num=1),
            TextFragment("Line 2", 10, 25, 100, 12, page_num=1),
            TextFragment("Line 3", 10, 10, 100, 12, page_num=2),
        ]
        para = Paragraph(fragments, BlockType.BODY)
        assert para.start_page == 1
        assert para.end_page == 2
        assert para.spans_pages == True


class TestParagraphBuilder:
    """Tests for ParagraphBuilder service"""
    
    @pytest.fixture
    def builder(self):
        """Create ParagraphBuilder instance"""
        return ParagraphBuilder()
    
    @pytest.fixture
    def page_dimensions(self):
        """Standard page dimensions"""
        return {"width": 595, "height": 842}
    
    @pytest.fixture
    def multi_paragraph_fragments(self):
        """Fragments forming multiple paragraphs on one page"""
        return [
            # Paragraph 1 - Title
            TextFragment("CHAPTER ONE", 200, 50, 195, 20, font_name="Times-Bold", font_size=20),
            # Paragraph 2 - Subtitle
            TextFragment("THE BOY WHO LIVED", 180, 80, 235, 16, font_name="Times-Italic", font_size=16),
            # Paragraph 3 - Body text
            TextFragment("Mr. and Mrs. Dursley, of number", 72, 150, 450, 12, font_size=11),
            TextFragment("four, Privet Drive, were proud", 72, 165, 450, 12, font_size=11),
            TextFragment("to say that they were perfectly", 72, 180, 450, 12, font_size=11),
            TextFragment("normal, thank you very much.", 72, 195, 450, 12, font_size=11),
            # Gap - new paragraph
            # Paragraph 4 - Body text
            TextFragment("They were the last people", 72, 240, 450, 12, font_size=11),
            TextFragment("you'd expect to be involved", 72, 255, 450, 12, font_size=11),
            TextFragment("in anything strange or", 72, 270, 450, 12, font_size=11),
            TextFragment("mysterious, because they just", 72, 285, 450, 12, font_size=11),
            TextFragment("didn't hold with such nonsense.", 72, 300, 450, 12, font_size=11),
        ]
    
    def test_builder_initialization(self, builder):
        """Test that ParagraphBuilder initializes correctly"""
        assert builder is not None
        assert hasattr(builder, 'build_paragraphs')
    
    def test_build_paragraphs_count(self, builder, multi_paragraph_fragments, page_dimensions):
        """Test correct number of paragraphs identified"""
        paragraphs = builder.build_paragraphs(multi_paragraph_fragments, page_dimensions)
        assert len(paragraphs) == 4  # Title, Subtitle, Body1, Body2
    
    def test_build_paragraphs_types(self, builder, multi_paragraph_fragments, page_dimensions):
        """Test that paragraphs have correct types"""
        paragraphs = builder.build_paragraphs(multi_paragraph_fragments, page_dimensions)
        
        assert paragraphs[0].block_type == BlockType.TITLE
        assert paragraphs[1].block_type == BlockType.SUBTITLE
        assert paragraphs[2].block_type == BlockType.BODY
        assert paragraphs[3].block_type == BlockType.BODY
    
    def test_build_paragraphs_text(self, builder, multi_paragraph_fragments, page_dimensions):
        """Test that paragraph text is correctly joined"""
        paragraphs = builder.build_paragraphs(multi_paragraph_fragments, page_dimensions)
        
        # Title
        assert paragraphs[0].text == "CHAPTER ONE"
        
        # Subtitle
        assert paragraphs[1].text == "THE BOY WHO LIVED"
        
        # Body paragraph 1
        assert "Mr. and Mrs. Dursley" in paragraphs[2].text
        assert "thank you very much." in paragraphs[2].text
        
        # Body paragraph 2
        assert "They were the last people" in paragraphs[3].text
        assert "such nonsense." in paragraphs[3].text
    
    def test_empty_fragments(self, builder, page_dimensions):
        """Test with empty fragment list"""
        paragraphs = builder.build_paragraphs([], page_dimensions)
        assert paragraphs == []
    
    def test_single_fragment(self, builder, page_dimensions):
        """Test with single fragment"""
        fragments = [TextFragment("Single line", 10, 10, 100, 12)]
        paragraphs = builder.build_paragraphs(fragments, page_dimensions)
        assert len(paragraphs) == 1
        assert paragraphs[0].text == "Single line"
    
    def test_large_gap_creates_new_paragraph(self, builder, page_dimensions):
        """Test that large vertical gaps create new paragraphs"""
        fragments = [
            TextFragment("Line 1", 72, 10, 100, 12, font_size=11),
            TextFragment("Line 2", 72, 25, 100, 12, font_size=11),
            TextFragment("Line 3", 72, 100, 100, 12, font_size=11),  # Large gap
            TextFragment("Line 4", 72, 115, 100, 12, font_size=11),
        ]
        
        paragraphs = builder.build_paragraphs(fragments, page_dimensions)
        assert len(paragraphs) == 2
        assert paragraphs[0].text == "Line 1 Line 2"
        assert paragraphs[1].text == "Line 3 Line 4"
    
    def test_font_change_creates_new_paragraph(self, builder, page_dimensions):
        """Test that font size change triggers new paragraph"""
        fragments = [
            TextFragment("Title text", 200, 10, 195, 20, font_size=20),
            TextFragment("Body line 1", 72, 50, 450, 12, font_size=11),
            TextFragment("Body line 2", 72, 65, 450, 12, font_size=11),
        ]
        
        paragraphs = builder.build_paragraphs(fragments, page_dimensions)
        assert len(paragraphs) == 2  # Title and body
    
    def test_paragraph_across_pages(self, builder, page_dimensions):
        """Test that paragraphs can span across pages"""
        fragments = [
            # Page 1 - end of paragraph
            TextFragment("Mr. and Mrs. Dursley, of", 72, 10, 450, 12, font_size=11, page_num=1),
            TextFragment("number four, Privet Drive,", 72, 25, 450, 12, font_size=11, page_num=1),
            TextFragment("were proud to", 72, 40, 450, 12, font_size=11, page_num=1),
            # Page 2 - continuation of same paragraph (starts at top, same formatting)
            TextFragment("say that they were perfectly", 72, 10, 450, 12, font_size=11, page_num=2),
            TextFragment("normal, thank you very much.", 72, 25, 450, 12, font_size=11, page_num=2),
        ]
        
        paragraphs = builder.build_paragraphs(fragments, page_dimensions)
        
        # Should be one paragraph spanning two pages
        assert len(paragraphs) == 1
        para = paragraphs[0]
        assert para.fragment_count == 5
        assert para.spans_pages == True
        assert para.start_page == 1
        assert para.end_page == 2
        assert "Mr. and Mrs. Dursley" in para.text
        assert "thank you very much." in para.text
    
    def test_page_break_new_paragraph(self, builder, page_dimensions):
        """Test that page break with large gap creates new paragraph"""
        fragments = [
            # Page 1 - complete paragraph
            TextFragment("End of paragraph on page 1.", 72, 200, 450, 12, font_size=11, page_num=1),
            TextFragment("More text here.", 72, 215, 450, 12, font_size=11, page_num=1),
            # Page 2 - new paragraph (starts lower on page, not at top)
            TextFragment("New paragraph on page 2.", 72, 150, 450, 12, font_size=11, page_num=2),
            TextFragment("Continuing here.", 72, 165, 450, 12, font_size=11, page_num=2),
        ]
        
        paragraphs = builder.build_paragraphs(fragments, page_dimensions)
        
        # Should be two separate paragraphs
        assert len(paragraphs) == 2
        assert paragraphs[0].start_page == 1
        assert paragraphs[0].end_page == 1
        assert paragraphs[1].start_page == 2
        assert paragraphs[1].end_page == 2
    
    def test_mixed_blocks_with_header(self, builder, page_dimensions):
        """Test paragraphs with headers/footers present"""
        fragments = [
            # Header
            TextFragment("Book Title", 250, 10, 100, 8, font_size=8, page_num=1),
            # Title
            TextFragment("CHAPTER ONE", 200, 50, 195, 20, font_size=20, page_num=1),
            # Body
            TextFragment("Body text line 1", 72, 150, 450, 12, font_size=11, page_num=1),
            TextFragment("Body text line 2", 72, 165, 450, 12, font_size=11, page_num=1),
            # Page number
            TextFragment("1", 290, 820, 15, 10, font_size=9, page_num=1),
        ]
        
        paragraphs = builder.build_paragraphs(fragments, page_dimensions)
        
        # Headers and page numbers should be excluded
        texts = [p.text for p in paragraphs]
        assert "Book Title" not in texts
        assert "1" not in texts
        
        # Title and body should remain
        assert "CHAPTER ONE" in texts
        assert any("Body text" in t for t in texts)
    
    def test_get_paragraphs_by_type(self, builder, multi_paragraph_fragments, page_dimensions):
        """Test filtering paragraphs by type"""
        paragraphs = builder.build_paragraphs(multi_paragraph_fragments, page_dimensions)
        
        titles = builder.get_paragraphs_by_type(paragraphs, BlockType.TITLE)
        subtitles = builder.get_paragraphs_by_type(paragraphs, BlockType.SUBTITLE)
        body = builder.get_paragraphs_by_type(paragraphs, BlockType.BODY)
        
        assert len(titles) == 1
        assert len(subtitles) == 1
        assert len(body) == 2

