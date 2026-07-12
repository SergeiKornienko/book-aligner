"""
Tests for text alignment service.
Matches text fragments between donor and sample PDFs.
"""

import pytest
from backend.services.alignment import TextAligner
from backend.services.models import TextFragment
from backend.services.block_classifier import BlockType


class TestTextFragment:
    """Tests for TextFragment data class"""
    
    def test_create_fragment(self):
        """Test creating a text fragment"""
        fragment = TextFragment(
            text="Hello world",
            x=10.0,
            y=20.0,
            width=100.0,
            height=12.0,
            font_name="Times-Roman",
            font_size=12.0
        )
        assert fragment.text == "Hello world"
        assert fragment.x == 10.0
        assert fragment.y == 20.0
        assert fragment.width == 100.0
        assert fragment.height == 12.0
    
    def test_fragment_equality(self):
        """Test that fragments with same content are equal"""
        f1 = TextFragment("Test", 0, 0, 10, 10)
        f2 = TextFragment("Test", 0, 0, 10, 10)
        assert f1 == f2
    
    def test_fragment_inequality(self):
        """Test that fragments with different content are not equal"""
        f1 = TextFragment("Test1", 0, 0, 10, 10)
        f2 = TextFragment("Test2", 0, 0, 10, 10)
        assert f1 != f2
    
    def test_page_num_default(self):
        """Test that page_num defaults to 0"""
        fragment = TextFragment("Test", 0, 0, 10, 10)
        assert fragment.page_num == 0


class TestTextAligner:
    """Tests for TextAligner service"""
    
    @pytest.fixture
    def sample_fragments(self):
        """Create sample text fragments for testing"""
        donor = [
            TextFragment("The boy who lived", 10.0, 10.0, 100.0, 12.0, "Times-Roman", 12.0),
            TextFragment("Mr. and Mrs. Dursley", 10.0, 30.0, 120.0, 12.0, "Times-Roman", 12.0),
            TextFragment("were proud to say", 10.0, 50.0, 110.0, 12.0, "Times-Roman", 12.0),
            TextFragment("that they were perfectly normal", 10.0, 70.0, 180.0, 12.0, "Times-Roman", 12.0),
        ]
        
        sample = [
            TextFragment("Мальчик-который-выжил", 15.0, 15.0, 130.0, 14.0, "Arial", 12.0),
            TextFragment("Мистер и миссис Дурсль", 15.0, 35.0, 140.0, 14.0, "Arial", 12.0),
            TextFragment("гордились тем, что", 15.0, 55.0, 120.0, 14.0, "Arial", 12.0),
            TextFragment("они совершенно нормальные люди", 15.0, 75.0, 200.0, 14.0, "Arial", 12.0),
        ]
        
        return donor, sample
    
    @pytest.fixture
    def aligner(self):
        """Create TextAligner instance"""
        return TextAligner()
    
    def test_initialization(self, aligner):
        """Test that TextAligner initializes correctly"""
        assert aligner is not None
        assert hasattr(aligner, 'align')
    
    def test_align_basic(self, aligner, sample_fragments):
        """Test basic alignment of matched fragments"""
        donor, sample = sample_fragments
        
        # Use simple sequential matching (without AI for now)
        aligned = aligner.align_sequential(donor, sample)
        
        assert len(aligned) == len(donor)
        assert len(aligned) == len(sample)
        
        # Check that positions are preserved from donor
        for i, (donor_frag, sample_frag) in enumerate(aligned):
            assert sample_frag.x == donor_frag.x
            assert sample_frag.y == donor_frag.y
            assert sample_frag.width == donor_frag.width
            assert sample_frag.height == donor_frag.height
    
    def test_align_different_lengths(self, aligner):
        """Test alignment when donor and sample have different lengths"""
        donor = [
            TextFragment("Short", 10, 10, 50, 10),
        ]
        sample = [
            TextFragment("Короткий", 10, 10, 60, 10),
            TextFragment("Длинный", 10, 30, 60, 10),
        ]
        
        # Should handle gracefully
        aligned = aligner.align_sequential(donor, sample)
        assert len(aligned) == min(len(donor), len(sample))
    
    def test_align_empty_lists(self, aligner):
        """Test alignment with empty fragments"""
        aligned = aligner.align_sequential([], [])
        assert aligned == []
    
    def test_prepare_alignment_data(self, aligner, sample_fragments):
        """Test preparation of alignment data"""
        donor, sample = sample_fragments
        
        donor_texts, sample_texts = aligner.prepare_texts(donor, sample)
        
        assert len(donor_texts) == len(donor)
        assert len(sample_texts) == len(sample)
        assert donor_texts[0] == "The boy who lived"
        assert sample_texts[0] == "Мальчик-который-выжил"


class TestTextAlignerWithClassifier:
    """Tests for TextAligner with block classification support"""
    
    @pytest.fixture
    def aligner_with_classifier(self):
        """Create TextAligner with classifier enabled"""
        return TextAligner(use_ai=True, use_classifier=True)
    
    @pytest.fixture
    def mixed_donor_fragments(self):
        """Donor fragments with different block types"""
        return [
            # Title
            TextFragment("CHAPTER ONE", 200, 50, 195, 20, font_name="Times-Bold", font_size=20, page_num=1),
            # Subtitle
            TextFragment("THE BOY WHO LIVED", 180, 80, 235, 16, font_name="Times-Italic", font_size=16, page_num=1),
            # Body paragraph 1
            TextFragment("Mr. and Mrs. Dursley, of number four, Privet", 72, 150, 450, 12, font_size=11, page_num=1),
            TextFragment("Drive, were proud to say that they were perfectly", 72, 165, 450, 12, font_size=11, page_num=1),
            TextFragment("normal, thank you very much.", 72, 180, 450, 12, font_size=11, page_num=1),
            # Body paragraph 2
            TextFragment("They were the last people you'd expect to be", 72, 220, 450, 12, font_size=11, page_num=1),
            TextFragment("involved in anything strange or mysterious.", 72, 235, 450, 12, font_size=11, page_num=1),
            # Header
            TextFragment("Harry Potter", 250, 10, 100, 8, font_size=8, page_num=1),
            # Page number
            TextFragment("1", 290, 820, 15, 10, font_size=9, page_num=1),
        ]
    
    @pytest.fixture
    def mixed_sample_fragments(self):
        """Sample (translation) fragments with different block types"""
        return [
            # Title
            TextFragment("ГЛАВА ПЕРВАЯ", 200, 50, 195, 20, font_name="Arial-Bold", font_size=20, page_num=1),
            # Subtitle
            TextFragment("МАЛЬЧИК-КОТОРЫЙ-ВЫЖИЛ", 160, 80, 275, 16, font_name="Arial-Italic", font_size=16, page_num=1),
            # Body paragraph 1
            TextFragment("Мистер и миссис Дурсль проживали в доме", 72, 150, 450, 12, font_size=11, page_num=1),
            TextFragment("номер четыре по Тисовой улице и гордились", 72, 165, 450, 12, font_size=11, page_num=1),
            TextFragment("тем, что они совершенно нормальные люди.", 72, 180, 450, 12, font_size=11, page_num=1),
            # Body paragraph 2
            TextFragment("Они были последними людьми, от которых", 72, 220, 450, 12, font_size=11, page_num=1),
            TextFragment("можно было ожидать чего-то странного.", 72, 235, 450, 12, font_size=11, page_num=1),
            # Header
            TextFragment("Гарри Поттер", 250, 10, 100, 8, font_size=8, page_num=1),
            # Page number
            TextFragment("1", 290, 820, 15, 10, font_size=9, page_num=1),
        ]
    
    def test_classifier_enabled(self, aligner_with_classifier):
        """Test that classifier mode is enabled"""
        assert aligner_with_classifier.use_classifier == True
    
    def test_classify_and_filter(self, aligner_with_classifier, mixed_donor_fragments):
        """Test classification and filtering of fragments"""
        page_dims = {"width": 595, "height": 842}
        
        classified = aligner_with_classifier.classify_fragments(
            mixed_donor_fragments, page_dims
        )
        
        # Should classify all fragments
        assert len(classified) == len(mixed_donor_fragments)
        
        # Check that types are assigned
        for fragment, block_type, confidence in classified:
            assert block_type is not None
            assert confidence > 0
    
    def test_filter_ignored_blocks(self, aligner_with_classifier, mixed_donor_fragments):
        """Test that headers and page numbers are filtered out"""
        page_dims = {"width": 595, "height": 842}
        
        body_fragments = aligner_with_classifier.filter_body_fragments(
            mixed_donor_fragments, page_dims
        )
        
        # Should exclude header and page number (2 fragments)
        # But include title, subtitle, and body text (7 fragments)
        assert len(body_fragments) == 7
        
        # Check that no header/page number remains
        texts = [f.text for f in body_fragments]
        assert "Harry Potter" not in texts  # Header removed
        assert "1" not in texts  # Page number removed
        
        # Body text should remain (check partial match since fragments contain full lines)
        assert any("Mr. and Mrs. Dursley" in text for text in texts)
        assert any("thank you very much" in text for text in texts)
    
    def test_align_with_classifier(self, aligner_with_classifier, 
                                    mixed_donor_fragments, mixed_sample_fragments):
        """Test full alignment with classifier filtering"""
        page_dims = {"width": 595, "height": 842}
        
        aligned = aligner_with_classifier.align_with_classifier(
            mixed_donor_fragments, 
            mixed_sample_fragments, 
            page_dims
        )
        
        # Should have aligned pairs (excluding headers/page numbers)
        assert len(aligned) > 0
        
        # Check that aligned fragments preserve donor positions
        for donor_frag, sample_frag in aligned:
            assert sample_frag.x == donor_frag.x
            assert sample_frag.y == donor_frag.y
    
    def test_extract_special_blocks(self, aligner_with_classifier, mixed_donor_fragments):
        """Test extraction of special blocks (titles, subtitles)"""
        page_dims = {"width": 595, "height": 842}
        
        special_blocks = aligner_with_classifier.extract_special_blocks(
            mixed_donor_fragments, page_dims
        )
        
        # Should have title and subtitle
        assert len(special_blocks) == 2
        
        types = [block_type for _, block_type in special_blocks]
        assert BlockType.TITLE in types
        assert BlockType.SUBTITLE in types