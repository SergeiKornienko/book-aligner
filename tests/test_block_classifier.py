"""
Tests for block classification service.
Identifies different types of text blocks in PDF pages.
"""

import pytest
from backend.services.alignment import TextAligner
from backend.services.models import TextFragment
from backend.services.block_classifier import BlockClassifier, BlockType


class TestBlockType:
    """Tests for BlockType enum"""
    
    def test_block_types_exist(self):
        """Test that all required block types are defined"""
        assert hasattr(BlockType, 'TITLE')
        assert hasattr(BlockType, 'SUBTITLE')
        assert hasattr(BlockType, 'BODY')
        assert hasattr(BlockType, 'CAPTION')
        assert hasattr(BlockType, 'HEADER')
        assert hasattr(BlockType, 'FOOTER')
        assert hasattr(BlockType, 'PAGE_NUMBER')
        assert hasattr(BlockType, 'DROPCAP')
        assert hasattr(BlockType, 'UNKNOWN')


class TestBlockClassifier:
    """Tests for BlockClassifier service"""
    
    @pytest.fixture
    def classifier(self):
        """Create BlockClassifier instance"""
        return BlockClassifier()
    
    @pytest.fixture
    def page_dimensions(self):
        """Sample page dimensions"""
        return {"width": 595, "height": 842}  # A4 in points
    
    def test_classify_title(self, classifier, page_dimensions):
        """Test classification of chapter title"""
        fragments = [
            TextFragment("CHAPTER ONE", 
                        x=200, y=100, width=195, height=24, 
                        font_name="Times-Bold", font_size=24)
        ]
        
        block_type, confidence = classifier.classify_block(
            fragments, page_dimensions
        )
        
        assert block_type == BlockType.TITLE
        assert confidence > 0.5
    
    def test_classify_subtitle(self, classifier, page_dimensions):
        """Test classification of subtitle"""
        fragments = [
            TextFragment("The Boy Who Lived", 
                        x=200, y=140, width=195, height=18, 
                        font_name="Times-Italic", font_size=16)
        ]
        
        block_type, confidence = classifier.classify_block(
            fragments, page_dimensions
        )
        
        assert block_type == BlockType.SUBTITLE
    
    def test_classify_body_text(self, classifier, page_dimensions):
        """Test classification of regular body text"""
        fragments = [
            TextFragment("Mr. and Mrs. Dursley, of number four, Privet Drive,", 
                        x=72, y=200, width=450, height=12, 
                        font_name="Times-Roman", font_size=11)
        ]
        
        block_type, confidence = classifier.classify_block(
            fragments, page_dimensions
        )
        
        assert block_type == BlockType.BODY
    
    def test_classify_page_number(self, classifier, page_dimensions):
        """Test classification of page number"""
        fragments = [
            TextFragment("42", 
                        x=290, y=800, width=15, height=10, 
                        font_name="Times-Roman", font_size=9)
        ]
        
        block_type, confidence = classifier.classify_block(
            fragments, page_dimensions
        )
        
        assert block_type == BlockType.PAGE_NUMBER
    
    def test_classify_header(self, classifier, page_dimensions):
        """Test classification of page header"""
        fragments = [
            TextFragment("Harry Potter and the Philosopher's Stone", 
                        x=200, y=20, width=195, height=10, 
                        font_name="Times-Italic", font_size=9)
        ]
        
        block_type, confidence = classifier.classify_block(
            fragments, page_dimensions
        )
        
        assert block_type == BlockType.HEADER
    
    def test_classify_footer(self, classifier, page_dimensions):
        """Test classification of page footer"""
        fragments = [
            TextFragment("Copyright © 1997 J.K. Rowling", 
                        x=72, y=810, width=200, height=8, 
                        font_name="Times-Roman", font_size=7)
        ]
        
        block_type, confidence = classifier.classify_block(
            fragments, page_dimensions
        )
        
        assert block_type == BlockType.FOOTER
    
    def test_classify_caption_near_image(self, classifier, page_dimensions):
        """Test classification of image caption"""
        # Captions are typically below images, centered, in smaller font
        fragments = [
            TextFragment("Fig. 1: Hogwarts Castle at sunset", 
                        x=200, y=600, width=195, height=10, 
                        font_name="Times-Italic", font_size=9)
        ]
        
        block_type, confidence = classifier.classify_block(
            fragments, page_dimensions
        )
        
        assert block_type == BlockType.CAPTION
    
    def test_classify_dropcap(self, classifier, page_dimensions):
        """Test classification of drop cap (first letter)"""
        fragments = [
            TextFragment("M", 
                        x=72, y=200, width=30, height=36, 
                        font_name="Times-Bold", font_size=36)
        ]
        
        block_type, confidence = classifier.classify_block(
            fragments, page_dimensions
        )
        
        assert block_type == BlockType.DROPCAP
    
    def test_classify_empty_block(self, classifier, page_dimensions):
        """Test classification with empty fragment list"""
        block_type, confidence = classifier.classify_block([], page_dimensions)
        assert block_type == BlockType.UNKNOWN
        assert confidence == 0.0
    
    def test_classify_multiline_body(self, classifier, page_dimensions):
        """Test that multiline body text is classified as BODY"""
        fragments = [
            TextFragment("First line of body text", 
                        x=72, y=200, width=450, height=12, font_size=11),
            TextFragment("Second line of body text", 
                        x=72, y=215, width=450, height=12, font_size=11),
            TextFragment("Third line of body text", 
                        x=72, y=230, width=450, height=12, font_size=11),
        ]
        
        block_type, confidence = classifier.classify_block(
            fragments, page_dimensions
        )
        
        assert block_type == BlockType.BODY
        assert confidence > 0.7  # Higher confidence for multiple lines