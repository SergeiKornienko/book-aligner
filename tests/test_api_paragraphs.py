"""
Tests for paragraph-based API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.services.models import TextFragment
from backend.services.paragraph_builder import ParagraphBuilder, Paragraph
from backend.services.block_classifier import BlockType


client = TestClient(app)


class TestParagraphAPI:
    """Tests for paragraph-related API endpoints"""
    
    @pytest.fixture
    def sample_fragments_json(self):
        """Sample fragments as JSON-serializable dicts"""
        return [
            {
                "text": "CHAPTER ONE",
                "x": 200.0, "y": 50.0, "width": 195.0, "height": 20.0,
                "font_name": "Times-Bold", "font_size": 20.0, "page_num": 0
            },
            {
                "text": "THE BOY WHO LIVED",
                "x": 180.0, "y": 80.0, "width": 235.0, "height": 16.0,
                "font_name": "Times-Italic", "font_size": 16.0, "page_num": 0
            },
            {
                "text": "Mr. and Mrs. Dursley, of number four, Privet",
                "x": 72.0, "y": 150.0, "width": 450.0, "height": 12.0,
                "font_name": "Times-Roman", "font_size": 11.0, "page_num": 0
            },
            {
                "text": "Drive, were proud to say that they were perfectly",
                "x": 72.0, "y": 165.0, "width": 450.0, "height": 12.0,
                "font_name": "Times-Roman", "font_size": 11.0, "page_num": 0
            },
            {
                "text": "normal, thank you very much.",
                "x": 72.0, "y": 180.0, "width": 450.0, "height": 12.0,
                "font_name": "Times-Roman", "font_size": 11.0, "page_num": 0
            },
            {
                "text": "They were the last people you'd expect to be",
                "x": 72.0, "y": 240.0, "width": 450.0, "height": 12.0,
                "font_name": "Times-Roman", "font_size": 11.0, "page_num": 0
            },
            {
                "text": "involved in anything strange or mysterious.",
                "x": 72.0, "y": 255.0, "width": 450.0, "height": 12.0,
                "font_name": "Times-Roman", "font_size": 11.0, "page_num": 0
            },
        ]
    
    @pytest.fixture
    def page_dimensions(self):
        """Standard page dimensions"""
        return {"width": 595, "height": 842}
    
    def test_build_paragraphs_endpoint_exists(self):
        """Test that paragraphs endpoint is registered"""
        response = client.post(
            "/paragraphs/build",
            json={
                "fragments": [],
                "page_dimensions": {"width": 595, "height": 842}
            }
        )
        assert response.status_code != 404
    
    def test_build_paragraphs_empty(self):
        """Test building paragraphs with empty fragments"""
        response = client.post(
            "/paragraphs/build",
            json={
                "fragments": [],
                "page_dimensions": {"width": 595, "height": 842}
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "paragraphs" in data
        assert len(data["paragraphs"]) == 0
        assert data["total_paragraphs"] == 0
    
    def test_build_paragraphs_basic(self, sample_fragments_json, page_dimensions):
        """Test building paragraphs from fragments"""
        response = client.post(
            "/paragraphs/build",
            json={
                "fragments": sample_fragments_json,
                "page_dimensions": page_dimensions
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "paragraphs" in data
        assert data["total_paragraphs"] > 0
        
        # Should have multiple paragraphs
        paragraphs = data["paragraphs"]
        assert len(paragraphs) >= 3  # Title, subtitle, at least 1 body
    
    def test_build_paragraphs_structure(self, sample_fragments_json, page_dimensions):
        """Test structure of returned paragraphs"""
        response = client.post(
            "/paragraphs/build",
            json={
                "fragments": sample_fragments_json,
                "page_dimensions": page_dimensions
            }
        )
        
        data = response.json()
        paragraphs = data["paragraphs"]
        
        for para in paragraphs:
            # Each paragraph should have required fields
            assert "text" in para
            assert "block_type" in para
            assert "fragment_count" in para
            assert "start_page" in para
            assert "end_page" in para
            assert "spans_pages" in para
    
    def test_build_paragraphs_types(self, sample_fragments_json, page_dimensions):
        """Test that paragraphs have correct types"""
        response = client.post(
            "/paragraphs/build",
            json={
                "fragments": sample_fragments_json,
                "page_dimensions": page_dimensions
            }
        )
        
        data = response.json()
        paragraphs = data["paragraphs"]
        
        types = [p["block_type"] for p in paragraphs]
        assert "TITLE" in types
        assert "SUBTITLE" in types
        assert "BODY" in types
    
    def test_align_paragraphs_endpoint_exists(self):
        """Test that paragraph alignment endpoint is registered"""
        response = client.post(
            "/paragraphs/align",
            json={
                "donor_paragraphs": [],
                "sample_paragraphs": [],
                "use_ai": False
            }
        )
        assert response.status_code != 404
    
    def test_align_paragraphs_empty(self):
        """Test aligning empty paragraphs"""
        response = client.post(
            "/paragraphs/align",
            json={
                "donor_paragraphs": [],
                "sample_paragraphs": [],
                "use_ai": False
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "aligned_paragraphs" in data
        assert len(data["aligned_paragraphs"]) == 0
    
    def test_align_paragraphs_basic(self):
        """Test basic paragraph alignment"""
        donor_paragraphs = [
            {
                "text": "CHAPTER ONE",
                "block_type": "TITLE",
                "fragment_count": 1,
                "start_page": 0, "end_page": 0,
                "spans_pages": False
            },
            {
                "text": "Mr. and Mrs. Dursley were proud to say they were perfectly normal.",
                "block_type": "BODY",
                "fragment_count": 3,
                "start_page": 0, "end_page": 0,
                "spans_pages": False
            }
        ]
        
        sample_paragraphs = [
            {
                "text": "ГЛАВА ПЕРВАЯ",
                "block_type": "TITLE",
                "fragment_count": 1,
                "start_page": 0, "end_page": 0,
                "spans_pages": False
            },
            {
                "text": "Мистер и миссис Дурсль гордились тем, что они совершенно нормальные люди.",
                "block_type": "BODY",
                "fragment_count": 3,
                "start_page": 0, "end_page": 0,
                "spans_pages": False
            }
        ]
        
        response = client.post(
            "/paragraphs/align",
            json={
                "donor_paragraphs": donor_paragraphs,
                "sample_paragraphs": sample_paragraphs,
                "use_ai": True
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "aligned_paragraphs" in data
        assert data["total_pairs"] == 2