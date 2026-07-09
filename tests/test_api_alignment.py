"""
Tests for alignment API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import json
from backend.main import app
from backend.services.models import TextFragment
from backend.services.text_alignment import TextAligner


client = TestClient(app)


class TestAlignmentAPI:
    """Tests for alignment-related API endpoints"""
    
    @pytest.fixture
    def mock_aligner(self):
        """Mock TextAligner for testing"""
        with patch('backend.routes.TextAligner') as mock:
            yield mock
    
    @pytest.fixture
    def sample_donor_fragments(self):
        """Sample donor fragments"""
        return [
            {
                "text": "Harry Potter",
                "x": 10.0,
                "y": 10.0,
                "width": 100.0,
                "height": 12.0,
                "font_name": "Times-Roman",
                "font_size": 12.0
            },
            {
                "text": "Ron Weasley",
                "x": 10.0,
                "y": 30.0,
                "width": 120.0,
                "height": 12.0,
                "font_name": "Times-Roman",
                "font_size": 12.0
            }
        ]
    
    @pytest.fixture
    def sample_fragments(self):
        """Sample translation fragments"""
        return [
            {
                "text": "Гарри Поттер",
                "x": 15.0,
                "y": 15.0,
                "width": 130.0,
                "height": 14.0,
                "font_name": "Arial",
                "font_size": 12.0
            },
            {
                "text": "Рон Уизли",
                "x": 15.0,
                "y": 35.0,
                "width": 140.0,
                "height": 14.0,
                "font_name": "Arial",
                "font_size": 12.0
            }
        ]
    
    def test_align_endpoint_exists(self):
        """Test that alignment endpoint is registered"""
        response = client.post(
            "/align",
            json={
                "donor_fragments": [],
                "sample_fragments": []
            }
        )
        # Should not be 404 Not Found
        assert response.status_code != 404
    
    def test_align_empty_fragments(self):
        """Test alignment with empty fragment lists"""
        response = client.post(
            "/align",
            json={
                "donor_fragments": [],
                "sample_fragments": []
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "aligned_fragments" in data
        assert len(data["aligned_fragments"]) == 0
    
    def test_align_basic(self, sample_donor_fragments, sample_fragments):
        """Test basic alignment through API"""
        response = client.post(
            "/align",
            json={
                "donor_fragments": sample_donor_fragments,
                "sample_fragments": sample_fragments
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "aligned_fragments" in data
        assert len(data["aligned_fragments"]) == 2
        
        # Check that aligned fragments preserve donor positions
        aligned = data["aligned_fragments"]
        
        # Check first fragment position
        assert aligned[0]["x"] == 10.0
        assert aligned[0]["y"] == 10.0
        
        # Check second fragment position
        assert aligned[1]["x"] == 10.0
        assert aligned[1]["y"] == 30.0
        
        # Check that both expected texts are present (order may vary with AI)
        aligned_texts = [f["text"] for f in aligned]
        assert "Гарри Поттер" in aligned_texts
        assert "Рон Уизли" in aligned_texts
    
    def test_align_different_lengths(self, sample_donor_fragments):
        """Test alignment when lists have different lengths"""
        sample_fragments = [
            {
                "text": "Мальчик-который-выжил",
                "x": 15.0,
                "y": 15.0,
                "width": 130.0,
                "height": 14.0
            },
            {
                "text": "Мистер и миссис Дурсль",
                "x": 15.0,
                "y": 35.0,
                "width": 140.0,
                "height": 14.0
            },
            {
                "text": "Лишний фрагмент",
                "x": 15.0,
                "y": 55.0,
                "width": 100.0,
                "height": 14.0
            }
        ]
        
        response = client.post(
            "/align",
            json={
                "donor_fragments": sample_donor_fragments,
                "sample_fragments": sample_fragments
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        # Should align to the minimum length
        assert len(data["aligned_fragments"]) == 2
    
    def test_align_invalid_input(self):
        """Test alignment with invalid input"""
        response = client.post(
            "/align",
            json={
                "donor_fragments": "not_a_list",
                "sample_fragments": []
            }
        )
        assert response.status_code == 422  # Validation error
    
    def test_align_missing_fields(self, sample_donor_fragments):
        """Test alignment with missing required fields"""
        invalid_fragments = [
            {
                "text": "Missing fields"
                # Missing x, y, width, height
            }
        ]
        
        response = client.post(
            "/align",
            json={
                "donor_fragments": invalid_fragments,
                "sample_fragments": sample_donor_fragments
            }
        )
        
        # Should handle gracefully
        assert response.status_code in [200, 422, 500]
    
    def test_align_response_structure(self, sample_donor_fragments, sample_fragments):
        """Test the structure of alignment response"""
        response = client.post(
            "/align",
            json={
                "donor_fragments": sample_donor_fragments,
                "sample_fragments": sample_fragments
            }
        )
        
        data = response.json()
        
        # Check response structure
        assert "aligned_fragments" in data
        assert "total_pairs" in data
        assert data["total_pairs"] == len(data["aligned_fragments"])
        
        # Check each aligned fragment has required fields
        for fragment in data["aligned_fragments"]:
            assert "text" in fragment
            assert "x" in fragment
            assert "y" in fragment
            assert "width" in fragment
            assert "height" in fragment