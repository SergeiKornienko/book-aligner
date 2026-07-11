"""
Tests for the end-to-end processing API endpoint.
Accepts two PDFs and returns the final aligned PDF.
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import fitz
from backend.main import app


client = TestClient(app)


class TestProcessAPI:
    """Tests for the /process endpoint"""
    
    @pytest.fixture
    def sample_donor_pdf(self, tmp_path):
        """Create a sample donor PDF with English text"""
        pdf_path = tmp_path / "donor.pdf"
        
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        
        page.insert_text(fitz.Point(200, 50), "CHAPTER ONE", fontname="Times-Bold", fontsize=20)
        page.insert_text(fitz.Point(180, 80), "THE BOY WHO LIVED", fontname="Times-Italic", fontsize=16)
        page.insert_text(fitz.Point(72, 150), "Mr. and Mrs. Dursley, of number four, Privet Drive,", fontname="Times-Roman", fontsize=11)
        page.insert_text(fitz.Point(72, 165), "were proud to say that they were perfectly normal,", fontname="Times-Roman", fontsize=11)
        page.insert_text(fitz.Point(72, 180), "thank you very much.", fontname="Times-Roman", fontsize=11)
        page.insert_text(fitz.Point(72, 240), "They were the last people you'd expect to be", fontname="Times-Roman", fontsize=11)
        page.insert_text(fitz.Point(72, 255), "involved in anything strange or mysterious.", fontname="Times-Roman", fontsize=11)
        
        # Add a rectangle as fake "illustration"
        page.draw_rect(fitz.Rect(200, 400, 400, 500), color=(0.8, 0.8, 0.8), fill=(0.8, 0.8, 0.8))
        
        doc.save(str(pdf_path))
        doc.close()
        
        return str(pdf_path)
    
    @pytest.fixture
    def sample_sample_pdf(self, tmp_path):
        """Create a sample Russian translation PDF"""
        pdf_path = tmp_path / "sample.pdf"
        
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        
        # Use 'china-s' font for Cyrillic support
        page.insert_text(fitz.Point(200, 50), "ГЛАВА ПЕРВАЯ", fontname="china-s", fontsize=20)
        page.insert_text(fitz.Point(160, 80), "МАЛЬЧИК-КОТОРЫЙ-ВЫЖИЛ", fontname="china-s", fontsize=16)
        page.insert_text(fitz.Point(72, 150), "Мистер и миссис Дурсль проживали в доме", fontname="china-s", fontsize=11)
        page.insert_text(fitz.Point(72, 165), "номер четыре по Тисовой улице и гордились", fontname="china-s", fontsize=11)
        page.insert_text(fitz.Point(72, 180), "тем, что они совершенно нормальные люди.", fontname="china-s", fontsize=11)
        page.insert_text(fitz.Point(72, 240), "Они были последними людьми, от которых", fontname="china-s", fontsize=11)
        page.insert_text(fitz.Point(72, 255), "можно было ожидать чего-то странного.", fontname="china-s", fontsize=11)
        
        doc.save(str(pdf_path))
        doc.close()
        
        return str(pdf_path)
    
    def test_process_endpoint_exists(self):
        """Test that /process endpoint is registered"""
        response = client.post("/process")
        # Should not be 404 Not Found (even if it fails validation)
        assert response.status_code != 404
    
    def test_process_missing_files(self):
        """Test that endpoint requires both PDF files"""
        response = client.post("/process")
        assert response.status_code == 422  # Validation error
    
    def test_process_only_donor(self, sample_donor_pdf):
        """Test with only donor file"""
        with open(sample_donor_pdf, "rb") as f:
            response = client.post(
                "/process",
                files={"donor": ("donor.pdf", f, "application/pdf")}
            )
        assert response.status_code == 422  # Missing sample
    
    def test_process_wrong_format(self, tmp_path):
        """Test that non-PDF files are rejected"""
        txt_path = tmp_path / "test.txt"
        txt_path.write_text("Not a PDF")
        
        with open(txt_path, "rb") as f:
            response = client.post(
                "/process",
                files={
                    "donor": ("test.txt", f, "text/plain"),
                    "sample": ("test.txt", f, "text/plain")
                }
            )
        assert response.status_code == 400  # Bad request
    
    def test_process_basic(self, sample_donor_pdf, sample_sample_pdf):
        """Test full processing pipeline"""
        with open(sample_donor_pdf, "rb") as donor_f, open(sample_sample_pdf, "rb") as sample_f:
            response = client.post(
                "/process",
                files={
                    "donor": ("donor.pdf", donor_f, "application/pdf"),
                    "sample": ("sample.pdf", sample_f, "application/pdf")
                }
            )
        
        assert response.status_code == 200
        
        # Response should be a PDF file
        assert response.headers["content-type"] == "application/pdf"
        
        # Verify it's a valid PDF
        content = response.content
        assert len(content) > 0
        assert content[:4] == b"%PDF"  # PDF magic bytes
    
    def test_process_result_has_russian_text(self, sample_donor_pdf, sample_sample_pdf, tmp_path):
        """Test that result PDF contains Russian text"""
        with open(sample_donor_pdf, "rb") as donor_f, open(sample_sample_pdf, "rb") as sample_f:
            response = client.post(
                "/process",
                files={
                    "donor": ("donor.pdf", donor_f, "application/pdf"),
                    "sample": ("sample.pdf", sample_f, "application/pdf")
                }
            )
        
        result_path = tmp_path / "result.pdf"
        result_path.write_bytes(response.content)
        
        doc = fitz.open(str(result_path))
        assert doc.page_count == 1
        
        page = doc[0]
        text = page.get_text()
        
        # Debug output
        print(f"\n=== RESULT PDF TEXT ===")
        print(repr(text))
        print(f"=== END ===")
        
        assert "ГЛАВА ПЕРВАЯ" in text or "Глава" in text
        doc.close()

    def test_process_preserves_pages(self, sample_donor_pdf, sample_sample_pdf, tmp_path):
        """Test that page count is preserved"""
        with open(sample_donor_pdf, "rb") as donor_f, open(sample_sample_pdf, "rb") as sample_f:
            response = client.post(
                "/process",
                files={
                    "donor": ("donor.pdf", donor_f, "application/pdf"),
                    "sample": ("sample.pdf", sample_f, "application/pdf")
                }
            )
        
        result_path = tmp_path / "result.pdf"
        result_path.write_bytes(response.content)
        
        donor_doc = fitz.open(sample_donor_pdf)
        result_doc = fitz.open(str(result_path))
        
        assert result_doc.page_count == donor_doc.page_count
        
        donor_doc.close()
        result_doc.close()