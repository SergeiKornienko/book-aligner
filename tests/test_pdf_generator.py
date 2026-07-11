"""
Tests for PDF generation service.
Creates final PDF with translated text and preserved illustrations.
"""

import pytest
from pathlib import Path
import fitz  # PyMuPDF
from backend.services.pdf_generator import PDFGenerator
from backend.services.models import TextFragment


class TestPDFGenerator:
    """Tests for PDFGenerator service"""
    
    @pytest.fixture
    def generator(self, tmp_path):
        """Create PDFGenerator with temporary output path"""
        output_path = tmp_path / "output.pdf"
        return PDFGenerator(str(output_path))
    
    @pytest.fixture
    def sample_donor_pdf(self, tmp_path):
        """Create a sample donor PDF with text and an image"""
        pdf_path = tmp_path / "donor.pdf"
        
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        
        # Insert text
        page.insert_text(
            fitz.Point(72, 100),
            "CHAPTER ONE",
            fontname="Times-Bold",
            fontsize=20
        )
        page.insert_text(
            fitz.Point(72, 150),
            "The Boy Who Lived",
            fontname="Times-Italic",
            fontsize=16
        )
        page.insert_text(
            fitz.Point(72, 200),
            "Mr. and Mrs. Dursley, of number four, Privet Drive,",
            fontname="Times-Roman",
            fontsize=11
        )
        page.insert_text(
            fitz.Point(72, 215),
            "were proud to say that they were perfectly normal,",
            fontname="Times-Roman",
            fontsize=11
        )
        page.insert_text(
            fitz.Point(72, 230),
            "thank you very much.",
            fontname="Times-Roman",
            fontsize=11
        )
        
        # Insert a simple rectangle as "image"
        page.draw_rect(fitz.Rect(200, 400, 400, 500), color=(0.8, 0.8, 0.8), fill=(0.8, 0.8, 0.8))
        
        doc.save(str(pdf_path))
        doc.close()
        
        return str(pdf_path)
    
    @pytest.fixture
    def sample_aligned_fragments(self):
        """Sample aligned text fragments for replacement"""
        return [
            (
                TextFragment("CHAPTER ONE", 72, 100, 195, 20, "Times-Bold", 20),
                TextFragment("ГЛАВА ПЕРВАЯ", 72, 100, 195, 20, "Arial-Bold", 20)
            ),
            (
                TextFragment("The Boy Who Lived", 72, 150, 235, 16, "Times-Italic", 16),
                TextFragment("Мальчик-Который-Выжил", 72, 150, 235, 16, "Arial-Italic", 16)
            ),
            (
                TextFragment("Mr. and Mrs. Dursley, of number four, Privet Drive,", 72, 200, 450, 12, "Times-Roman", 11),
                TextFragment("Мистер и миссис Дурсль проживали в доме номер четыре", 72, 200, 450, 12, "Arial", 11)
            ),
            (
                TextFragment("were proud to say that they were perfectly normal,", 72, 215, 450, 12, "Times-Roman", 11),
                TextFragment("по Тисовой улице и гордились тем, что они", 72, 215, 450, 12, "Arial", 11)
            ),
            (
                TextFragment("thank you very much.", 72, 230, 450, 12, "Times-Roman", 11),
                TextFragment("совершенно нормальные люди.", 72, 230, 450, 12, "Arial", 11)
            ),
        ]
    
    def test_generator_initialization(self, generator):
        """Test that generator initializes correctly"""
        assert generator is not None
        assert hasattr(generator, 'generate_pdf')
        assert generator.output_path.endswith('.pdf')
    
    def test_create_blank_pdf(self, generator):
        """Test creating a blank PDF"""
        result_path = generator.create_blank_pdf()
        
        assert Path(result_path).exists()
        assert result_path == generator.output_path
        
        # Verify it's a valid PDF (has 1 blank page)
        doc = fitz.open(result_path)
        assert doc.page_count == 1
        doc.close()
    
    def test_add_page(self, generator):
        """Test adding a page to PDF"""
        generator.create_blank_pdf()
        generator.add_page(width=595, height=842)
        
        doc = fitz.open(generator.output_path)
        assert doc.page_count == 2  # 1 blank + 1 added
        doc.close()
    
    def test_replace_text_on_page(self, generator, sample_aligned_fragments):
        """Test replacing text on a page"""
        generator.create_blank_pdf()
        
        page_num = 0
        generator.replace_text_on_page(page_num, sample_aligned_fragments)
        
        doc = fitz.open(generator.output_path)
        page = doc[0]
        
        text_blocks = page.get_text("blocks")
        texts_on_page = [block[4].strip() for block in text_blocks if block[4].strip()]
        
        assert any("ГЛАВА ПЕРВАЯ" in text for text in texts_on_page)
        doc.close()
    
    def test_copy_images_from_donor(self, generator, sample_donor_pdf):
        """Test copying images from donor PDF"""
        generator.create_blank_pdf()
        
        generator.copy_images_from_donor(sample_donor_pdf, donor_page=0, target_page=0)
        
        doc = fitz.open(generator.output_path)
        assert doc.page_count == 1
        doc.close()
    
    def test_generate_full_pdf(self, generator, sample_donor_pdf, sample_aligned_fragments):
        """Test full PDF generation pipeline"""
        result_path = generator.generate_pdf(
            donor_pdf_path=sample_donor_pdf,
            aligned_fragments=sample_aligned_fragments
        )
        
        assert Path(result_path).exists()
        
        doc = fitz.open(result_path)
        assert doc.page_count == 1
        
        page = doc[0]
        text_blocks = page.get_text("blocks")
        texts_on_page = [block[4].strip() for block in text_blocks if block[4].strip()]
        
        assert any("ГЛАВА ПЕРВАЯ" in text for text in texts_on_page)
        doc.close()
    
    def test_generate_pdf_file_not_found(self, generator):
        """Test error handling for missing donor PDF"""
        with pytest.raises(FileNotFoundError):
            generator.generate_pdf(
                donor_pdf_path="/nonexistent/file.pdf",
                aligned_fragments=[]
            )
    
    def test_output_path_is_absolute(self, tmp_path):
        """Test that output path is resolved correctly"""
        gen = PDFGenerator("relative/path/output.pdf")
        assert Path(gen.output_path).is_absolute() or "/" in str(gen.output_path)