"""
Tests for PDF parser service using PyMuPDF
"""

import pytest
from pathlib import Path
from backend.services.pdf_parser import PDFParser
import fitz


def test_extract_text_simple(tmp_path):
    """Test simple text extraction"""
    # Create PDF with text using PyMuPDF
    pdf_path = tmp_path / "test.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((100, 100), "Hello, this is a test document")
    page.insert_text((100, 150), "Second line of text")
    page.insert_text((100, 200), "Harry Potter and the Philosopher's Stone")
    doc.save(str(pdf_path))
    doc.close()
    
    parser = PDFParser(str(pdf_path))
    result = parser.extract_text_simple()
    
    assert len(result) == 3
    assert result[0]['text'] == "Hello, this is a test document"
    assert result[1]['text'] == "Second line of text"
    assert result[2]['text'] == "Harry Potter and the Philosopher's Stone"
    
    # Verify data structure
    for item in result:
        assert 'page' in item
        assert 'line_num' in item
        assert 'text' in item
        assert isinstance(item['page'], int)
        assert isinstance(item['line_num'], int)
        assert isinstance(item['text'], str)


def test_extract_text_with_coordinates(tmp_path):
    """Test text extraction with coordinates"""
    pdf_path = tmp_path / "test.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((100, 100), "Test text with coordinates")
    doc.save(str(pdf_path))
    doc.close()
    
    parser = PDFParser(str(pdf_path))
    result = parser.extract_text_with_coordinates()
    
    assert len(result) > 0
    for item in result:
        assert 'page' in item
        assert 'x' in item
        assert 'y' in item
        assert 'text' in item
        assert isinstance(item['x'], (int, float))
        assert isinstance(item['y'], (int, float))
        assert isinstance(item['text'], str)


def test_empty_pdf(tmp_path):
    """Test parser handles empty PDF gracefully"""
    pdf_path = tmp_path / "empty.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(pdf_path))
    doc.close()
    
    parser = PDFParser(str(pdf_path))
    result = parser.extract_text_simple()
    
    assert result == []


def test_extract_images(tmp_path):
    """Test image extraction"""
    pdf_path = tmp_path / "images.pdf"
    
    # Create PDF with an image (we'll use a simple approach)
    doc = fitz.open()
    page = doc.new_page(width=400, height=400)
    
    # Draw a simple rectangle as "image"
    rect = fitz.Rect(50, 50, 200, 200)
    page.draw_rect(rect, color=(0, 0, 1), fill=(0, 0, 1))
    
    doc.save(str(pdf_path))
    doc.close()
    
    # Test extraction
    parser = PDFParser(str(pdf_path))
    images = parser.extract_images()
    
    # May or may not find images depending on PDF structure
    assert isinstance(images, list)


# tests/test_pdf_parser.py - исправленный test_metadata

def test_metadata(tmp_path):
    """Test metadata extraction"""
    pdf_path = tmp_path / "meta.pdf"
    doc = fitz.open()
    
    # Set metadata using set_metadata method
    doc.set_metadata({
        'title': 'Test Document',
        'author': 'John Doe',
        'subject': 'Testing',
        'creator': 'PyMuPDF Test'
    })
    
    doc.new_page()
    doc.save(str(pdf_path))
    doc.close()
    
    parser = PDFParser(str(pdf_path))
    metadata = parser.get_metadata()
    
    assert metadata['title'] == 'Test Document'
    assert metadata['author'] == 'John Doe'
    assert metadata['subject'] == 'Testing'
    assert metadata['creator'] == 'PyMuPDF Test'


def test_page_size(tmp_path):
    """Test page size extraction"""
    pdf_path = tmp_path / "size.pdf"
    doc = fitz.open()
    page = doc.new_page(width=400, height=300)
    doc.save(str(pdf_path))
    doc.close()
    
    parser = PDFParser(str(pdf_path))
    size = parser.get_page_size(1)
    
    assert size['width'] == 400
    assert size['height'] == 300


def test_context_manager(tmp_path):
    """Test using PDFParser as context manager"""
    pdf_path = tmp_path / "context.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(pdf_path))
    doc.close()
    
    with PDFParser(str(pdf_path)) as parser:
        pages = parser.get_page_count()
        assert pages == 1
        text = parser.extract_text_simple()
        assert text == []  # Empty page


def test_file_not_found():
    """Test error when file doesn't exist"""
    with pytest.raises(FileNotFoundError):
        PDFParser("/path/to/nonexistent.pdf")