"""
Tests for AI-powered text alignment.
Uses embeddings to match text fragments intelligently.
"""

import pytest
from backend.services.alignment import TextAligner
from backend.services.models import TextFragment


class TestAIPoweredAlignment:
    """Tests for AI-based text matching"""
    
    @pytest.fixture
    def aligner_with_ai(self):
        """Create TextAligner with AI enabled"""
        return TextAligner(use_ai=True)
    
    @pytest.fixture
    def aligner_without_ai(self):
        """Create TextAligner without AI"""
        return TextAligner(use_ai=False)
    
    @pytest.fixture
    def complex_donor_fragments(self):
        """Donor fragments in non-sequential order"""
        return [
            TextFragment("Harry Potter and the Philosopher's Stone", 10, 10, 200, 14),
            TextFragment("CHAPTER ONE", 10, 40, 100, 16, font_size=16),
            TextFragment("The Boy Who Lived", 10, 70, 120, 14),
            TextFragment("Mr. and Mrs. Dursley, of number four, Privet Drive,", 10, 100, 300, 12, font_size=10),
        ]
    
    @pytest.fixture
    def complex_sample_fragments(self):
        """Sample fragments in different order"""
        return [
            TextFragment("ГЛАВА ПЕРВАЯ", 10, 40, 100, 16, font_size=16),
            TextFragment("Мальчик-Который-Выжил", 10, 70, 150, 14),
            TextFragment("Гарри Поттер и философский камень", 10, 10, 200, 14),
            TextFragment("Мистер и миссис Дурсль проживали в доме номер четыре по Тисовой улице", 10, 100, 400, 12, font_size=10),
        ]
    
    def test_ai_mode_enabled(self, aligner_with_ai):
        """Test that AI mode is properly enabled"""
        assert aligner_with_ai.use_ai == True
    
    def test_ai_mode_disabled(self, aligner_without_ai):
        """Test that AI mode can be disabled"""
        assert aligner_without_ai.use_ai == False
    
    def test_prepare_texts_with_complex_data(self, aligner_with_ai, complex_donor_fragments, complex_sample_fragments):
        """Test text preparation for AI alignment"""
        donor_texts, sample_texts = aligner_with_ai.prepare_texts(
            complex_donor_fragments, 
            complex_sample_fragments
        )
        
        assert len(donor_texts) == 4
        assert len(sample_texts) == 4
        assert donor_texts[0] == "Harry Potter and the Philosopher's Stone"
        assert sample_texts[2] == "Гарри Поттер и философский камень"
    
    def test_ai_alignment_fallback(self, aligner_with_ai, complex_donor_fragments, complex_sample_fragments):
        """Test that AI alignment works (falls back to sequential for now)"""
        aligned = aligner_with_ai.align(complex_donor_fragments, complex_sample_fragments)
        
        assert len(aligned) == 4
        # Check that coordinates are preserved from donor
        for i, (donor_frag, sample_frag) in enumerate(aligned):
            assert sample_frag.x == donor_frag.x
            assert sample_frag.y == donor_frag.y
    
    def test_similarity_score_basic(self, aligner_with_ai):
        """Test similarity calculation between texts"""
        # Test exact match
        text1 = "Hello world"
        text2 = "Hello world"
        
        similarity = aligner_with_ai.calculate_similarity(text1, text2)
        assert similarity == 1.0
        
        # Test different languages
        text3 = "Hello world"
        text4 = "Привет мир"
        
        similarity = aligner_with_ai.calculate_similarity(text3, text4)
        assert 0.0 <= similarity <= 1.0
    
    def test_similarity_different_lengths(self, aligner_with_ai):
        """Test similarity with different length texts"""
        text1 = "The boy who lived"
        text2 = "Мальчик-который-выжил"
        
        similarity = aligner_with_ai.calculate_similarity(text1, text2)
        assert 0.0 <= similarity <= 1.0
    
    def test_find_best_match(self, aligner_with_ai):
        """Test finding best matching fragment"""
        target = TextFragment("Harry Potter", 0, 0, 100, 12)
        candidates = [
            TextFragment("Гарри Поттер", 10, 10, 100, 12),
            TextFragment("Гермиона Грейнджер", 10, 30, 120, 12),
            TextFragment("Волан-де-Морт", 10, 50, 100, 12),
        ]
        
        best_match, score = aligner_with_ai.find_best_match(target, candidates)
        
        assert best_match is not None
        assert 0.0 <= score <= 1.0
        # "Гарри Поттер" should be the best match for "Harry Potter"
        assert best_match.text == "Гарри Поттер"
    
    def test_align_with_different_ordering(self, aligner_with_ai):
        """Test alignment when fragments are in different order"""
        # Using more similar texts for better matching
        donor = [
            TextFragment("Harry Potter and the Chamber of Secrets", 10, 10, 200, 12),
            TextFragment("The Chamber of Secrets", 10, 30, 150, 12),
            TextFragment("Ron Weasley", 10, 50, 100, 12),
        ]
        
        sample = [
            TextFragment("Рон Уизли", 15, 50, 100, 12),
            TextFragment("Гарри Поттер и Тайная комната", 15, 10, 200, 12),
            TextFragment("Тайная комната", 15, 30, 150, 12),
        ]
        
        aligned = aligner_with_ai.align(donor, sample)
        
        assert len(aligned) == 3
        # Check that alignment matched correctly
        # "Harry Potter and the Chamber of Secrets" should match with "Гарри Поттер и Тайная комната"
        assert aligned[0][1].text == "Гарри Поттер и Тайная комната"
        # "The Chamber of Secrets" should match with "Тайная комната"
        assert aligned[1][1].text == "Тайная комната"
        # "Ron Weasley" should match with "Рон Уизли"
        assert aligned[2][1].text == "Рон Уизли"
    
    def test_transliteration_helper(self, aligner_with_ai):
        """Test Russian transliteration"""
        russian_text = "Гарри Поттер"
        transliterated = aligner_with_ai._transliterate_russian(russian_text)
        assert transliterated == "garri potter"
    
    def test_similarity_with_transliteration(self, aligner_with_ai):
        """Test that transliteration helps with cross-language matching"""
        text_en = "Harry Potter"
        text_ru = "Гарри Поттер"
        
        similarity = aligner_with_ai.calculate_similarity(text_en, text_ru)
        # Should have reasonable similarity due to transliteration
        assert similarity > 0.3, f"Expected similarity > 0.3, got {similarity}"