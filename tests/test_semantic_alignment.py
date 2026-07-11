"""
Tests for semantic text alignment using sentence transformers.
Provides true cross-language text matching.
"""

import pytest
from backend.services.text_alignment import TextAligner, TextFragment


class TestSemanticAligner:
    """Tests for semantic (embedding-based) text alignment"""
    
    @pytest.fixture
    def semantic_aligner(self):
        """Create TextAligner with semantic mode"""
        return TextAligner(use_ai=True, use_semantic=True)
    
    @pytest.fixture
    def english_fragments(self):
        """English text fragments"""
        return [
            TextFragment("Harry Potter and the Philosopher's Stone", 0, 0, 200, 14),
            TextFragment("The Boy Who Lived", 0, 30, 120, 14),
            TextFragment("Mr. and Mrs. Dursley were proud to say they were perfectly normal", 0, 60, 400, 12),
            TextFragment("Thank you very much", 0, 90, 150, 12),
        ]
    
    @pytest.fixture
    def russian_fragments(self):
        """Russian translation fragments (matching order)"""
        return [
            TextFragment("Гарри Поттер и философский камень", 0, 0, 250, 14),
            TextFragment("Мальчик-Который-Выжил", 0, 30, 150, 14),
            TextFragment("Мистер и миссис Дурсль гордились тем, что они совершенно нормальные", 0, 60, 450, 12),
            TextFragment("Большое спасибо", 0, 90, 130, 12),
        ]
    
    @pytest.fixture
    def scrambled_russian_fragments(self):
        """Russian fragments in different order"""
        return [
            TextFragment("Большое спасибо", 0, 90, 130, 12),
            TextFragment("Гарри Поттер и философский камень", 0, 0, 250, 14),
            TextFragment("Мистер и миссис Дурсль гордились тем, что они совершенно нормальные", 0, 60, 450, 12),
            TextFragment("Мальчик-Который-Выжил", 0, 30, 150, 14),
        ]
    
    def test_semantic_mode_enabled(self, semantic_aligner):
        """Test that semantic mode is properly configured"""
        assert semantic_aligner.use_semantic == True
        assert semantic_aligner.embedding_model is not None
    
    def test_semantic_similarity_identical(self, semantic_aligner):
        """Test semantic similarity for identical texts"""
        text1 = "Harry Potter"
        text2 = "Harry Potter"
        
        similarity = semantic_aligner.calculate_semantic_similarity(text1, text2)
        assert similarity > 0.9  # Very similar
        assert similarity <= 1.0
    
    def test_semantic_similarity_translation(self, semantic_aligner):
        """Test semantic similarity for translations"""
        text_en = "Harry Potter and the Philosopher's Stone"
        text_ru = "Гарри Поттер и философский камень"
        
        similarity = semantic_aligner.calculate_semantic_similarity(text_en, text_ru)
        assert similarity > 0.5  # Should recognize as similar
        print(f"\nSemantic similarity EN-RU: {similarity:.3f}")
    
    def test_semantic_similarity_different(self, semantic_aligner):
        """Test semantic similarity for completely different texts"""
        text1 = "Harry Potter"
        text2 = "Microwave oven instructions"
        
        similarity = semantic_aligner.calculate_semantic_similarity(text1, text2)
        assert similarity < 0.6  # Should be low
        print(f"\nSemantic similarity different: {similarity:.3f}")
    
    def test_semantic_best_match(self, semantic_aligner):
        """Test finding best match with semantic similarity"""
        target = TextFragment("The Boy Who Lived", 0, 0, 120, 14)
        candidates = [
            TextFragment("Мистер и миссис Дурсль", 0, 0, 140, 14),
            TextFragment("Мальчик-Который-Выжил", 0, 0, 150, 14),
            TextFragment("Гарри Поттер", 0, 0, 100, 14),
        ]
        
        best_match, score = semantic_aligner.find_best_match_semantic(target, candidates)
        
        assert best_match is not None
        assert best_match.text == "Мальчик-Который-Выжил"
        assert score > 0.5
    
    def test_semantic_alignment_ordered(self, semantic_aligner, english_fragments, russian_fragments):
        """Test alignment with matching order"""
        aligned = semantic_aligner._align_with_semantic(english_fragments, russian_fragments)
        
        assert len(aligned) == 4
        # Should match correctly
        assert aligned[0][1].text == "Гарри Поттер и философский камень"
        assert aligned[1][1].text == "Мальчик-Который-Выжил"
        assert aligned[2][1].text == "Мистер и миссис Дурсль гордились тем, что они совершенно нормальные"
    
    def test_semantic_alignment_scrambled(self, semantic_aligner, english_fragments, scrambled_russian_fragments):
        """Test alignment with scrambled order — semantic should fix it"""
        aligned = semantic_aligner._align_with_semantic(english_fragments, scrambled_russian_fragments)
        
        assert len(aligned) == 4
        
        # Check each English fragment matched to correct Russian translation
        texts = {d.text: s.text for d, s in aligned}
        assert texts["Harry Potter and the Philosopher's Stone"] == "Гарри Поттер и философский камень"
        assert texts["The Boy Who Lived"] == "Мальчик-Который-Выжил"