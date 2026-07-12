"""Similarity calculation methods."""

from difflib import SequenceMatcher
from backend.services.alignment.text_utils import normalize_text, transliterate_russian
from backend.logger import logger


def calculate_similarity(text1: str, text2: str) -> float:
    text1_clean = normalize_text(text1)
    text2_clean = normalize_text(text2)
    
    base_similarity = SequenceMatcher(None, text1_clean, text2_clean).ratio()
    
    if base_similarity < 0.3:
        text1_translit = transliterate_russian(text1_clean)
        text2_translit = transliterate_russian(text2_clean)
        translit_similarity = SequenceMatcher(None, text1_translit, text2_translit).ratio()
        return max(base_similarity, translit_similarity)
    
    return base_similarity


def calculate_semantic_similarity(embedding_model, text1: str, text2: str) -> float:
    if not embedding_model:
        return calculate_similarity(text1, text2)
    try:
        from numpy import dot
        from numpy.linalg import norm
        embeddings = embedding_model.encode([text1, text2])
        vec1, vec2 = embeddings[0], embeddings[1]
        cosine_sim = dot(vec1, vec2) / (norm(vec1) * norm(vec2))
        return float((cosine_sim + 1) / 2)
    except Exception:
        return calculate_similarity(text1, text2)