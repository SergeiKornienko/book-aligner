"""Fragment matching methods."""

from typing import List, Tuple, Optional
from backend.services.models import TextFragment
from backend.services.alignment.similarity import calculate_similarity, calculate_semantic_similarity


def find_best_match(
    target: TextFragment,
    candidates: List[TextFragment]
) -> Tuple[Optional[TextFragment], float]:
    if not candidates:
        return None, 0.0
    
    best_match = None
    best_score = 0.0
    
    for candidate in candidates:
        score = calculate_similarity(target.text, candidate.text)
        if score > best_score:
            best_score = score
            best_match = candidate
    
    return best_match, best_score


def find_best_match_semantic(
    embedding_model,
    target: TextFragment,
    candidates: List[TextFragment]
) -> Tuple[Optional[TextFragment], float]:
    if not candidates:
        return None, 0.0
    
    best_match = None
    best_score = 0.0
    
    for candidate in candidates:
        score = calculate_semantic_similarity(embedding_model, target.text, candidate.text)
        if score > best_score:
            best_score = score
            best_match = candidate
    
    return best_match, best_score