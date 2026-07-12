"""Semantic alignment using sentence embeddings."""

from typing import List, Tuple
from backend.services.models import TextFragment
from backend.services.alignment.matching import find_best_match_semantic


def align_with_semantic(
    embedding_model,
    donor_fragments: List[TextFragment],
    sample_fragments: List[TextFragment],
    match_threshold: float = 0.4
) -> List[Tuple[TextFragment, TextFragment]]:
    aligned_pairs = []
    available_samples = list(sample_fragments)
    
    for donor_frag in donor_fragments:
        if not available_samples:
            break
        
        best_match, score = find_best_match_semantic(
            embedding_model, donor_frag, available_samples
        )
        
        if best_match and score > match_threshold:
            aligned_sample = TextFragment(
                text=best_match.text,
                x=donor_frag.x, y=donor_frag.y,
                width=donor_frag.width, height=donor_frag.height,
                font_name=best_match.font_name,
                font_size=best_match.font_size,
                page_num=donor_frag.page_num
            )
            aligned_pairs.append((donor_frag, aligned_sample))
            available_samples.remove(best_match)
        else:
            aligned_sample = TextFragment(
                text=donor_frag.text,
                x=donor_frag.x, y=donor_frag.y,
                width=donor_frag.width, height=donor_frag.height,
                font_name=donor_frag.font_name,
                font_size=donor_frag.font_size,
                page_num=donor_frag.page_num
            )
            aligned_pairs.append((donor_frag, aligned_sample))
    
    return aligned_pairs