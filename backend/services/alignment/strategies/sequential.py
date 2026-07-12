"""Sequential alignment strategy."""

from typing import List, Tuple
from backend.services.models import TextFragment


def align_sequential(
    donor_fragments: List[TextFragment],
    sample_fragments: List[TextFragment]
) -> List[Tuple[TextFragment, TextFragment]]:
    aligned_pairs = []
    min_length = min(len(donor_fragments), len(sample_fragments))
    
    for i in range(min_length):
        donor_frag = donor_fragments[i]
        sample_frag = sample_fragments[i]
        
        aligned_sample = TextFragment(
            text=sample_frag.text,
            x=donor_frag.x, y=donor_frag.y,
            width=donor_frag.width, height=donor_frag.height,
            font_name=sample_frag.font_name,
            font_size=sample_frag.font_size,
            page_num=donor_frag.page_num
        )
        aligned_pairs.append((donor_frag, aligned_sample))
    
    return aligned_pairs