"""Main TextAligner class with all proxy methods for backward compatibility."""

from typing import List, Tuple
from backend.services.models import TextFragment
from backend.services.block_classifier import BlockClassifier
from backend.services.alignment.strategies import align_sequential, align_with_ai, align_with_semantic
from backend.services.alignment.classifier_mixin import (
    classify_fragments, filter_body_fragments, extract_special_blocks
)
from backend.services.alignment.similarity import calculate_similarity, calculate_semantic_similarity
from backend.services.alignment.matching import find_best_match, find_best_match_semantic
from backend.services.alignment.text_utils import normalize_text, transliterate_russian
from backend.logger import logger


class TextAligner:
    def __init__(self, use_ai=False, use_classifier=False, use_semantic=False):
        self.use_ai = use_ai
        self.use_classifier = use_classifier
        self.use_semantic = use_semantic
        self.classifier = BlockClassifier() if use_classifier else None
        self.match_threshold = 0.1
        self.embedding_model = None
        
        if use_semantic:
            self._load_semantic_model()
    
    def _load_semantic_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('LaBSE')
            logger.info("Loaded LaBSE semantic model")
        except Exception as e:
            logger.warning(f"Could not load semantic model: {e}")
            self.use_semantic = False
    
    # ── Proxy methods for backward compatibility ──
    
    def prepare_texts(self, donor, sample):
        return [f.text for f in donor], [f.text for f in sample]
    
    def _normalize_text(self, text):
        return normalize_text(text)
    
    def _transliterate_russian(self, text):
        return transliterate_russian(text)
    
    def calculate_similarity(self, text1, text2):
        return calculate_similarity(text1, text2)
    
    def calculate_semantic_similarity(self, text1, text2):
        return calculate_semantic_similarity(self.embedding_model, text1, text2)
    
    def find_best_match(self, target, candidates):
        return find_best_match(target, candidates)
    
    def find_best_match_semantic(self, target, candidates):
        return find_best_match_semantic(self.embedding_model, target, candidates)
    
    def align_sequential(self, donor, sample):
        return align_sequential(donor, sample)
    
    def _align_with_ai(self, donor, sample):
        return align_with_ai(donor, sample, self.match_threshold)
    
    def _align_with_semantic(self, donor, sample):
        return align_with_semantic(self.embedding_model, donor, sample)
    
    def align(self, donor, sample):
        if self.use_semantic:
            return align_with_semantic(self.embedding_model, donor, sample)
        elif self.use_ai:
            return align_with_ai(donor, sample, self.match_threshold)
        else:
            return align_sequential(donor, sample)
    
    def classify_fragments(self, fragments, page_dimensions):
        return classify_fragments(self.classifier, fragments, page_dimensions)
    
    def filter_body_fragments(self, fragments, page_dimensions):
        return filter_body_fragments(self.classifier, fragments, page_dimensions)
    
    def extract_special_blocks(self, fragments, page_dimensions):
        return extract_special_blocks(self.classifier, fragments, page_dimensions)
    
    def align_with_classifier(self, donor, sample, page_dimensions):
        if not self.use_classifier:
            return self.align(donor, sample)
        donor_body = self.filter_body_fragments(donor, page_dimensions)
        sample_body = self.filter_body_fragments(sample, page_dimensions)
        return self.align(donor_body, sample_body)