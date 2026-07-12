"""Block classifier integration for alignment."""

from typing import List, Tuple
from backend.services.models import TextFragment
from backend.services.block_classifier import BlockClassifier, BlockType


def classify_fragments(classifier, fragments, page_dimensions):
    if not classifier:
        return [(f, BlockType.BODY, 0.5) for f in fragments]
    classified = []
    for fragment in fragments:
        block_type, confidence = classifier.classify_block([fragment], page_dimensions)
        classified.append((fragment, block_type, confidence))
    return classified


def filter_body_fragments(classifier, fragments, page_dimensions):
    if not classifier:
        return fragments
    classified = classify_fragments(classifier, fragments, page_dimensions)
    return [f for f, bt, c in classified if not classifier.should_ignore(bt)]


def extract_special_blocks(classifier, fragments, page_dimensions):
    if not classifier:
        return []
    classified = classify_fragments(classifier, fragments, page_dimensions)
    return [(f, bt) for f, bt, c in classified if classifier.should_align_separately(bt)]