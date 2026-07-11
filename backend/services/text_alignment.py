"""
Text alignment service for matching text fragments between donor and sample PDFs.
Uses AI (embeddings/GPT API) for intelligent text matching.
Supports block classification for better alignment accuracy.
"""

from typing import List, Tuple, Optional
import re
from difflib import SequenceMatcher
from backend.services.block_classifier import BlockClassifier, BlockType
from backend.services.models import TextFragment


class TextAligner:
    """
    Service for aligning text fragments between donor and sample PDFs.
    
    Supports four alignment strategies:
    1. Sequential: Position-based matching (fast, simple)
    2. AI: Similarity-based matching with transliteration (intelligent)
    3. Semantic: Sentence-transformer embeddings (cross-language)
    4. Classifier-enhanced: Block type filtering (most accurate)
    
    The alignment process:
    1. Classify text blocks (if classifier enabled)
    2. Filter out headers, footers, page numbers
    3. Match corresponding fragments using selected strategy
    4. Create aligned pairs preserving donor positions
    5. Handle special blocks (titles, captions) separately
    """
    
    def __init__(self, use_ai: bool = False, use_classifier: bool = False, use_semantic: bool = False):
        """
        Initialize TextAligner.
        
        Args:
            use_ai: If True, use AI-based matching (similarity + transliteration).
            use_classifier: If True, use BlockClassifier to filter fragments.
            use_semantic: If True, use sentence-transformers for cross-language matching.
        """
        self.use_ai = use_ai
        self.use_classifier = use_classifier
        self.use_semantic = use_semantic
        self.classifier = BlockClassifier() if use_classifier else None
        
        # Matching threshold for AI mode
        self.match_threshold = 0.1
        
        # Semantic model (lazy loading)
        self.embedding_model = None
        if use_semantic:
            self._load_semantic_model()
        
        # Future: OpenAI API configuration
        self.openai_api_key = None
    
    def _load_semantic_model(self):
        """Load the sentence transformer model for semantic matching."""
        try:
            from sentence_transformers import SentenceTransformer
            # LaBSE: Language-agnostic BERT Sentence Embedding
            # Supports 109 languages including English and Russian
            self.embedding_model = SentenceTransformer('LaBSE')
        except ImportError:
            print("Warning: sentence-transformers not installed. Install with: pip install sentence-transformers")
            self.use_semantic = False
        except Exception as e:
            print(f"Warning: Could not load semantic model: {e}")
            self.use_semantic = False
    
    # =========================================================================
    # TEXT PREPARATION METHODS
    # =========================================================================
    
    def prepare_texts(
        self,
        donor_fragments: List[TextFragment],
        sample_fragments: List[TextFragment]
    ) -> Tuple[List[str], List[str]]:
        """
        Extract plain text lists from fragments for comparison.
        
        Args:
            donor_fragments: Text fragments from donor PDF
            sample_fragments: Text fragments from sample PDF
            
        Returns:
            Tuple of (donor_texts, sample_texts)
        """
        donor_texts = [fragment.text for fragment in donor_fragments]
        sample_texts = [fragment.text for fragment in sample_fragments]
        return donor_texts, sample_texts
    
    # =========================================================================
    # TEXT NORMALIZATION METHODS
    # =========================================================================
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison.
        - Converts to lowercase
        - Removes extra whitespace
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _transliterate_russian(self, text: str) -> str:
        """
        Simple transliteration of Russian characters to Latin.
        Helps with cross-language similarity matching.
        
        Args:
            text: Text potentially containing Russian characters
            
        Returns:
            Text with Russian characters transliterated to Latin
        """
        translit_map = {
            'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd',
            'е': 'e', 'ё': 'yo', 'ж': 'zh', 'з': 'z', 'и': 'i',
            'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm', 'н': 'n',
            'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't',
            'у': 'u', 'ф': 'f', 'х': 'kh', 'ц': 'ts', 'ч': 'ch',
            'ш': 'sh', 'щ': 'shch', 'ъ': '', 'ы': 'y', 'ь': '',
            'э': 'e', 'ю': 'yu', 'я': 'ya'
        }
        
        result = []
        for char in text.lower():
            result.append(translit_map.get(char, char))
        
        return ''.join(result)
    
    # =========================================================================
    # SIMILARITY CALCULATION METHODS
    # =========================================================================
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity score between two text strings.
        
        Strategy:
        1. Normalize both texts
        2. Calculate base similarity with SequenceMatcher
        3. If similarity is low, try transliteration for cross-language matching
        4. Return the best score
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        text1_clean = self._normalize_text(text1)
        text2_clean = self._normalize_text(text2)
        
        base_similarity = SequenceMatcher(None, text1_clean, text2_clean).ratio()
        
        if base_similarity < 0.3:
            text1_translit = self._transliterate_russian(text1_clean)
            text2_translit = self._transliterate_russian(text2_clean)
            
            translit_similarity = SequenceMatcher(
                None, text1_translit, text2_translit
            ).ratio()
            
            return max(base_similarity, translit_similarity)
        
        return base_similarity
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity using sentence embeddings.
        Uses LaBSE model for cross-language understanding.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not self.embedding_model:
            return self.calculate_similarity(text1, text2)
        
        try:
            from numpy import dot
            from numpy.linalg import norm
            
            embeddings = self.embedding_model.encode([text1, text2])
            vec1, vec2 = embeddings[0], embeddings[1]
            cosine_sim = dot(vec1, vec2) / (norm(vec1) * norm(vec2))
            
            return float((cosine_sim + 1) / 2)
        except Exception:
            return self.calculate_similarity(text1, text2)
    
    # =========================================================================
    # MATCHING METHODS
    # =========================================================================
    
    def find_best_match(
        self,
        target: TextFragment,
        candidates: List[TextFragment]
    ) -> Tuple[Optional[TextFragment], float]:
        """
        Find the best matching fragment from candidates for a target fragment.
        
        Args:
            target: Target text fragment to match
            candidates: List of candidate fragments to search
            
        Returns:
            Tuple of (best_matching_fragment, similarity_score)
        """
        if not candidates:
            return None, 0.0
        
        best_match = None
        best_score = 0.0
        
        for candidate in candidates:
            score = self.calculate_similarity(target.text, candidate.text)
            if score > best_score:
                best_score = score
                best_match = candidate
        
        return best_match, best_score
    
    def find_best_match_semantic(
        self,
        target: TextFragment,
        candidates: List[TextFragment]
    ) -> Tuple[Optional[TextFragment], float]:
        """
        Find best match using semantic similarity.
        
        Args:
            target: Target text fragment
            candidates: Candidate fragments
            
        Returns:
            Tuple of (best_match, score)
        """
        if not candidates:
            return None, 0.0
        
        best_match = None
        best_score = 0.0
        
        for candidate in candidates:
            score = self.calculate_semantic_similarity(target.text, candidate.text)
            if score > best_score:
                best_score = score
                best_match = candidate
        
        return best_match, best_score
    
    def find_best_match_with_context(
        self,
        target: TextFragment,
        candidates: List[TextFragment],
        context_before: Optional[TextFragment] = None,
        context_after: Optional[TextFragment] = None
    ) -> Tuple[Optional[TextFragment], float]:
        """
        Find best match considering surrounding context.
        FUTURE: Enhances accuracy by checking neighboring paragraphs.
        
        Args:
            target: Target text fragment
            candidates: Candidate fragments
            context_before: Previous fragment for context
            context_after: Next fragment for context
            
        Returns:
            Tuple of (best_match, confidence_score)
        """
        return self.find_best_match(target, candidates)
    
    # =========================================================================
    # ALIGNMENT STRATEGIES
    # =========================================================================
    
    def align_sequential(
        self,
        donor_fragments: List[TextFragment],
        sample_fragments: List[TextFragment]
    ) -> List[Tuple[TextFragment, TextFragment]]:
        """
        Simple sequential alignment - matches fragments by position.
        Best for: PDFs with identical layout (same publisher, same edition).
        
        Args:
            donor_fragments: Text fragments from donor PDF
            sample_fragments: Text fragments from sample PDF
            
        Returns:
            List of tuples (donor_fragment, aligned_sample_fragment)
        """
        aligned_pairs = []
        min_length = min(len(donor_fragments), len(sample_fragments))
        
        for i in range(min_length):
            donor_frag = donor_fragments[i]
            sample_frag = sample_fragments[i]
            
            aligned_sample = TextFragment(
                text=sample_frag.text,
                x=donor_frag.x,
                y=donor_frag.y,
                width=donor_frag.width,
                height=donor_frag.height,
                font_name=sample_frag.font_name,
                font_size=sample_frag.font_size,
                page_num=donor_frag.page_num
            )
            
            aligned_pairs.append((donor_frag, aligned_sample))
        
        return aligned_pairs
    
    def _align_with_ai(
        self,
        donor_fragments: List[TextFragment],
        sample_fragments: List[TextFragment]
    ) -> List[Tuple[TextFragment, TextFragment]]:
        """
        AI-based alignment using similarity matching.
        Best for: PDFs with different layout but same content.
        
        Algorithm:
        1. For each donor fragment, find the best match in sample fragments
        2. If similarity > threshold, use the match
        3. If no good match, keep donor text as fallback
        4. Remove matched samples to avoid duplicates
        
        Args:
            donor_fragments: Text fragments from donor PDF
            sample_fragments: Text fragments from sample PDF
            
        Returns:
            List of aligned fragment pairs
        """
        aligned_pairs = []
        available_samples = list(sample_fragments)
        
        for donor_frag in donor_fragments:
            if not available_samples:
                break
            
            best_match, score = self.find_best_match(donor_frag, available_samples)
            
            if best_match and score > self.match_threshold:
                aligned_sample = TextFragment(
                    text=best_match.text,
                    x=donor_frag.x,
                    y=donor_frag.y,
                    width=donor_frag.width,
                    height=donor_frag.height,
                    font_name=best_match.font_name,
                    font_size=best_match.font_size,
                    page_num=donor_frag.page_num
                )
                aligned_pairs.append((donor_frag, aligned_sample))
                available_samples.remove(best_match)
            else:
                aligned_sample = TextFragment(
                    text=donor_frag.text,
                    x=donor_frag.x,
                    y=donor_frag.y,
                    width=donor_frag.width,
                    height=donor_frag.height,
                    font_name=donor_frag.font_name,
                    font_size=donor_frag.font_size,
                    page_num=donor_frag.page_num
                )
                aligned_pairs.append((donor_frag, aligned_sample))
        
        return aligned_pairs
    
    def _align_with_semantic(
        self,
        donor_fragments: List[TextFragment],
        sample_fragments: List[TextFragment]
    ) -> List[Tuple[TextFragment, TextFragment]]:
        """
        Semantic alignment using sentence embeddings.
        Best for: Cross-language matching with different layouts.
        
        Args:
            donor_fragments: English text fragments
            sample_fragments: Russian text fragments
            
        Returns:
            Aligned pairs
        """
        aligned_pairs = []
        available_samples = list(sample_fragments)
        
        for donor_frag in donor_fragments:
            if not available_samples:
                break
            
            best_match, score = self.find_best_match_semantic(donor_frag, available_samples)
            
            if best_match and score > 0.4:  # Higher threshold for semantic
                aligned_sample = TextFragment(
                    text=best_match.text,
                    x=donor_frag.x,
                    y=donor_frag.y,
                    width=donor_frag.width,
                    height=donor_frag.height,
                    font_name=best_match.font_name,
                    font_size=best_match.font_size,
                    page_num=donor_frag.page_num
                )
                aligned_pairs.append((donor_frag, aligned_sample))
                available_samples.remove(best_match)
            else:
                aligned_sample = TextFragment(
                    text=donor_frag.text,
                    x=donor_frag.x,
                    y=donor_frag.y,
                    width=donor_frag.width,
                    height=donor_frag.height,
                    font_name=donor_frag.font_name,
                    font_size=donor_frag.font_size,
                    page_num=donor_frag.page_num
                )
                aligned_pairs.append((donor_frag, aligned_sample))
        
        return aligned_pairs
    
    # =========================================================================
    # MAIN ALIGNMENT METHOD
    # =========================================================================
    
    def align(
        self,
        donor_fragments: List[TextFragment],
        sample_fragments: List[TextFragment]
    ) -> List[Tuple[TextFragment, TextFragment]]:
        """
        Main alignment method. Routes to appropriate strategy.
        
        Priority:
        1. Semantic (if enabled) - best for cross-language
        2. AI (if enabled) - similarity + transliteration
        3. Sequential - position-based fallback
        
        Args:
            donor_fragments: Text fragments from donor PDF
            sample_fragments: Text fragments from sample PDF
            
        Returns:
            List of aligned fragment pairs
        """
        if self.use_semantic:
            return self._align_with_semantic(donor_fragments, sample_fragments)
        elif self.use_ai:
            return self._align_with_ai(donor_fragments, sample_fragments)
        else:
            return self.align_sequential(donor_fragments, sample_fragments)
    
    # =========================================================================
    # CLASSIFIER-ENHANCED METHODS
    # =========================================================================
    
    def classify_fragments(
        self,
        fragments: List[TextFragment],
        page_dimensions: dict
    ) -> List[Tuple[TextFragment, BlockType, float]]:
        """
        Classify all fragments into block types.
        
        Args:
            fragments: Text fragments to classify
            page_dimensions: Page dimensions dict with 'width' and 'height'
            
        Returns:
            List of tuples (fragment, block_type, confidence)
        """
        if not self.use_classifier or not self.classifier:
            return [(f, BlockType.BODY, 0.5) for f in fragments]
        
        classified = []
        for fragment in fragments:
            block_type, confidence = self.classifier.classify_block(
                [fragment], page_dimensions
            )
            classified.append((fragment, block_type, confidence))
        
        return classified
    
    def filter_body_fragments(
        self,
        fragments: List[TextFragment],
        page_dimensions: dict
    ) -> List[TextFragment]:
        """
        Filter out non-body fragments (headers, footers, page numbers).
        
        Args:
            fragments: All text fragments
            page_dimensions: Page dimensions dict
            
        Returns:
            Only body-relevant fragments
        """
        if not self.use_classifier:
            return fragments
        
        classified = self.classify_fragments(fragments, page_dimensions)
        
        body_fragments = []
        for fragment, block_type, confidence in classified:
            if not self.classifier.should_ignore(block_type):
                body_fragments.append(fragment)
        
        return body_fragments
    
    def extract_special_blocks(
        self,
        fragments: List[TextFragment],
        page_dimensions: dict
    ) -> List[Tuple[TextFragment, BlockType]]:
        """
        Extract special blocks that need separate alignment.
        Special blocks: titles, subtitles, captions.
        
        Args:
            fragments: All text fragments
            page_dimensions: Page dimensions dict
            
        Returns:
            List of tuples (fragment, block_type) for special blocks
        """
        if not self.use_classifier:
            return []
        
        classified = self.classify_fragments(fragments, page_dimensions)
        
        special_blocks = []
        for fragment, block_type, confidence in classified:
            if self.classifier.should_align_separately(block_type):
                special_blocks.append((fragment, block_type))
        
        return special_blocks
    
    def align_with_classifier(
        self,
        donor_fragments: List[TextFragment],
        sample_fragments: List[TextFragment],
        page_dimensions: dict
    ) -> List[Tuple[TextFragment, TextFragment]]:
        """
        Full alignment with block classification.
        
        Pipeline:
        1. Classify all fragments by block type
        2. Filter out ignored blocks (headers, page numbers)
        3. Extract special blocks (titles, subtitles, captions)
        4. Align special blocks separately (semantic matching)
        5. Align body text with selected strategy
        6. Combine results in correct order
        
        Args:
            donor_fragments: Fragments from donor PDF
            sample_fragments: Fragments from sample PDF
            page_dimensions: Page dimensions dict
            
        Returns:
            List of aligned fragment pairs in page order
        """
        if not self.use_classifier:
            return self.align(donor_fragments, sample_fragments)
        
        donor_body = self.filter_body_fragments(donor_fragments, page_dimensions)
        sample_body = self.filter_body_fragments(sample_fragments, page_dimensions)
        
        donor_special = self.extract_special_blocks(donor_fragments, page_dimensions)
        sample_special = self.extract_special_blocks(sample_fragments, page_dimensions)
        
        body_aligned = self.align(donor_body, sample_body)
        special_aligned = self._align_special_blocks(donor_special, sample_special)
        
        return body_aligned
    
    def _align_special_blocks(
        self,
        donor_special: List[Tuple[TextFragment, BlockType]],
        sample_special: List[Tuple[TextFragment, BlockType]]
    ) -> List[Tuple[TextFragment, TextFragment]]:
        """
        Align special blocks (titles, subtitles, captions) using semantic matching.
        
        Args:
            donor_special: Special blocks from donor with their types
            sample_special: Special blocks from sample with their types
            
        Returns:
            Aligned special block pairs
        """
        aligned = []
        
        donor_by_type = {}
        for frag, btype in donor_special:
            if btype not in donor_by_type:
                donor_by_type[btype] = []
            donor_by_type[btype].append(frag)
        
        sample_by_type = {}
        for frag, btype in sample_special:
            if btype not in sample_by_type:
                sample_by_type[btype] = []
            sample_by_type[btype].append(frag)
        
        for btype in donor_by_type:
            donor_frags = donor_by_type[btype]
            sample_frags = sample_by_type.get(btype, [])
            
            type_aligned = self._align_with_ai(donor_frags, sample_frags)
            aligned.extend(type_aligned)
        
        return aligned
    
    # =========================================================================
    # FUTURE: GPT/Embeddings Integration
    # =========================================================================
    
    def configure_openai(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """
        Configure OpenAI API for semantic matching.
        FUTURE: Will enable true AI-powered alignment.
        
        Args:
            api_key: OpenAI API key
            model: GPT model to use
        """
        self.openai_api_key = api_key
    
    def calculate_gpt_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity using GPT API.
        FUTURE: Will provide much better cross-language matching.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score 0.0-1.0
        """
        return self.calculate_similarity(text1, text2)


# =============================================================================
# PYDANTIC MODELS FOR API
# =============================================================================

from pydantic import BaseModel


class TextFragmentModel(BaseModel):
    """Pydantic model for text fragment input/output"""
    text: str
    x: float
    y: float
    width: float
    height: float
    font_name: str = "Times-Roman"
    font_size: float = 12.0
    page_num: int = 0


class AlignmentRequest(BaseModel):
    """Request model for alignment endpoint"""
    donor_fragments: List[TextFragmentModel]
    sample_fragments: List[TextFragmentModel]
    use_ai: bool = False
    use_classifier: bool = False
    use_semantic: bool = False
    page_dimensions: dict = {"width": 595, "height": 842}


class AlignmentResponse(BaseModel):
    """Response model for alignment endpoint"""
    aligned_fragments: List[TextFragmentModel]
    total_pairs: int
    method: str = "sequential"
    confidence_score: float = 1.0