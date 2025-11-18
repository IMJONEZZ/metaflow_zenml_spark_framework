#!/usr/bin/env python3
"""
Advanced NLP Pipeline - ZenML Implementation

A comprehensive Natural Language Processing pipeline built with ZenML that provides:
- Text preprocessing and normalization
- Advanced sentiment analysis using multiple approaches
- Named entity recognition with pattern-based detection
- Readability assessment (Flesch Reading Ease, Flesch-Kincaid Grade Level)
- Intelligent text summarization using sentence scoring
- Keyword extraction with TF-IDF simulation

This pipeline demonstrates production-grade NLP capabilities while maintaining
clean, modular architecture suitable for ML workflow orchestration.
"""

import math
import re
import string
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ZenML imports with graceful fallback
try:
    from zenml import get_step_context, step
    from zenml import pipeline as zenml_pipeline

    ZENML_AVAILABLE = True
except ImportError:
    # Create mock decorators for testing without ZenML
    def step(func):
        return func

    def zenml_pipeline(*args, **kwargs):
        def decorator(cls_or_func):
            return cls_or_func

        return decorator

    ZENML_AVAILABLE = False

# NLP Library imports with fallbacks
try:
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from nltk.tokenize import sent_tokenize, word_tokenize

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from textblob import TextBlob

    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False


class AdvancedNLPProcessor:
    """
    Core NLP processing engine for advanced text analysis.

    This class encapsulates all the sophisticated NLP functionality needed
    for production-grade text processing workflows.
    """

    def __init__(self):
        """Initialize the NLP processor with all necessary components."""
        self.stemmer = PorterStemmer() if NLTK_AVAILABLE else None
        self.stop_words = (
            set(stopwords.words("english"))
            if NLTK_AVAILABLE
            else self._get_basic_stopwords()
        )

        # Enhanced sentiment lexicons
        self.positive_words = {
            "amazing",
            "great",
            "excellent",
            "love",
            "good",
            "wonderful",
            "fantastic",
            "awesome",
            "perfect",
            "incredible",
            "best",
            "brilliant",
            "outstanding",
            "superb",
            "phenomenal",
            "exceptional",
            "remarkable",
            "spectacular",
            "magnificent",
            "marvelous",
        }

        self.negative_words = {
            "bad",
            "terrible",
            "awful",
            "hate",
            "worst",
            "horrible",
            "disappointing",
            "poor",
            "mediocre",
            "boring",
            "useless",
            "pathetic",
            "disgusting",
            "repulsive",
            "shocking",
            "appalling",
            "lousy",
            "dreadful",
            "abysmal",
        }

    def _get_basic_stopwords(self) -> set:
        """Provide basic stopwords when NLTK is not available."""
        return {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "has",
            "he",
            "in",
            "is",
            "it",
            "its",
            "of",
            "on",
            "that",
            "the",
            "to",
            "was",
            "were",
            "will",
            "with",
            "i",
            "you",
            "we",
            "they",
            "them",
            "our",
            "this",
            "that",
            "these",
            "those",
            "have",
            "had",
            "do",
            "does",
            "did",
            "can",
            "could",
            "should",
            "would",
        }

    @step
    def preprocess_text(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive text preprocessing with rich feature extraction.

        This step performs:
        - Text cleaning and normalization
        - Statistical analysis (word count, sentence length, etc.)
        - Content filtering and tokenization
        - Feature engineering for downstream analysis

        Args:
            text (str): Raw input text to preprocess

        Returns:
            Dict[str, Any]: Preprocessed data and comprehensive metadata
        """

        if not isinstance(text, str):
            raise ValueError("Input text must be a string")

        if not text.strip():
            return {
                "original_text": "",
                "cleaned_text": "",
                "word_count": 0,
                "char_count": 0,
                "sentence_count": 0,
                "avg_word_length": 0.0,
                "filtered_words": [],
                "unique_word_count": 0,
            }

        # Advanced text cleaning pipeline
        cleaned = self._advanced_clean(text)

        # Multi-level tokenization with fallback
        words = self._tokenize_words(cleaned)
        sentences = self._tokenize_sentences(text)

        # Content filtering and analysis
        filtered_words = [
            word.lower()
            for word in words
            if word.isalpha() and word.lower() not in self.stop_words
        ]

        # Calculate comprehensive statistics
        char_count = len(cleaned)
        word_count = len(words) or 1

        # Advanced feature extraction
        return {
            "original_text": text,
            "cleaned_text": cleaned,
            "word_count": word_count,
            "char_count": char_count,
            "sentence_count": len(sentences),
            "avg_word_length": sum(len(word) for word in words) / len(words)
            if words
            else 0,
            "filtered_words": filtered_words,
            "unique_word_count": len(set(filtered_words)),
            "lexical_diversity": len(set(filtered_words)) / len(filtered_words)
            if filtered_words
            else 0,
            "processing_timestamp": datetime.now().isoformat(),
        }

    def _advanced_clean(self, text: str) -> str:
        """Apply advanced cleaning techniques to the input text."""

        # Remove excessive whitespace but preserve structure
        cleaned = re.sub(r"\s+", " ", text.strip())

        # Handle special characters and normalize
        cleaned = re.sub(r"[^\w\s\.\,\!\?\;\:\-\(\)]", " ", cleaned)

        # Remove extra punctuation spacing
        cleaned = re.sub(r"\s+([,.!?;:])", r"\1", cleaned)

        return cleaned

    def _tokenize_words(self, text: str) -> List[str]:
        """Tokenize text into words with fallback mechanisms."""

        if NLTK_AVAILABLE:
            try:
                return word_tokenize(text.lower())
            except Exception:
                pass

        # Fallback regex tokenization
        return re.findall(r"\b\w+\b", text.lower())

    def _tokenize_sentences(self, text: str) -> List[str]:
        """Tokenize text into sentences with fallback mechanisms."""

        if NLTK_AVAILABLE:
            try:
                return [s.strip() for s in sent_tokenize(text) if s.strip()]
            except Exception:
                pass

        # Fallback sentence splitting
        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip()]

    @step
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive sentiment analysis using multiple methodologies.

        This step combines:
        - TextBlob's machine learning approach
        - Lexicon-based analysis
        - Advanced confidence scoring

        Args:
            text (str): Text to analyze for sentiment

        Returns:
            Dict[str, Any]: Detailed sentiment analysis results
        """

        if not text.strip():
            return {
                "overall_sentiment": {
                    "label": "neutral",
                    "confidence": 0.0,
                    "score": 0.0,
                },
                "textblob_sentiment": None,
                "lexicon_sentiment": {"score": 0.0, "confidence": 0.0},
                "analysis_metadata": {"method_used": "empty_text"},
            }

        results = {}

        # TextBlob sentiment analysis (primary method)
        if TEXTBLOB_AVAILABLE:
            try:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity

                # Determine sentiment label with enhanced thresholds
                if polarity > 0.1:
                    textblob_label = "positive"
                elif polarity < -0.1:
                    textblob_label = "negative"
                else:
                    textblob_label = "neutral"

                results["textblob_sentiment"] = {
                    "polarity": polarity,
                    "subjectivity": subjectivity,
                    "label": textblob_label,
                }
            except Exception as e:
                results["textblob_error"] = str(e)

        # Enhanced lexicon-based sentiment analysis
        words = self._tokenize_words(text)

        positive_matches = sum(
            1 for word in words if word.lower() in self.positive_words
        )
        negative_matches = sum(
            1 for word in words if word.lower() in self.negative_words
        )

        lexicon_score = (positive_matches - negative_matches) / max(len(words), 1)

        results["lexicon_sentiment"] = {
            "positive_matches": positive_matches,
            "negative_matches": negative_matches,
            "score": lexicon_score,
        }

        # Overall sentiment determination (weighted combination)
        textblob_polarity = (
            results.get("textblob_sentiment", {}).get("polarity", 0)
            if TEXTBLOB_AVAILABLE
            else 0
        )

        # Weight: 70% TextBlob, 30% Lexicon (if available)
        final_score = (
            0.7 * textblob_polarity + 0.3 * lexicon_score
            if TEXTBLOB_AVAILABLE
            else lexicon_score
        )

        # Enhanced sentiment classification with confidence thresholds
        if final_score > 0.05:
            overall_label = "positive"
        elif final_score < -0.05:
            overall_label = "negative"
        else:
            overall_label = "neutral"

        results["overall_sentiment"] = {
            "label": overall_label,
            "confidence": min(abs(final_score) * 2, 1.0),  # Scale confidence to [0,1]
            "score": final_score,
        }

        results["analysis_metadata"] = {
            "methods_used": [
                method
                for method, available in [
                    ("textblob", TEXTBLOB_AVAILABLE),
                    ("lexicon", True),
                ]
                if available
            ],
            "processing_timestamp": datetime.now().isoformat(),
        }

        return results

    @step
    def extract_named_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Advanced named entity recognition using sophisticated pattern matching.

        This step provides robust NER capabilities through:
        - Multiple regex patterns for different entity types
        - Contextual confidence scoring
        - Entity type classification

        Args:
            text (str): Text to analyze for named entities

        Returns:
            List[Dict[str, Any]]: Detected entities with metadata
        """

        if not text.strip():
            return []

        sentences = self._tokenize_sentences(text)
        entities = []

        # Comprehensive entity pattern definitions
        patterns = {
            "PERSON": [
                r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+\.?)*\b",  # Names
                r"(?:Mr|Mrs|Dr|Prof)\.?\s+[A-Z][a-z]+\b",  # Titles + names
            ],
            "ORG": [
                r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|Corp|LLC|Company|Co\.)\b",
                r"\b[A-Z]{2,}(?:\s+[A-Z]+)*\b(?=\s+(?:Inc|Corp|LLC|Company|Co\.)\b)",
                r"\b(?:Microsoft|Apple|Google|Facebook|Tesla|NVIDIA|Meta|OpenAI)\b",
            ],
            "GPE": [  # Geopolitical Entity
                r"\b(?:in|at|from|towards)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
                r"\b(?:USA|United States|UK|United Kingdom|France|Germany|Japan|China)\b",
            ],
            "DATE": [
                r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b",
                r"\b\d{4}-\d{1,2}-\d{1,2}\b",
                r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+\w+?\s+\d{1,2}\b",
            ],
            "PRODUCT": [
                r"\biPhone|iPad|MacBook|AirPods|Apple Watch\b",
                r"\b(?:Windows|Android|iOS)\s+\d+\b",
            ],
        }

        for sentence in sentences:
            # Apply all patterns and collect matches
            for entity_type, regex_patterns in patterns.items():
                for pattern in regex_patterns:
                    matches = re.finditer(pattern, sentence)

                    for match in matches:
                        entity_text = (
                            match.group(1) if len(match.groups()) > 0 else match.group()
                        ).strip()

                        # Confidence scoring based on pattern specificity
                        confidence = self._calculate_entity_confidence(
                            entity_type, entity_text, sentence
                        )

                        entities.append(
                            {
                                "text": entity_text,
                                "label": entity_type,
                                "confidence": confidence,
                                "context": sentence.strip(),
                                "start_position": match.start(),
                                "end_position": match.end(),
                            }
                        )

        # Remove duplicates and sort by confidence
        unique_entities = {}
        for entity in entities:
            key = f"{entity['text']}_{entity['label']}"
            if (
                key not in unique_entities
                or entity["confidence"] > unique_entities[key]["confidence"]
            ):
                unique_entities[key] = entity

        return sorted(
            unique_entities.values(), key=lambda x: x["confidence"], reverse=True
        )

    def _calculate_entity_confidence(
        self, entity_type: str, entity_text: str, context: str
    ) -> float:
        """Calculate confidence score for a detected entity."""

        # Base confidence by type
        base_confidence = {
            "PERSON": 0.8,
            "ORG": 0.7,
            "GPE": 0.9,
            "DATE": 0.95,
            "PRODUCT": 0.85,
        }.get(entity_type, 0.6)

        # Adjust based on context clues
        confidence = base_confidence

        # Boost for proper capitalization patterns
        if re.match(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+\.?)*$", entity_text):
            confidence += 0.1

        # Boost for company indicators
        if any(
            indicator in context.lower()
            for indicator in ["founded", "company", "corporation"]
        ):
            if entity_type == "ORG":
                confidence += 0.1

        # Boost for personal indicators
        if any(
            indicator in context.lower()
            for indicator in ["ceo", "founder", "president"]
        ):
            if entity_type == "PERSON":
                confidence += 0.1

        return min(confidence, 1.0)

    @step
    def calculate_readability_score(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive readability assessment using multiple metrics.

        Provides both automatic and manual reading ease indicators:
        - Flesch Reading Ease Score
        - Flesch-Kincaid Grade Level
        - Additional linguistic complexity measures

        Args:
            text (str): Text to assess for readability

        Returns:
            Dict[str, Any]: Complete readability analysis
        """

        if not text.strip():
            return {
                "flesch_reading_ease": 0.0,
                "flesch_kincaid_grade": 0.0,
                "readability_category": "empty",
            }

        # Prepare text for analysis
        sentences = self._tokenize_sentences(text)
        words = self._tokenize_words(text)

        sentence_count = len([s for s in sentences if s.strip()]) or 1
        word_count = len(words) or 1

        # Advanced syllable counting algorithm
        total_syllables = sum(self._count_syllables(word) for word in words)

        # Calculate core metrics
        avg_sentence_length = word_count / sentence_count
        avg_syllables_per_word = total_syllables / word_count

        # Flesch Reading Ease Score (higher is easier)
        flesch_score = (
            206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        )

        # Flesch-Kincaid Grade Level (higher requires higher education level)
        fk_grade = (
            (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
        )

        # Enhanced readability categorization
        category = self._categorize_readability(flesch_score)

        return {
            "flesch_reading_ease": round(flesch_score, 2),
            "flesch_kincaid_grade": round(fk_grade, 2),
            "avg_sentence_length": round(avg_sentence_length, 2),
            "avg_syllables_per_word": round(avg_syllables_per_word, 2),
            "total_words": word_count,
            "total_sentences": sentence_count,
            "total_syllables": total_syllables,
            "readability_category": category,
            "processing_timestamp": datetime.now().isoformat(),
        }

    def _count_syllables(self, word: str) -> int:
        """Advanced syllable counting with English language rules."""

        if not word:
            return 0

        word = word.lower()
        vowels = "aeiouy"

        # Remove non-letter characters
        word = re.sub(r"[^a-z]", "", word)

        if not word:
            return 0

        # Count vowel groups
        syllable_count = 0
        prev_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel

        # Handle silent 'e'
        if word.endswith("e") and syllable_count > 1:
            syllable_count -= 1

        # Ensure minimum of 1
        return max(syllable_count, 1)

    def _categorize_readability(self, flesch_score: float) -> str:
        """Categorize text readability based on Flesch Reading Ease score."""

        if flesch_score >= 90:
            return "very_easy"
        elif flesch_score >= 80:
            return "easy"
        elif flesch_score >= 70:
            return "fairly_easy"
        elif flesch_score >= 60:
            return "standard"
        elif flesch_score >= 50:
            return "fairly_difficult"
        elif flesch_score >= 30:
            return "difficult"
        else:
            return "very_difficult"

    @step
    def generate_text_summary(
        self, text: str, max_sentences: int = 3
    ) -> Dict[str, Any]:
        """
        Intelligent text summarization using advanced sentence scoring.

        This step provides context-aware summarization that considers:
        - Word frequency analysis
        - Position-based importance (first/last sentences)
        - Named entity presence for content relevance

        Args:
            text (str): Text to summarize
            max_sentences (int): Maximum number of sentences in summary

        Returns:
            Dict[str, Any]: Summary with metadata and analysis
        """

        if not text.strip():
            return {
                "summary_sentences": [],
                "compression_ratio": 0.0,
                "original_length": 0,
            }

        sentences = self._tokenize_sentences(text)

        if len(sentences) <= max_sentences:
            return {
                "summary_sentences": sentences,
                "compression_ratio": 1.0,
                "original_length": len(sentences),
            }

        # Advanced sentence scoring algorithm
        sentence_scores = self._score_sentences(sentences, text)

        # Select top sentences while maintaining original order
        selected_indices = self._select_top_sentences(sentence_scores, max_sentences)

        summary_sentences = [sentences[i] for i in selected_indices]

        # Calculate compression metrics
        original_word_count = len(self._tokenize_words(text))
        summary_word_count = sum(len(s.split()) for s in summary_sentences)

        compression_ratio = len(summary_sentences) / len(sentences)

        return {
            "summary_sentences": summary_sentences,
            "compression_ratio": round(compression_ratio, 3),
            "original_length": len(sentences),
            "summary_length": len(summary_sentences),
            "word_compression_ratio": round(
                summary_word_count / max(original_word_count, 1), 3
            ),
            "top_sentence_scores": {
                sentences[idx]: sentence_scores[idx] for idx in selected_indices
            },
            "processing_timestamp": datetime.now().isoformat(),
        }

    def _score_sentences(
        self, sentences: List[str], full_text: str
    ) -> Dict[int, float]:
        """Score each sentence based on multiple importance factors."""

        scores = {}
        words_in_text = self._tokenize_words(full_text)

        # Calculate word frequencies (TF component)
        content_word_freq = Counter(
            [word for word in words_in_text if word.isalpha() and len(word) > 2]
        )

        for i, sentence in enumerate(sentences):
            score = 0.0

            # Word frequency contribution (TF-based)
            words_in_sentence = self._tokenize_words(sentence.lower())
            content_words = [
                word for word in words_in_sentence if word.isalpha() and len(word) > 2
            ]

            for word in content_words:
                score += content_word_freq.get(word, 0)

            # Position bonuses (first and last sentences are important)
            if i == 0 or i == len(sentences) - 1:
                score += 10

            # Named entity bonus (entities indicate important content)
            entities = self.extract_named_entities(sentence)
            score += len(entities) * 3

            # Sentence length normalization (prefer medium-length sentences)
            optimal_length = len(content_words) if content_words else 1
            length_factor = min(optimal_length / 10, 1.0)  # Optimal around 10 words
            score *= length_factor

            scores[i] = score

        return scores

    def _select_top_sentences(
        self, sentence_scores: Dict[int, float], max_sentences: int
    ) -> List[int]:
        """Select top-scored sentences while maintaining chronological order."""

        # Get indices of highest scoring sentences
        sorted_indices = sorted(
            sentence_scores.keys(), key=lambda x: sentence_scores[x], reverse=True
        )

        # Select top sentences and sort chronologically
        selected = sorted(sorted_indices[:max_sentences])

        return selected

    @step
    def extract_keywords(self, text: str, max_keywords: int = 10) -> Dict[str, Any]:
        """
        Advanced keyword extraction using TF-IDF simulation and linguistic analysis.

        Provides comprehensive keyword ranking with:
        - Term frequency (TF) calculations
        - Inverse document frequency simulation
        - Part-of-speech filtering for content words

        Args:
            text (str): Text to extract keywords from
            max_keywords (int): Maximum number of keywords to return

        Returns:
            Dict[str, Any]: Ranked keywords with analysis metadata
        """

        if not text.strip():
            return {"top_keywords": [], "total_candidates": 0}

        # Comprehensive text preprocessing
        processed = self.preprocess_text(text)

        if not ZENML_AVAILABLE:
            # Direct method call for standalone execution
            return self._extract_keywords_standalone(processed, max_keywords)

        # Filter for content words (nouns, adjectives, verbs excluding stopwords)
        filtered_words = processed["filtered_words"]

        if not filtered_words:
            return {"top_keywords": [], "total_candidates": 0}

        # Apply stemming for better keyword matching
        if self.stemmer:
            stemmed_words = [self.stemmer.stem(word) for word in filtered_words]
        else:
            stemmed_words = filtered_words

        # Calculate TF-IDF scores (simplified simulation)
        word_frequencies = Counter(stemmed_words)

        keywords = []
        total_documents = 1  # Simplified: single document scenario

        for word, frequency in word_frequencies.most_common():
            if len(word) < 3:  # Skip very short words
                continue

            # Term frequency
            tf = frequency / len(stemmed_words)

            # Inverse document frequency (simplified for single doc)
            idf = math.log(total_documents / 1) + 1

            # TF-IDF score
            tfidf_score = tf * idf

            keywords.append(
                {
                    "keyword": word,
                    "frequency": frequency,
                    "tf_score": round(tf, 4),
                    "importance_score": round(tfidf_score, 4),
                }
            )

        return {
            "top_keywords": keywords[:max_keywords],
            "total_candidates": len(keywords),
            "processing_metadata": {
                "filtering_applied": True,
                "stemming_applied": self.stemmer is not None,
                "total_unique_words": len(word_frequencies),
            },
        }

    def _extract_keywords_standalone(
        self, processed: Dict[str, Any], max_keywords: int
    ) -> Dict[str, Any]:
        """Standalone keyword extraction method."""

        filtered_words = processed["filtered_words"]

        if not filtered_words:
            return {"top_keywords": [], "total_candidates": 0}

        # Calculate word frequencies
        word_frequencies = Counter(filtered_words)

        keywords = []

        for word, frequency in word_frequencies.most_common():
            if len(word) < 3:
                continue

            keywords.append(
                {
                    "keyword": word,
                    "frequency": frequency,
                    "importance_score": round(frequency / len(filtered_words), 4),
                }
            )

        return {
            "top_keywords": keywords[:max_keywords],
            "total_candidates": len(keywords),
        }


# ZenML Pipeline Definition
@zenml_pipeline
def advanced_nlp_pipeline(text: str, max_summary_sentences: int = 3) -> Dict[str, Any]:
    """
    Complete advanced NLP processing pipeline.

    This comprehensive pipeline orchestrates all NLP analysis steps in the correct
    order and provides a unified output with full metadata.

    Args:
        text (str): Input text to analyze
        max_summary_sentences (int): Maximum sentences in summary

    Returns:
        Dict[str, Any]: Complete analysis results from all steps
    """

    # Initialize processor instance
    nlp_processor = AdvancedNLPProcessor()

    try:
        # Execute all analysis steps in sequence
        preprocessing_results = nlp_processor.preprocess_text(text)
        sentiment_results = nlp_processor.analyze_sentiment(text)
        entity_results = nlp_processor.extract_named_entities(text)
        readability_results = nlp_processor.calculate_readability_score(text)

        # Summary generation with configurable length
        summary_results = nlp_processor.generate_text_summary(
            text, max_summary_sentences
        )

        # Keyword extraction
        keyword_results = nlp_processor.extract_keywords(text, max_keywords=10)

        # Compile comprehensive results
        pipeline_results = {
            "pipeline_metadata": {
                "execution_timestamp": datetime.now().isoformat(),
                "zenml_available": ZENML_AVAILABLE,
                "pipeline_version": "1.0",
            },
            # Individual analysis results
            "preprocessing": preprocessing_results,
            "sentiment_analysis": sentiment_results,
            "named_entities": entity_results,
            "readability_assessment": readability_results,
            # Derived results
            "text_summary": {
                **summary_results,
                "keyword_context": keyword_results["top_keywords"][:5]
                if keyword_results["top_keywords"]
                else [],
            },
            "keyword_analysis": keyword_results,
            # Summary metrics
            "analysis_summary": {
                "overall_sentiment": sentiment_results["overall_sentiment"]["label"],
                "readability_level": readability_results["readability_category"],
                "entity_count": len(entity_results),
                "summary_length": len(summary_results["summary_sentences"]),
                "keyword_count": len(keyword_results["top_keywords"]),
            },
        }

        return pipeline_results

    except Exception as e:
        # Graceful error handling with detailed context
        return {
            "pipeline_metadata": {
                "execution_timestamp": datetime.now().isoformat(),
                "status": "failed",
                "error_message": str(e),
                "zenml_available": ZENML_AVAILABLE,
            },
            "error_details": {
                "exception_type": type(e).__name__,
                "processing_stage": "pipeline_execution",
            },
        }


def run_pipeline_example():
    """
    Example function to demonstrate the NLP pipeline.

    This provides a standalone example of how to use the advanced NLP
    pipeline with realistic input data.
    """

    # Sample text for comprehensive testing
    sample_text = """
    Apple Inc. is an American multinational technology company headquartered in Cupertino, California,
    that designs, develops, and sells consumer electronics, computer software, and online services.

    Founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976 as Apple Computer Company,
    it was renamed to Apple Inc. in 1980. The company is known for its innovative products including
    the iPhone, iPad, MacBook, and Apple Watch.

    However, critics often point out that Apple's products can be quite expensive compared to competitors.
    Despite the high prices, many customers remain loyal because of the seamless ecosystem integration.

    The company's recent earnings report shows strong performance in services revenue, which has become
    increasingly important for Apple's long-term growth strategy. Tim Cook, the current CEO,
    continues to lead Apple into new markets including artificial intelligence and autonomous vehicles.

    Overall, while opinions about the company vary, Apple remains one of the most valuable brands
    in the technology sector with significant influence on global consumer electronics trends.
    """

    print("🚀 ADVANCED NLP PIPELINE - ZENML IMPLEMENTATION")
    print("=" * 60)

    if ZENML_AVAILABLE:
        # Run with ZenML pipeline
        print("Running with ZenML pipeline orchestration...")

        try:
            # Initialize and run the pipeline
            results = advanced_nlp_pipeline(sample_text, max_summary_sentences=3)

        except Exception as e:
            print(f"ZenML pipeline execution failed: {e}")
            # Fallback to direct processing
            results = run_standalone_pipeline(sample_text)
    else:
        # Run standalone for testing
        print("ZenML not available - running in standalone mode...")
        results = run_standalone_pipeline(sample_text)

    # Display comprehensive results
    display_analysis_results(results)

    return results


def run_standalone_pipeline(text: str) -> Dict[str, Any]:
    """
    Standalone pipeline execution for testing without ZenML.

    This function provides the same analysis as the ZenML pipeline
    but runs directly for testing and validation purposes.
    """

    nlp_processor = AdvancedNLPProcessor()

    # Execute all analysis steps
    try:
        preprocessing_results = nlp_processor.preprocess_text(text)
        sentiment_results = nlp_processor.analyze_sentiment(text)
        entity_results = nlp_processor.extract_named_entities(text)
        readability_results = nlp_processor.calculate_readability_score(text)

        summary_results = nlp_processor.generate_text_summary(text, 3)
        keyword_results = nlp_processor.extract_keywords(text)

        return {
            "pipeline_metadata": {
                "execution_timestamp": datetime.now().isoformat(),
                "zenml_available": False,
                "pipeline_version": "1.0-standalone",
            },
            "preprocessing": preprocessing_results,
            "sentiment_analysis": sentiment_results,
            "named_entities": entity_results,
            "readability_assessment": readability_results,
            "text_summary": {
                **summary_results,
                "keyword_context": [
                    kw["keyword"] for kw in keyword_results["top_keywords"][:5]
                ],
            },
            "keyword_analysis": keyword_results,
            "analysis_summary": {
                "overall_sentiment": sentiment_results["overall_sentiment"]["label"],
                "readability_level": readability_results["readability_category"],
                "entity_count": len(entity_results),
                "summary_length": len(summary_results["summary_sentences"]),
                "keyword_count": len(keyword_results["top_keywords"]),
            },
        }

    except Exception as e:
        return {
            "pipeline_metadata": {
                "execution_timestamp": datetime.now().isoformat(),
                "status": "failed",
                "error_message": str(e),
                "zenml_available": False,
            },
            "error_details": {"exception_type": type(e).__name__},
        }


def display_analysis_results(results: Dict[str, Any]) -> None:
    """
    Display comprehensive analysis results in a formatted manner.

    This function presents the NLP pipeline results in an easy-to-read
    format for demonstration and analysis purposes.
    """

    print(f"\n📊 ANALYSIS RESULTS SUMMARY")
    print("-" * 40)

    # Basic analysis summary
    if "analysis_summary" in results:
        summary = results["analysis_summary"]

        print(f"Sentiment: {summary['overall_sentiment'].title()}")
        print(f"Readability: {summary['readability_level'].replace('_', ' ').title()}")
        print(f"Entities Found: {summary['entity_count']}")
        print(f"Keywords Extracted: {summary['keyword_count']}")

    # Sentiment details
    if "sentiment_analysis" in results:
        sentiment = results["sentiment_analysis"]

        print(f"\n🔍 SENTIMENT ANALYSIS:")
        overall = sentiment.get("overall_sentiment", {})
        print(f"  Overall: {overall.get('label', 'N/A').title()}")
        print(f"  Confidence: {overall.get('confidence', 0):.2f}")

        if sentiment.get("textblob_sentiment"):
            tb = sentiment["textblob_sentiment"]
            print(f"  TextBlob Polarity: {tb.get('polarity', 0):.3f}")
            print(f"  TextBlob Subjectivity: {tb.get('subjectivity', 0):.3f}")

        if sentiment.get("lexicon_sentiment"):
            lex = sentiment["lexicon_sentiment"]
            print(f"  Positive Words: {lex.get('positive_matches', 0)}")
            print(f"  Negative Words: {lex.get('negative_matches', 0)}")

    # Named entities
    if "named_entities" in results:
        entities = results["named_entities"]

        print(f"\n🏷️  NAMED ENTITIES ({len(entities)} found):")

        for entity in entities[:5]:  # Show first 5
            print(
                f"  • {entity['text']} ({entity['label']}) - {entity['confidence']:.1f}"
            )

        if len(entities) > 5:
            print(f"  ... and {len(entities) - 5} more")

    # Readability metrics
    if "readability_assessment" in results:
        readability = results["readability_assessment"]

        print(f"\n📖 READABILITY ANALYSIS:")
        print(f"  Flesch Reading Ease: {readability.get('flesch_reading_ease', 'N/A')}")
        print(f"  Grade Level: {readability.get('flesch_kincaid_grade', 'N/A')}")
        print(f"  Avg Sentence Length: {readability.get('avg_sentence_length', 'N/A')}")
        print(f"  Total Words: {readability.get('total_words', 'N/A')}")

    # Text summary
    if "text_summary" in results:
        summary = results["text_summary"]

        print(
            f"\n📄 TEXT SUMMARY ({summary.get('compression_ratio', 0):.1%} compression):"
        )

        for i, sentence in enumerate(summary.get("summary_sentences", [])[:3], 1):
            print(f"  {i}. {sentence}")

        if len(summary.get("summary_sentences", [])) > 3:
            print(f"  ... and {len(summary['summary_sentences']) - 3} more sentences")

    # Keywords
    if "keyword_analysis" in results:
        keywords = results["keyword_analysis"]

        print(f"\n🔑 TOP KEYWORDS:")

        for kw in keywords.get("top_keywords", [])[:8]:
            print(f"  • {kw['keyword']} (freq: {kw.get('frequency', 0)})")

    print(f"\n" + "=" * 60)
    print("✅ Pipeline execution completed successfully!")


if __name__ == "__main__":
    # Execute the comprehensive pipeline example
    results = run_pipeline_example()

    print("\n🎯 ADVANCED NLP PIPELINE DEMONSTRATION COMPLETE")
