#!/usr/bin/env python3
"""
Standalone Advanced NLP Pipeline

A comprehensive, framework-agnostic NLP processing pipeline that provides:
- Text preprocessing and cleaning
- Advanced sentiment analysis
- Named entity recognition simulation
- Readability scoring
- Text summarization
- Keyword extraction and frequency analysis

This implementation uses only standard NLP libraries (NLTK, TextBlob)
and works without any ML framework dependencies.
"""

import math
import re
import string
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

# Import our tested NLP libraries
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
    A comprehensive NLP processor that provides advanced text analysis capabilities.

    Features:
    - Text preprocessing and normalization
    - Multi-source sentiment analysis
    - Simulated named entity recognition
    - Readability assessment
    - Intelligent summarization
    - Keyword extraction and frequency analysis
    """

    def __init__(self):
        """Initialize the NLP processor with necessary components."""
        self.stemmer = PorterStemmer() if NLTK_AVAILABLE else None
        self.stop_words = (
            set(stopwords.words("english"))
            if NLTK_AVAILABLE
            else self._get_basic_stopwords()
        )

    def _get_basic_stopwords(self) -> set:
        """Provide basic stopwords if NLTK is not available."""
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
        }

    def preprocess_text(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive text preprocessing and feature extraction.

        Args:
            text (str): Raw input text to preprocess

        Returns:
            Dict[str, Any]: Preprocessed text and metadata
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        # Basic cleaning
        cleaned = text.strip()
        cleaned = re.sub(r"\s+", " ", cleaned)  # Normalize whitespace

        # Character-level analysis
        char_count = len(cleaned)
        word_count = (
            len(word_tokenize(cleaned)) if NLTK_AVAILABLE else len(cleaned.split())
        )
        sentence_count = (
            len(sent_tokenize(cleaned))
            if NLTK_AVAILABLE
            else cleaned.count(".") + cleaned.count("!") + cleaned.count("?")
        )

        # Extract words and filter
        if NLTK_AVAILABLE:
            words = word_tokenize(cleaned.lower())
        else:
            words = re.findall(r"\b\w+\b", cleaned.lower())

        # Remove punctuation and stopwords
        filtered_words = [
            word for word in words if word.isalpha() and word not in self.stop_words
        ]

        # Generate metadata
        return {
            "original_text": text,
            "cleaned_text": cleaned,
            "word_count": word_count,
            "char_count": char_count,
            "sentence_count": sentence_count,
            "avg_word_length": sum(len(word) for word in words) / len(words)
            if words
            else 0,
            "filtered_words": filtered_words,
            "unique_word_count": len(set(filtered_words)),
        }

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Perform comprehensive sentiment analysis using multiple approaches.

        Args:
            text (str): Text to analyze for sentiment

        Returns:
            Dict[str, Any]: Detailed sentiment analysis results
        """
        if not text.strip():
            return {"overall_sentiment": "neutral", "confidence": 0.0}

        results = {}

        # TextBlob sentiment analysis (if available)
        if TEXTBLOB_AVAILABLE:
            try:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity

                # Determine sentiment category
                if polarity > 0.1:
                    sentiment_label = "positive"
                elif polarity < -0.1:
                    sentiment_label = "negative"
                else:
                    sentiment_label = "neutral"

                results["textblob_sentiment"] = {
                    "polarity": polarity,
                    "subjectivity": subjectivity,
                    "label": sentiment_label,
                }
            except Exception as e:
                results["textblob_error"] = str(e)

        # Simple lexicon-based sentiment (fallback/additional analysis)
        positive_words = {
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
        }
        negative_words = {
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
        }

        words = word_tokenize(text.lower()) if NLTK_AVAILABLE else text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)

        lexicon_score = (positive_count - negative_count) / max(len(words), 1)
        results["lexicon_sentiment"] = {
            "positive_words": positive_count,
            "negative_words": negative_count,
            "score": lexicon_score,
        }

        # Overall sentiment determination
        if "textblob_sentiment" in results:
            textblob_polarity = results["textblob_sentiment"]["polarity"]
        else:
            textblob_polarity = 0

        final_sentiment = (textblob_polarity + lexicon_score) / 2

        if final_sentiment > 0.05:
            overall_label = "positive"
        elif final_sentiment < -0.05:
            overall_label = "negative"
        else:
            overall_label = "neutral"

        results["overall_sentiment"] = {
            "label": overall_label,
            "confidence": abs(final_sentiment),
            "score": final_sentiment,
        }

        return results

    def extract_named_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities using pattern-based approach (since spaCy model isn't available).

        Args:
            text (str): Text to analyze for named entities

        Returns:
            List[Dict[str, Any]]: Extracted entities with confidence scores
        """
        if not text.strip():
            return []

        sentences = sent_tokenize(text) if NLTK_AVAILABLE else re.split(r"[.!?]+", text)
        entities = []

        for sentence in sentences:
            # Pattern-based entity detection
            patterns = {
                "organization": [
                    r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|Corp|LLC|Company|Co\.)\b"
                ],
                "person": [r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b"],
                "location": [r"\b(?:in|at|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"],
                "date": [
                    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b"
                ],
            }

            for entity_type, regex_patterns in patterns.items():
                for pattern in regex_patterns:
                    matches = re.finditer(pattern, sentence)
                    for match in matches:
                        entity_text = (
                            match.group(1) if len(match.groups()) > 0 else match.group()
                        )
                        entities.append(
                            {
                                "text": entity_text.strip(),
                                "label": entity_type,
                                "confidence": 0.7,  # Pattern-based confidence
                                "context": sentence.strip(),
                            }
                        )

        return entities

    def calculate_readability_score(self, text: str) -> Dict[str, Any]:
        """
        Calculate various readability metrics for the given text.

        Args:
            text (str): Text to analyze for readability

        Returns:
            Dict[str, Any]: Readability analysis results
        """
        if not text.strip():
            return {"grade_level": 0, "readability_category": "empty"}

        # Prepare text analysis
        sentences = sent_tokenize(text) if NLTK_AVAILABLE else re.split(r"[.!?]+", text)
        words = word_tokenize(text) if NLTK_AVAILABLE else re.findall(r"\b\w+\b", text)

        sentence_count = len([s for s in sentences if s.strip()]) or 1
        word_count = len(words) or 1

        # Calculate syllables (approximation)
        def count_syllables(word):
            word = word.lower()
            vowels = "aeiouy"
            syllable_count = 0
            prev_was_vowel = False

            for char in word:
                if char in vowels:
                    if not prev_was_vowel:
                        syllable_count += 1
                    prev_was_vowel = True
                else:
                    prev_was_vowel = False

            # Handle silent 'e'
            if word.endswith("e") and syllable_count > 1:
                syllable_count -= 1

            return max(syllable_count, 1)

        total_syllables = sum(count_syllables(word) for word in words)

        # Calculate Flesch Reading Ease
        avg_sentence_length = word_count / sentence_count
        avg_syllables_per_word = total_syllables / word_count

        flesch_score = (
            206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        )

        # Calculate Flesch-Kincaid Grade Level
        fk_grade = (
            (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
        )

        # Determine readability category
        if flesch_score >= 90:
            category = "very_easy"
        elif flesch_score >= 80:
            category = "easy"
        elif flesch_score >= 70:
            category = "fairly_easy"
        elif flesch_score >= 60:
            category = "standard"
        elif flesch_score >= 50:
            category = "fairly_difficult"
        elif flesch_score >= 30:
            category = "difficult"
        else:
            category = "very_difficult"

        return {
            "flesch_reading_ease": round(flesch_score, 2),
            "flesch_kincaid_grade": round(fk_grade, 2),
            "avg_sentence_length": round(avg_sentence_length, 2),
            "avg_syllables_per_word": round(avg_syllables_per_word, 2),
            "readability_category": category,
            "total_words": word_count,
            "total_sentences": sentence_count,
        }

    def generate_text_summary(self, text: str, max_sentences: int = 3) -> List[str]:
        """
        Generate an intelligent summary by extracting key sentences.

        Args:
            text (str): Text to summarize
            max_sentences (int): Maximum number of sentences in summary

        Returns:
            List[str]: Summary sentences
        """
        if not text.strip():
            return []

        # Tokenize into sentences
        sentences = sent_tokenize(text) if NLTK_AVAILABLE else re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= max_sentences:
            return sentences

        # Score sentences based on key factors
        sentence_scores = {}

        for i, sentence in enumerate(sentences):
            score = 0

            # Word frequency scoring
            if NLTK_AVAILABLE:
                words = word_tokenize(sentence.lower())
            else:
                words = re.findall(r"\b\w+\b", sentence.lower())

            # Filter content words (non-stopwords)
            content_words = [
                word for word in words if word.isalpha() and word not in self.stop_words
            ]

            # Count word frequencies
            word_freq = Counter(content_words)

            # Score based on word frequency
            for word in set(content_words):
                score += word_freq[word]

            # Bonus for position (first and last sentences are important)
            if i == 0 or i == len(sentences) - 1:
                score += 5

            # Bonus for sentences with proper nouns (entities)
            proper_nouns = len(re.findall(r"\b[A-Z][a-z]+\b", sentence))
            score += proper_nouns * 2

            # Normalize by sentence length
            if len(content_words) > 0:
                score = score / len(content_words)

            sentence_scores[i] = score

        # Select top sentences
        top_sentences_indices = sorted(
            sentence_scores.keys(), key=lambda x: sentence_scores[x], reverse=True
        )[:max_sentences]
        top_sentences_indices.sort()  # Maintain original order

        summary = [sentences[i] for i in top_sentences_indices]

        return summary

    def extract_keywords(
        self, text: str, max_keywords: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Extract and rank keywords from the text.

        Args:
            text (str): Text to extract keywords from
            max_keywords (int): Maximum number of keywords to return

        Returns:
            List[Dict[str, Any]]: Keywords with scores
        """
        if not text.strip():
            return []

        # Get preprocessed data
        processed = self.preprocess_text(text)
        filtered_words = processed["filtered_words"]

        if NLTK_AVAILABLE:
            # Use stemming for better keyword matching
            stemmed_words = [self.stemmer.stem(word) for word in filtered_words]
        else:
            stemmed_words = filtered_words

        # Calculate word frequencies
        word_freq = Counter(stemmed_words)

        # TF-IDF simulation (simplified version without document corpus)
        keywords = []
        total_words = len(stemmed_words)

        for word, frequency in word_freq.most_common(
            max_keywords * 2
        ):  # Get more than needed for filtering
            if len(word) < 3:  # Skip very short words
                continue

            tf = frequency / total_words
            idf = 1.0  # Simplified - in real implementation would use corpus

            score = tf * idf
            keywords.append(
                {
                    "keyword": word,
                    "frequency": frequency,
                    "tf_score": round(tf, 4),
                    "importance_score": round(score, 4),
                }
            )

        return keywords[:max_keywords]

    def comprehensive_analysis(self, text: str) -> Dict[str, Any]:
        """
        Perform a complete NLP analysis of the input text.

        Args:
            text (str): Input text to analyze

        Returns:
            Dict[str, Any]: Complete analysis results
        """
        if not text.strip():
            return {"error": "Empty input text"}

        try:
            # Run all analyses
            preprocessing = self.preprocess_text(text)
            sentiment_analysis = self.analyze_sentiment(text)
            entities = self.extract_named_entities(text)
            readability = self.calculate_readability_score(text)
            summary = self.generate_text_summary(text, max_sentences=3)
            keywords = self.extract_keywords(text, max_keywords=10)

            return {
                "input_metadata": {
                    "text_length": len(text),
                    "word_count": preprocessing["word_count"],
                    "sentence_count": preprocessing["sentence_count"],
                },
                "preprocessing_results": {
                    "cleaned_text_length": len(preprocessing["cleaned_text"]),
                    "unique_words": preprocessing["unique_word_count"],
                    "avg_word_length": preprocessing["avg_word_length"],
                },
                "sentiment_analysis": sentiment_analysis,
                "named_entities": entities,
                "readability_assessment": readability,
                "text_summary": {
                    "summary_sentences": summary,
                    "compression_ratio": len(summary)
                    / max(preprocessing["sentence_count"], 1),
                },
                "keyword_extraction": {
                    "top_keywords": keywords,
                    "total_unique_keywords": len(keywords),
                },
                "processing_timestamp": str(__import__("datetime").datetime.now()),
            }

        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}

    def batch_process_texts(self, texts: List[str]) -> Dict[int, Dict[str, Any]]:
        """
        Process multiple texts in batch mode.

        Args:
            texts (List[str]): List of input texts to process

        Returns:
            Dict[int, Dict[str, Any]]: Results indexed by text position
        """
        results = {}

        for i, text in enumerate(texts):
            try:
                results[i] = self.comprehensive_analysis(text)
            except Exception as e:
                results[i] = {"error": f"Failed to process text {i}: {str(e)}"}

        return results


# Utility functions for backward compatibility
def preprocess_text(text: str) -> Dict[str, Any]:
    """Backward-compatible preprocessing function."""
    processor = AdvancedNLPProcessor()
    return processor.preprocess_text(text)


def analyze_sentiment(text: str) -> Dict[str, Any]:
    """Backward-compatible sentiment analysis function."""
    processor = AdvancedNLPProcessor()
    return processor.analyze_sentiment(text)


def extract_named_entities(text: str) -> List[Dict[str, Any]]:
    """Backward-compatible NER function."""
    processor = AdvancedNLPProcessor()
    return processor.extract_named_entities(text)


def calculate_readability_score(text: str) -> Dict[str, Any]:
    """Backward-compatible readability function."""
    processor = AdvancedNLPProcessor()
    return processor.calculate_readability_score(text)


def generate_text_summary(text: str, max_sentences: int = 3) -> List[str]:
    """Backward-compatible summary function."""
    processor = AdvancedNLPProcessor()
    return processor.generate_text_summary(text, max_sentences)


def extract_keywords_simple(text: str) -> List[str]:
    """Simple keyword extraction for basic usage."""
    processor = AdvancedNLPProcessor()
    keywords = processor.extract_keywords(text)
    return [kw["keyword"] for kw in keywords]


def main():
    """Demo function showing how to use the advanced NLP pipeline."""

    # Sample text for demonstration
    sample_text = """
    Apple Inc. is an amazing technology company founded by Steve Jobs in Cupertino, California.
    They make incredible products like the iPhone and MacBook. However, their prices can be
    quite expensive for many consumers. The company has grown tremendously over the years and
    continues to innovate in the smartphone industry. Their latest earnings report shows strong
    performance despite economic challenges.
    """

    print("🚀 ADVANCED NLP PIPELINE DEMO")
    print("=" * 50)

    # Initialize processor
    processor = AdvancedNLPProcessor()

    # Run comprehensive analysis
    results = processor.comprehensive_analysis(sample_text)

    print(f"📊 Analysis Results:")
    print(f"Sentiment: {results['sentiment_analysis']['overall_sentiment']['label']}")
    print(f"Readability: {results['readability_assessment']['readability_category']}")
    print(f"Entities Found: {len(results['named_entities'])}")
    print(f"Summary Sentences: {len(results['text_summary']['summary_sentences'])}")
    print(
        f"Top Keywords: {[kw['keyword'] for kw in results['keyword_extraction']['top_keywords'][:5]]}"
    )

    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
