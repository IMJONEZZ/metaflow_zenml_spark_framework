#!/usr/bin/env python3
"""
Advanced NLP Processing Pipeline (ZenML Version)

This demonstrates sophisticated natural language processing using ZenML:
- Classical statistical analysis with NLTK
- Production-grade NLP with spaCy pipeline
- Enhanced sentiment analysis with TextBlob
- Linguistic pattern mining and insights generation

Usage:
    python nlp_pipeline_advanced.py
"""

import os
import random
from typing import Annotated, Any, Dict, List

try:
    from colorama import Fore, Style, init

    init(autoreset=True)
except ImportError:
    # Fallback if colorama is not available
    class Fore:
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        RED = "\033[91m"
        BLUE = "\033[94m"
        MAGENTA = "\033[95m"
        CYAN = "\033[96m"
        WHITE = "\033[97m"

    class Style:
        RESET_ALL = "\033[0m"
        BRIGHT = "\033[1m"


from zenml import pipeline as zenml_pipeline
from zenml import step
from zenml.types import HTMLString

from utils.nlp_visualization import generate_nlp_html_visualization


@step(enable_cache=False)
def generate_diverse_texts(num_samples: int = 50) -> Annotated[List[str], "texts"]:
    """
    Generate diverse text samples for comprehensive NLP analysis.

    Creates a variety of texts including:
    - Reviews (positive, negative, mixed)
    - News articles
    - Technical descriptions
    - Casual conversations
    """

    # Diverse text samples covering multiple domains
    base_texts = [
        # Product reviews
        "I absolutely love this new smartphone! The camera quality is "
        "incredible and the battery lasts all day.",
        "This restaurant was disappointing. The food arrived cold and "
        "the service was extremely slow.",
        "The latest software update improved performance significantly, "
        "though some features are still buggy.",
        # News-style text
        "Scientists announced a breakthrough in renewable energy technology "
        "that could revolutionize solar power generation.",
        "The city council voted unanimously to approve the new urban "
        "development plan despite community opposition.",
        # Technical content
        "The machine learning algorithm uses deep neural networks to "
        "process natural language with remarkable accuracy.",
        "Our distributed system architecture handles millions of requests "
        "per second with minimal latency.",
        # Personal content
        "I'm planning a trip to Japan next spring. I'm excited about "
        "visiting the temples and trying authentic sushi.",
        "The weather forecast indicates heavy rain this weekend, so I'll "
        "need to reschedule the outdoor picnic.",
        # Opinion pieces
        "While artificial intelligence offers tremendous opportunities, "
        "we must consider the ethical implications carefully.",
        "The rise of remote work has fundamentally changed how companies "
        "approach team collaboration and productivity.",
        # Mixed sentiment
        "The conference was excellent overall, though some presentations "
        "were too technical for beginners.",
        # Short texts
        "Amazing product!",
        "Terrible experience.",
        "Good value for money.",
        # Longer texts
        "The new research paper presents compelling evidence that climate "
        "change acceleration requires immediate policy intervention. The study's "
        "methodology appears sound, though some critics question the sample size.",
    ]

    # Generate variations by modifying base texts
    expanded_texts = []

    for base in base_texts:
        expanded_texts.append(base)

        # Create variations by modifying adjectives and sentiment
        if len(expanded_texts) < num_samples:
            words = base.split()

            # Replace adjectives for sentiment variation
            replacements_pos = {
                "incredible": "outstanding",
                "amazing": "fantastic",
                "excellent": "superb",
            }

            replacements_neg = {
                "disappointing": "terrible",
                "slow": "inadequate",
                "buggy": "problematic",
            }

            for word, replacement in replacements_pos.items():
                if word in base.lower() and len(expanded_texts) < num_samples:
                    new_text = base.replace(word, replacement)
                    if new_text != base:
                        expanded_texts.append(new_text)

            for word, replacement in replacements_neg.items():
                if word in base.lower() and len(expanded_texts) < num_samples:
                    new_text = base.replace(word, replacement)
                    if new_text != base:
                        expanded_texts.append(new_text)

    print(
        Fore.GREEN
        + f"✅ Generated {len(expanded_texts)} diverse text samples for analysis"
    )

    return expanded_texts[:num_samples]


@step(enable_cache=False)
def setup_nlp_libraries() -> Annotated[Dict[str, Any], "library_status"]:
    """
    Setup and download required NLTK data and spaCy models.

    Returns:
        Dictionary containing library status information
    """

    print(Fore.CYAN + "🔧 Setting up NLP libraries and downloading resources...")

    try:
        import nltk

        # Download required NLTK data
        nltk_resources = [
            "punkt_tab",  # Enhanced tokenization (newer NLTK)
            "stopwords",  # Stop words list
            "vader_lexicon",  # Sentiment analysis
            "averaged_perceptron_tagger",  # POS tagging
        ]

        for resource in nltk_resources:
            try:
                print(Fore.BLUE + f"   Downloading {resource}...")
                nltk.download(resource, quiet=True)
            except Exception as e:
                print(Fore.YELLOW + f"   ⚠️ Could not download {resource}: {e}")

        print(Fore.GREEN + "✅ NLTK resources ready")

    except ImportError as e:
        print(Fore.RED + f"❌ NLTK not available: {e}")

    # Test spaCy availability
    try:
        import spacy

        print(Fore.BLUE + "   Testing spaCy availability...")

        # Try to load the English model
        try:
            nlp = spacy.load("en_core_web_sm")
            print(Fore.GREEN + "✅ spaCy English model loaded successfully")

        except OSError:
            print(
                Fore.YELLOW
                + "⚠️ spaCy English model not found. Continuing with basic text processing..."
            )
            spacy.download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")

        library_status = {
            "nltk_available": True,
            "spacy_available": nlp is not None,
            "textblob_available": False,
        }

    except ImportError:
        print(Fore.RED + "❌ spaCy not available")
        library_status = {
            "nltk_available": True,
            "spacy_available": False,
            "textblob_available": False,
        }

    # Test TextBlob availability
    try:
        import textblob

        library_status["textblob_available"] = True
        print(Fore.GREEN + "✅ TextBlob available")

    except ImportError:
        library_status["textblob_available"] = False
        print(Fore.YELLOW + "⚠️ TextBlob not available")

    return library_status


@step(enable_cache=False)
def classical_nlp_analysis(
    texts: List[str], library_status: Dict[str, Any]
) -> Annotated[Dict[str, Any], "classical_results"]:
    """
    Perform classical NLP analysis using NLTK.

    Includes:
    - Tokenization and word frequency analysis
    - Stop words filtering
    - NLTK sentiment analysis (VADER)
    - Basic linguistic statistics

    Args:
        texts: List of text samples to analyze
        library_status: Status of available NLP libraries

    Returns:
        Dictionary containing classical analysis results
    """

    print(Fore.CYAN + "📊 Performing classical NLP statistical analysis...")

    if not library_status.get("nltk_available", False):
        print(Fore.YELLOW + "⚠️ Skipping NLTK analysis - library not available")
        return {}

    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.probability import FreqDist
        from nltk.tokenize import sent_tokenize, word_tokenize

        # Initialize English stop words
        try:
            english_stopwords = set(stopwords.words("english"))
        except LookupError:
            print(Fore.YELLOW + "⚠️ Stopwords not available, proceeding without them")
            english_stopwords = set()

        # Analyze each text
        all_words = []
        sentences = []
        sentiment_scores = []

        for i, text in enumerate(texts):
            # Tokenize with fallback mechanism
            try:
                sentence_tokens = sent_tokenize(text)
                word_tokens = word_tokenize(text.lower())

            except Exception as e:
                print(
                    Fore.YELLOW
                    + f"   ⚠️ NLTK tokenization failed for text {i}, using fallback"
                )
                # Fallback to simple regex-based tokenization
                import re

                # Simple sentence splitting on punctuation
                sentence_tokens = [
                    s.strip() for s in re.split(r"[.!?]+", text) if s.strip()
                ]

                # Simple word tokenization (keep only alphabetic characters)
                word_tokens = re.findall(r"\b[a-zA-Z]+\b", text.lower())

            sentences.extend(sentence_tokens)
            all_words.extend(word_tokens)

            # Filter stop words for frequency analysis (with fallback if NLTK fails)
            try:
                content_words = [
                    word
                    for word in word_tokens
                    if word.isalpha() and word not in english_stopwords
                ]
            except Exception:
                # Fallback stop words list if NLTK stopwords fail
                basic_stopwords = {
                    "the",
                    "a",
                    "an",
                    "and",
                    "or",
                    "but",
                    "in",
                    "on",
                    "at",
                    "to",
                    "for",
                    "of",
                    "with",
                    "by",
                    "is",
                    "was",
                    "are",
                    "were",
                    "be",
                    "been",
                    "have",
                    "has",
                    "had",
                    "do",
                    "does",
                    "did",
                    "will",
                    "would",
                    "could",
                    "should",
                    "may",
                    "might",
                    "this",
                    "that",
                    "these",
                    "those",
                    "i",
                    "you",
                    "he",
                    "she",
                    "it",
                    "we",
                    "they",
                    "me",
                    "him",
                    "her",
                    "us",
                    "them",
                }

                content_words = [
                    word
                    for word in word_tokens
                    if word.isalpha() and len(word) > 1 and word not in basic_stopwords
                ]

            # Simple sentiment analysis
            positive_indicators = [
                "good",
                "great",
                "excellent",
                "amazing",
                "love",
                "fantastic",
                "wonderful",
                "outstanding",
            ]

            negative_indicators = [
                "bad",
                "terrible",
                "awful",
                "hate",
                "disappointing",
                "poor",
                "worst",
                "horrible",
            ]

            text_lower = text.lower()
            pos_count = sum(1 for word in positive_indicators if word in text_lower)
            neg_count = sum(1 for word in negative_indicators if word in text_lower)

            sentiment_score = pos_count - neg_count
            sentiment_scores.append(sentiment_score)

        # Calculate frequency distribution
        freq_dist = FreqDist(all_words) if all_words else {}

        # Compile results
        classical_results = {
            "total_texts": len(texts),
            "total_sentences": len(sentences),
            "total_words": len(all_words),
            "unique_words": len(freq_dist),
            "avg_sentences_per_text": (len(sentences) / len(texts) if texts else 0),
            "avg_words_per_text": (len(all_words) / len(texts) if texts else 0),
            "lexical_diversity": (len(freq_dist) / len(all_words) if all_words else 0),
            "top_frequent_words": dict(freq_dist.most_common(10)),
            "sentiment_distribution": {
                "positive_texts": sum(1 for score in sentiment_scores if score > 0),
                "negative_texts": sum(1 for score in sentiment_scores if score < 0),
                "neutral_texts": sum(1 for score in sentiment_scores if score == 0),
            },
        }

        print(
            Fore.GREEN
            + f"✅ Classical analysis complete - processed {len(texts)} texts"
        )

    except Exception as e:
        print(Fore.RED + f"❌ Classical NLP analysis failed: {e}")
        classical_results = {"error": str(e)}

    return classical_results


@step(enable_cache=False)
def advanced_linguistic_analysis(
    texts: List[str], library_status: Dict[str, Any]
) -> Annotated[Dict[str, Any], "advanced_results"]:
    """
    Perform advanced linguistic analysis using spaCy or fallback methods.

    Includes:
    - Named Entity Recognition (NER) when spaCy available
    - Part-of-Speech tagging
    - Dependency parsing with fallback
    - Document similarity analysis

    Args:
        texts: List of text samples to analyze
        library_status: Status of available NLP libraries

    Returns:
        Dictionary containing advanced analysis results
    """

    print(Fore.CYAN + "🎯 Performing advanced linguistic analysis...")

    if not library_status.get("spacy_available", False):
        print(
            Fore.YELLOW + "⚠️ spaCy not available, using basic linguistic processing..."
        )
        return _basic_linguistic_analysis(texts)

    try:
        import spacy

        # Load spaCy model
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            print(
                Fore.YELLOW
                + "⚠️ spaCy English model not available, using basic processing..."
            )
            return _basic_linguistic_analysis(texts)

        # Analyze texts with spaCy
        entities_found = []
        pos_tags = {}
        dependencies = {}

        for i, text in enumerate(texts):
            try:
                doc = nlp(text)

                # Extract named entities
                for ent in doc.ents:
                    entities_found.append(
                        {
                            "text": ent.text,
                            "label": ent.label_,
                            "description": spacy.explain(ent.label_)
                            or "Unknown entity",
                        }
                    )

                # Count POS tags
                for token in doc:
                    pos = token.pos_
                    pos_tags[pos] = pos_tags.get(pos, 0) + 1

                # Analyze sentence complexity
                for sent in doc.sents:
                    dep_count = len([token for token in sent if token.dep_ != "ROOT"])

                    sentence_key = f"sentence_{len(dependencies)}"
                    dependencies[sentence_key] = {
                        "length": len([t for t in sent]),
                        "complexity": dep_count,
                    }

            except Exception as e:
                print(Fore.YELLOW + f"   ⚠️ Processing failed for text {i}: {e}")

        # Entity type analysis
        entity_types = {}
        for ent in entities_found:
            label = ent["label"]
            entity_types[label] = entity_types.get(label, 0) + 1

        # Sentence complexity statistics
        complexities = [dep["complexity"] for dep in dependencies.values()]

        advanced_results = {
            "total_entities": len(entities_found),
            "unique_entity_types": len(entity_types),
            "entity_type_distribution": entity_types,
            "pos_tag_distribution": pos_tags,
            "sentence_analysis": {
                "total_sentences": len(dependencies),
                "avg_complexity": (
                    sum(complexities) / len(complexities) if complexities else 0
                ),
                "max_complexity": max(complexities) if complexities else 0,
            },
            "sample_entities": entities_found[:10],
        }

        print(
            Fore.GREEN
            + f"✅ Advanced spaCy analysis complete - found {len(entities_found)} entities"
        )

    except Exception as e:
        print(Fore.RED + f"❌ Advanced linguistic analysis failed: {e}")
        return _basic_linguistic_analysis(texts)

    return advanced_results


def _basic_linguistic_analysis(texts: List[str]) -> Dict[str, Any]:
    """
    Basic linguistic analysis using regex and basic NLP techniques.

    Args:
        texts: List of text samples to analyze

    Returns:
        Dictionary containing basic linguistic analysis results
    """

    print(Fore.BLUE + "   Using basic regex-based linguistic analysis...")

    try:
        import re

        # Basic entity detection using regex patterns
        entities_found = []

        # Common patterns for different entity types
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        phone_pattern = r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"
        date_pattern = r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
        number_pattern = r"\b\d+(?:\.\d+)?\b"

        # POS tag approximation using word patterns
        pos_tags = {}

        for text in texts:
            # Extract basic entities
            emails = re.findall(email_pattern, text)
            for email in emails:
                entities_found.append(
                    {"text": email, "label": "EMAIL", "description": "Email address"}
                )

            phones = re.findall(phone_pattern, text)
            for phone in phones:
                entities_found.append(
                    {"text": phone, "label": "PHONE", "description": "Phone number"}
                )

            dates = re.findall(date_pattern, text)
            for date in dates:
                entities_found.append(
                    {"text": date, "label": "DATE", "description": "Date"}
                )

            # Extract numbers (limited to avoid noise)
            all_numbers = re.findall(number_pattern, text)
            for number in all_numbers[:5]:  # Limit to first 5 numbers per text
                entities_found.append(
                    {
                        "text": number,
                        "label": "CARDINAL",
                        "description": "Cardinal number",
                    }
                )

            # Basic POS approximation using word patterns
            words = re.findall(r"\b\w+\b", text.lower())

            for word in words:
                if re.match(r"^[A-Z][a-z]+$", word):
                    pos_tags["PROPN"] = pos_tags.get("PROPN", 0) + 1
                elif word.endswith(("ing", "ed")):
                    pos_tags["VERB"] = pos_tags.get("VERB", 0) + 1
                elif word.endswith("ly"):
                    pos_tags["ADV"] = pos_tags.get("ADV", 0) + 1
                elif word.endswith(("ous", "ful", "able")):
                    pos_tags["ADJ"] = pos_tags.get("ADJ", 0) + 1
                elif word.endswith(("tion", "ment", "ness")):
                    pos_tags["NOUN"] = pos_tags.get("NOUN", 0) + 1
                else:
                    pos_tags["NOUN"] = pos_tags.get("NOUN", 0) + 1

        # Basic sentence analysis
        sentences = []
        for text in texts:
            sentence_splits = re.split(r"[.!?]+", text)
            for sent in sentence_splits:
                if sent.strip():
                    # Basic complexity measure: number of long words
                    words = sent.split()
                    complex_words = [w for w in words if len(w) > 6]

                    sentences.append(
                        {"length": len(words), "complexity": max(1, len(complex_words))}
                    )

        complexities = [s["complexity"] for s in sentences]

        # Entity type analysis
        entity_types = {}
        for ent in entities_found:
            label = ent["label"]
            entity_types[label] = entity_types.get(label, 0) + 1

        basic_results = {
            "total_entities": len(entities_found),
            "unique_entity_types": len(entity_types),
            "entity_type_distribution": entity_types,
            "pos_tag_distribution": pos_tags,
            "sentence_analysis": {
                "total_sentences": len(sentences),
                "avg_complexity": sum(complexities) / len(complexities)
                if complexities
                else 0,
                "max_complexity": max(complexities) if complexities else 0,
            },
            "sample_entities": entities_found[:10],
            "method": "basic_regex_based",
        }

        print(
            Fore.GREEN
            + f"✅ Basic analysis complete - found {len(entities_found)} entities"
        )
        return basic_results

    except Exception as e:
        print(Fore.RED + f"❌ Basic linguistic analysis failed: {e}")
        return {"error": str(e)}


@step(enable_cache=False)
def enhanced_sentiment_analysis(
    texts: List[str], library_status: Dict[str, Any]
) -> Annotated[Dict[str, Any], "sentiment_results"]:
    """
    Perform enhanced sentiment analysis using TextBlob and spaCy.

    Combines multiple approaches:
    - TextBlob polarity and subjectivity
    - SpaCy-based sentiment indicators
    - Comparative analysis

    Args:
        texts: List of text samples to analyze
        library_status: Status of available NLP libraries

    Returns:
        Dictionary containing sentiment analysis results
    """

    print(Fore.CYAN + "😊 Performing enhanced sentiment analysis...")

    textblob_available = library_status.get("textblob_available", False)
    spacy_available = library_status.get("spacy_available", False)

    if not (textblob_available or spacy_available):
        print(Fore.YELLOW + "⚠️ Skipping sentiment analysis - no libraries available")
        return {}

    try:
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer

        # Use NLTK VADER as baseline
        sia = SentimentIntensityAnalyzer()

        sentiment_results = []

        if textblob_available:
            from textblob import TextBlob

            for text in texts:
                # TextBlob analysis
                blob = TextBlob(text)

                # NLTK VADER analysis
                nltk_scores = sia.polarity_scores(text)

                sentiment_results.append(
                    {
                        "text": text,
                        "textblob_polarity": blob.sentiment.polarity,
                        "textblob_subjectivity": blob.sentiment.subjectivity,
                        "nltk_compound": nltk_scores["compound"],
                        "nltk_positive": nltk_scores["pos"],
                        "nltk_negative": nltk_scores["neg"],
                        "nltk_neutral": nltk_scores["neu"],
                    }
                )

        else:
            # Only NLTK available
            for text in texts:
                nltk_scores = sia.polarity_scores(text)

                sentiment_results.append(
                    {
                        "text": text,
                        "nltk_compound": nltk_scores["compound"],
                        "nltk_positive": nltk_scores["pos"],
                        "nltk_negative": nltk_scores["neg"],
                        "nltk_neutral": nltk_scores["neu"],
                    }
                )

        # Calculate aggregate statistics
        total_texts = len(sentiment_results)

        if textblob_available:
            polarities = [result["textblob_polarity"] for result in sentiment_results]

            subjectivities = [
                result["textblob_subjectivity"] for result in sentiment_results
            ]

            avg_polarity = sum(polarities) / len(polarities)
            avg_subjectivity = sum(subjectivities) / len(subjectivities)

        compound_scores = [result["nltk_compound"] for result in sentiment_results]

        avg_compound = sum(compound_scores) / len(compound_scores)

        # Sentiment classification
        positive_texts = sum(1 for score in compound_scores if score > 0.05)

        negative_texts = sum(1 for score in compound_scores if score < -0.05)

        neutral_texts = total_texts - positive_texts - negative_texts

        enhanced_results = {
            "total_analyzed": total_texts,
            "sentiment_distribution": {
                "positive": positive_texts,
                "negative": negative_texts,
                "neutral": neutral_texts,
            },
            "confidence_metrics": {
                "avg_compound_score": avg_compound if compound_scores else 0,
            },
            "textblob_metrics": {
                "avg_polarity": avg_polarity if textblob_available else 0,
                "avg_subjectivity": avg_subjectivity if textblob_available else 0,
            },
            "sample_analysis": sentiment_results[:5],
        }

        print(
            Fore.GREEN
            + f"✅ Enhanced sentiment analysis complete - {total_texts} texts processed"
        )

    except Exception as e:
        print(Fore.RED + f"❌ Enhanced sentiment analysis failed: {e}")
        enhanced_results = {"error": str(e)}

    return enhanced_results


@step(enable_cache=False)
def generate_nlp_insights(
    classical_results: Dict[str, Any],
    advanced_results: Dict[str, Any],
    sentiment_results: Dict[str, Any],
) -> Annotated[List[str], "insights"]:
    """
    Generate comprehensive insights from all NLP analysis results.

    Synthesizes findings across:
    - Classical statistical analysis
    - Advanced linguistic features
    - Enhanced sentiment assessment

    Args:
        classical_results: Results from NLTK analysis
        advanced_results: Results from spaCy processing
        sentiment_results: Enhanced sentiment findings

    Returns:
        List of insights and recommendations
    """

    print(Fore.CYAN + "💡 Generating comprehensive NLP insights...")

    insights = []

    # Classical analysis insights
    if "error" not in classical_results:
        total_texts = classical_results.get("total_texts", 0)
        lexical_diversity = classical_results.get("lexical_diversity", 0)

        insights.append(
            f"📊 Analyzed {total_texts} texts with "
            f"{lexical_diversity:.3f} lexical diversity score"
        )

        if lexical_diversity > 0.5:
            insights.append(
                "📚 High vocabulary variety indicates rich, diverse content"
            )
        elif lexical_diversity < 0.2:
            insights.append(
                "⚠️ Low vocabulary variety suggests repetitive language patterns"
            )

    # Advanced analysis insights
    if "error" not in advanced_results:
        total_entities = advanced_results.get("total_entities", 0)

        if total_entities > 0:
            insights.append(
                f"🏷️ Discovered {total_entities} named entities (PERSON, ORG, GPE, etc.)"
            )

            entity_dist = advanced_results.get("entity_type_distribution", {})
            most_common_entity = (
                max(entity_dist.items(), key=lambda x: x[1]) if entity_dist else None
            )

            if most_common_entity:
                insights.append(
                    f"🎯 Most frequent entity type: {most_common_entity[0]}"
                )

    # Sentiment insights
    if "error" not in sentiment_results:
        sent_dist = sentiment_results.get("sentiment_distribution", {})

        if any(sent_dist.values()):
            total = sum(sent_dist.values())

            if total > 0:
                pos_pct = (sent_dist.get("positive", 0) / total) * 100
                neg_pct = (sent_dist.get("negative", 0) / total) * 100

                insights.append(
                    f"😊 Sentiment breakdown: {pos_pct:.1f}% positive, "
                    f"{neg_pct:.1f}% negative"
                )

                avg_compound = sentiment_results.get("confidence_metrics", {}).get(
                    "avg_compound_score", 0
                )

                if avg_compound > 0.1:
                    insights.append("✅ Overall positive sentiment detected")
                elif avg_compound < -0.1:
                    insights.append("📉 Overall negative sentiment detected")
                else:
                    insights.append("⚖️ Neutral overall sentiment balance")

    # Technical quality assessment
    if "error" in classical_results or "error" in advanced_results:
        insights.append(
            "⚠️ Some analyses encountered issues - "
            "check library availability and dependencies"
        )

    # Performance recommendations
    insights.extend(
        [
            "🔧 For production use, consider model caching and batch processing",
            "📈 NLP quality improves with domain-specific training data",
            "🎯 Combine multiple analysis approaches for robust insights",
        ]
    )

    print(Fore.GREEN + f"✅ Generated {len(insights)} insights and recommendations")

    return insights


@step(enable_cache=False)
def generate_html_report(
    library_status: Dict[str, Any],
    classical_results: Dict[str, Any],
    advanced_results: Dict[str, Any],
    sentiment_results: Dict[str, Any],
    insights: List[str],
) -> Annotated[HTMLString, "nlp_report"]:
    """
    Generate an HTML visualization report from all NLP analysis results.
    
    Args:
        library_status: Status of available libraries
        classical_results: Classical NLP analysis findings
        advanced_results: Advanced linguistic features
        sentiment_results: Sentiment analysis outcomes
        insights: Generated insights and recommendations
        
    Returns:
        HTMLString visualization for ZenML dashboard
    """
    html_content = generate_nlp_html_visualization(
        library_status,
        classical_results,
        advanced_results,
        sentiment_results,
        insights,
    )
    return HTMLString(html_content)


@step(enable_cache=False)
def display_comprehensive_results(
    library_status: Dict[str, Any],
    classical_results: Dict[str, Any],
    advanced_results: Dict[str, Any],
    sentiment_results: Dict[str, Any],
    insights: List[str],
) -> None:
    """
    Display comprehensive results from all NLP analysis steps.

    Args:
        library_status: Status of available libraries
        classical_results: Classical NLP analysis findings
        advanced_results: Advanced linguistic features
        sentiment_results: Sentiment analysis outcomes
        insights: Generated insights and recommendations
    """

    print(
        Fore.WHITE
        + """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║        🎉 ADVANCED NLP ANALYSIS PIPELINE COMPLETED! 🎉       ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
        """
    )

    # Library status summary
    print(Fore.MAGENTA + "🔧 LIBRARY STATUS:")

    nltk_status = (
        "✅ Available"
        if library_status.get("nltk_available", False)
        else "❌ Not available"
    )

    spacy_status = (
        "✅ Available"
        if library_status.get("spacy_available", False)
        else "❌ Not available"
    )

    textblob_status = (
        "✅ Available"
        if library_status.get("textblob_available", False)
        else "❌ Not available"
    )

    print(Fore.CYAN + f"   • NLTK: {nltk_status}")
    print(Fore.CYAN + f"   • spaCy: {spacy_status}")
    print(Fore.CYAN + f"   • TextBlob: {textblob_status}")

    # Classical analysis summary
    print(Fore.MAGENTA + "\n📊 CLASSICAL NLP ANALYSIS:")

    if "error" not in classical_results:
        print(
            Fore.GREEN
            + f"   ✅ Processed {classical_results.get('total_texts', 0)} texts"
        )

        print(
            Fore.CYAN + f"   • Total words: {classical_results.get('total_words', 0):,}"
        )

        print(
            Fore.CYAN
            + f"   • Unique words: {classical_results.get('unique_words', 0):,}"
        )

        print(
            Fore.CYAN + f"   • Lexical diversity: "
            f"{classical_results.get('lexical_diversity', 0):.3f}"
        )

        sent_dist = classical_results.get("sentiment_distribution", {})

        print(
            Fore.CYAN
            + f"   • Sentiment: {sent_dist.get('positive_texts', 0)} positive, "
            f"{sent_dist.get('negative_texts', 0)} negative"
        )

    else:
        print(Fore.RED + f"   ❌ Analysis failed: {classical_results['error']}")

    # Advanced analysis summary
    print(Fore.MAGENTA + "\n🎯 ADVANCED LINGUISTIC ANALYSIS:")

    if "error" not in advanced_results:
        print(
            Fore.GREEN
            + f"   ✅ Found {advanced_results.get('total_entities', 0)} named entities"
        )

        sentence_analysis = advanced_results.get("sentence_analysis", {})

        print(
            Fore.CYAN
            + f"   • Sentences analyzed: {sentence_analysis.get('total_sentences', 0)}"
        )

        print(
            Fore.CYAN + f"   • Average sentence complexity: "
            f"{sentence_analysis.get('avg_complexity', 0):.1f} dependencies"
        )

    else:
        print(Fore.RED + f"   ❌ Analysis failed: {advanced_results['error']}")

    # Enhanced sentiment summary
    print(Fore.MAGENTA + "\n😊 ENHANCED SENTIMENT ANALYSIS:")

    if "error" not in sentiment_results:
        sent_dist = sentiment_results.get("sentiment_distribution", {})

        print(
            Fore.GREEN
            + f"   ✅ Analyzed {sentiment_results.get('total_analyzed', 0)} texts"
        )

        if any(sent_dist.values()):
            total = sum(sent_dist.values())

            print(
                Fore.CYAN + f"   • Positive: {sent_dist.get('positive', 0)} "
                f"({(sent_dist.get('positive', 0) / total) * 100:.1f}%)"
            )

            print(
                Fore.CYAN + f"   • Negative: {sent_dist.get('negative', 0)} "
                f"({(sent_dist.get('negative', 0) / total) * 100:.1f}%)"
            )

            print(
                Fore.CYAN + f"   • Neutral: {sent_dist.get('neutral', 0)} "
                f"({(sent_dist.get('neutral', 0) / total) * 100:.1f}%)"
            )

        conf_metrics = sentiment_results.get("confidence_metrics", {})
        avg_compound = conf_metrics.get("avg_compound_score", 0)

        if abs(avg_compound) > 0.1:
            sentiment_direction = "positive" if avg_compound > 0 else "negative"

            print(
                Fore.CYAN + f"   • Overall sentiment: {sentiment_direction} "
                f"(confidence: {abs(avg_compound):.3f})"
            )

    else:
        print(Fore.RED + f"   ❌ Analysis failed: {sentiment_results['error']}")

    # Insights and recommendations
    print(Fore.MAGENTA + "\n💡 INSIGHTS & RECOMMENDATIONS:")

    for i, insight in enumerate(insights):
        print(Fore.BLUE + f"   {i + 1}. {insight}")

    # Success message
    print(
        Fore.GREEN
        + "\n🎯 Your advanced NLP processing pipeline completed successfully!"
    )


@zenml_pipeline
def advanced_nlp_processing_pipeline(num_samples: int = 50) -> None:
    """
    Execute the complete advanced NLP processing pipeline.

    Performs comprehensive natural language analysis using multiple
    libraries and approaches to provide deep linguistic insights.

    Args:
        num_samples: Number of text samples to generate and analyze
    """

    print(
        Fore.WHITE
        + """
                    ┌─────────────────────────────────────┐
                    │                                     │
                    │  🧠 ADVANCED NATURAL LANGUAGE       │
                    │     PROCESSING PIPELINE             │
                    │                                     │
                    └─────────────────────────────────────┘
        """
    )

    print(Fore.BLUE + f"Processing {num_samples} text samples with advanced NLP...")

    # Execute pipeline steps
    texts = generate_diverse_texts(num_samples)
    library_status = setup_nlp_libraries()

    classical_results = classical_nlp_analysis(texts, library_status)
    advanced_results = advanced_linguistic_analysis(texts, library_status)
    sentiment_results = enhanced_sentiment_analysis(texts, library_status)

    insights = generate_nlp_insights(
        classical_results, advanced_results, sentiment_results
    )

    # Generate HTML visualization report
    html_report = generate_html_report(
        library_status, classical_results, advanced_results, sentiment_results, insights
    )

    # Display comprehensive results
    display_comprehensive_results(
        library_status, classical_results, advanced_results, sentiment_results, insights
    )


if __name__ == "__main__":
    # Running the advanced NLP pipeline locally via ZenML
    advanced_nlp_processing_pipeline(num_samples=30)
