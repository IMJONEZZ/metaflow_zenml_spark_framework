#!/usr/bin/env python3
"""
Core NLP Functionality Test Script

This script tests the essential NLP libraries and functionality
to ensure our advanced pipeline components work correctly.
"""

import sys
import traceback
from typing import Any, Dict


def test_nltk() -> Dict[str, Any]:
    """Test NLTK functionality"""
    results = {"library": "NLTK", "status": "unknown", "tests": {}}

    try:
        import nltk

        results["status"] = "imported"

        # Test basic tokenization
        from nltk.tokenize import word_tokenize, sent_tokenize

        test_text = "Hello world! This is a test sentence."

        words = word_tokenize(test_text)
        sentences = sent_tokenize(test_text)

        results["tests"]["tokenization"] = {
            "status": "passed",
            "words": words,
            "sentences": sentences,
        }

        # Test stemming (need to download punkt)
        try:
            stemmer = nltk.PorterStemmer()
            stemmed_words = [stemmer.stem(word) for word in words if word.isalpha()]

            results["tests"]["stemming"] = {
                "status": "passed",
                "original": words,
                "stemmed": stemmed_words,
            }
        except Exception as e:
            results["tests"]["stemming"] = {"status": "failed", "error": str(e)}

        # Test POS tagging
        try:
            pos_tags = nltk.pos_tag(words)
            results["tests"]["pos_tagging"] = {
                "status": "passed",
                "tags": pos_tags[:5],  # First 5 tags
            }
        except Exception as e:
            results["tests"]["pos_tagging"] = {"status": "failed", "error": str(e)}

        results["status"] = "working"

    except ImportError as e:
        results["status"] = "import_failed"
        results["error"] = str(e)
    except Exception as e:
        results["status"] = "runtime_error"
        results["error"] = str(e)

    return results


def test_textblob() -> Dict[str, Any]:
    """Test TextBlob functionality"""
    results = {"library": "TextBlob", "status": "unknown", "tests": {}}

    try:
        from textblob import TextBlob

        results["status"] = "imported"

        test_text = "I love natural language processing! It's amazing."
        blob = TextBlob(test_text)

        # Test basic properties
        results["tests"]["basic_properties"] = {
            "status": "passed",
            "sentiment_polarity": blob.sentiment.polarity,
            "sentiment_subjectivity": blob.sentiment.subjectivity,
            "noun_phrases_count": len(blob.noun_phrases),
        }

        # Test word counts
        results["tests"]["word_counts"] = {
            "status": "passed",
            "word_count": len(blob.words),
            "sentence_count": len(blob.sentences),
        }

        # Test spell check
        try:
            corrected = blob.correct()
            results["tests"]["spell_check"] = {
                "status": "passed",
                "original": str(blob),
                "corrected": str(corrected),
            }
        except Exception as e:
            results["tests"]["spell_check"] = {"status": "failed", "error": str(e)}

        results["status"] = "working"

    except ImportError as e:
        results["status"] = "import_failed"
        results["error"] = str(e)
    except Exception as e:
        results["status"] = "runtime_error"
        results["error"] = str(e)

    return results


def test_spacy() -> Dict[str, Any]:
    """Test spaCy functionality"""
    results = {"library": "spaCy", "status": "unknown", "tests": {}}

    try:
        import spacy

        results["status"] = "imported"

        # Try to load the English model
        try:
            nlp = spacy.load("en_core_web_sm")

            test_text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
            doc = nlp(test_text)

            # Test basic entity recognition
            entities = [(ent.text, ent.label_) for ent in doc.ents]

            results["tests"]["entity_recognition"] = {
                "status": "passed",
                "entities_found": len(entities),
                "sample_entities": entities[:3],
            }

            # Test POS tagging
            pos_tags = [
                (token.text, token.pos_, token.tag_)
                for token in doc
                if not token.is_space
            ]

            results["tests"]["pos_tagging"] = {
                "status": "passed",
                "tokens_analyzed": len([t for t in doc if not t.is_space]),
                "sample_tags": pos_tags[:5],
            }

            # Test dependency parsing
            dependencies = [
                (token.text, token.dep_, token.head.text)
                for token in doc
                if not token.is_space
            ]

            results["tests"]["dependency_parsing"] = {
                "status": "passed",
                "sample_deps": dependencies[:3],
            }

            results["status"] = "working"

        except OSError as e:
            # Model not available
            results["status"] = "model_missing"
            results["error"] = str(e)

    except ImportError as e:
        results["status"] = "import_failed"
        results["error"] = str(e)
    except Exception as e:
        results["status"] = "runtime_error"
        results["error"] = str(e)

    return results


def test_advanced_pipeline_functionality() -> Dict[str, Any]:
    """Test the specific functionality our advanced pipeline will use"""
    results = {
        "feature": "Advanced Pipeline Functions",
        "status": "unknown",
        "tests": {},
    }

    try:
        from textblob import TextBlob
        from nltk.tokenize import word_tokenize, sent_tokenize

        # Test sentiment analysis pipeline
        test_text = """
        This is an amazing product! I absolutely love it.
        However, the price could be better.
        """

        # TextBlob sentiment
        blob = TextBlob(test_text)
        sentiment_score = blob.sentiment.polarity

        # NLTK tokenization
        sentences = sent_tokenize(test_text)

        results["tests"]["sentiment_analysis"] = {
            "status": "passed" if -1 <= sentiment_score <= 1 else "failed",
            "score": sentiment_score,
            "sentences_analyzed": len(sentences),
        }

        # Test named entity recognition simulation
        entities_found = []
        for sentence in sentences:
            words = word_tokenize(sentence)
            capitalized_words = [w for w in words if w[0].isupper() and len(w) > 1]
            entities_found.extend(capitalized_words)

        results["tests"]["named_entity_detection"] = {
            "status": "passed",
            "potential_entities": entities_found,
        }

        # Test text statistics
        word_count = len(word_tokenize(test_text))
        char_count = len(test_text)

        results["tests"]["text_statistics"] = {
            "status": "passed",
            "word_count": word_count,
            "character_count": char_count,
            "average_word_length": sum(len(w) for w in word_tokenize(test_text))
            / len(word_tokenize(test_text)),
        }

        results["status"] = "working"

    except Exception as e:
        results["status"] = "failed"
        results["error"] = str(e)

    return results


def run_all_tests() -> None:
    """Run all NLP tests and display comprehensive results"""

    print("=" * 60)
    print("COMPREHENSIVE NLP CORE FUNCTIONALITY TEST")
    print("=" * 60)

    tests = [
        ("NLTK", test_nltk),
        ("TextBlob", test_textblob),
        ("spaCy", test_spacy),
        ("Advanced Pipeline", test_advanced_pipeline_functionality),
    ]

    results_summary = {}

    for name, test_func in tests:
        print(f"\n🔬 Testing {name}...")
        print("-" * 40)

        try:
            result = test_func()
            results_summary[name] = result

            print(f"Status: {result['status'].upper()}")

            # Display test results
            if "tests" in result:
                for test_name, test_result in result["tests"].items():
                    if isinstance(test_result, dict) and "status" in test_result:
                        status_icon = (
                            "✅" if test_result["status"] == "passed" else "❌"
                        )
                        print(f"{status_icon} {test_name}: {test_result['status']}")

                        if test_result["status"] == "passed":
                            # Show sample data
                            for key, value in test_result.items():
                                if key != "status" and isinstance(
                                    value, (int, float, str)
                                ):
                                    print(f"   {key}: {value}")
                    else:
                        status_icon = "✅"
                        print(f"{status_icon} {test_name}: passed")

            if "error" in result:
                print(f"❌ Error: {result['error']}")

        except Exception as e:
            print(f"❌ FAILED: {str(e)}")
            results_summary[name] = {"status": "failed", "error": str(e)}

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for name, result in results_summary.items():
        status = result.get("status", "unknown")
        if status == "working":
            print(f"✅ {name}: FULLY FUNCTIONAL")
        elif status == "model_missing":
            print(f"⚠️  {name}: LIBRARY OK, MODEL MISSING")
        elif status == "imported":
            print(f"🔶 {name}: IMPORTED, LIMITED FUNCTIONALITY")
        else:
            print(f"❌ {name}: NOT WORKING - {status}")

    # Check if core functionality is available
    working_libraries = [
        name
        for name, result in results_summary.items()
        if result.get("status") == "working"
    ]

    print(f"\n📊 Working Libraries: {len(working_libraries)}/3")

    if len(working_libraries) >= 2:
        print("🚀 Core NLP functionality is available for advanced pipeline!")
    else:
        print("⚠️  Limited NLP functionality - consider installing missing dependencies")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_all_tests()
