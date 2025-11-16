#!/usr/bin/env python3
"""
Simple NLP Processing Pipeline (ZenML Version)

This demonstrates NLP processing using ZenML:
- Text generation
- Simple text analysis
- Summary reporting

Usage:
    python nlp_pipeline_zenml_simple.py
"""

from datetime import datetime
from typing import Dict, List

# ZenML imports
from zenml import pipeline as zenml_pipeline
from zenml import step


@step(enable_cache=True)
def generate_text_samples(num_samples: int = 10) -> List[str]:
    """Generate sample texts for processing."""

    # Simple text samples
    texts = [
        "I love this product! It's amazing and works perfectly.",
        "The weather is terrible today. I'm feeling quite sad about it.",
        "This restaurant has excellent food, but the service could be better.",
        "The movie was boring and too long. I would not recommend it.",
        "I'm excited about the future of artificial intelligence technology.",
        "This book provides comprehensive insights into machine learning.",
        "The latest smartphone update has fixed several bugs and issues.",
        "I disagree with the policy changes implemented this quarter.",
        "The conference presentation was informative and engaging.",
        "This software tool significantly improved our team's productivity.",
    ]

    # Generate variations if needed
    import random

    for _ in range(num_samples - len(texts)):
        base = random.choice(texts)

        # Simple word replacement
        words = base.split()
        if len(words) > 5:
            # Replace some adjectives
            replacements = {
                "amazing": "incredible",
                "terrible": "awful",
                "excellent": "outstanding",
                "boring": "dull",
                "excited": "enthusiastic",
            }

            for i, word in enumerate(words):
                if word.lower() in replacements:
                    words[i] = replacements[word.lower()]

            texts.append(" ".join(words))

    print(f"Generated {len(texts)} text samples")
    return texts[:num_samples]


@step(enable_cache=False)
def analyze_texts(texts: List[str]) -> Dict:
    """Simple analysis of text samples."""

    results = {
        "sample_count": len(texts),
        "avg_length": sum(len(text) for text in texts) // len(texts) if texts else 0,
        "total_chars": sum(len(text) for text in texts),
    }

    # Simple sentiment analysis (keyword-based)
    positive_words = ["love", "amazing", "excellent", "excited", "productive"]
    negative_words = ["terrible", "sad", "boring", "disagree", "awful"]

    positive_count = 0
    negative_count = 0

    for text in texts:
        text_lower = text.lower()
        if any(word in text_lower for word in positive_words):
            positive_count += 1
        elif any(word in text_lower for word in negative_words):
            negative_count += 1

    results.update(
        {
            "positive_samples": positive_count,
            "negative_samples": negative_count,
            "neutral_samples": len(texts) - positive_count - negative_count,
        }
    )

    print(f"Analyzed {len(texts)} texts")
    return results


@step(enable_cache=False)
def extract_and_display_results(
    analysis_data: Dict,
) -> None:
    """Extract values from analysis results and display them."""

    # Handle different possible types of analysis_data
    try:
        # Try to get the actual data using getattr for safer access
        if "StepArtifact" in str(type(analysis_data)):
            # This is a StepArtifact - use getattr to avoid static analysis issues
            results = getattr(analysis_data, "value", None)
            if results is None:
                print("Error: Could not extract value from StepArtifact")
                return
        else:
            # Assume it's already a dict or convertable to one
            results = analysis_data

        # Ensure we have a dictionary-like object
        if not hasattr(results, "get"):
            print(
                f"Error: Results object doesn't support dict-like access. Type: {type(results)}"
            )
            return

        # Safely get values with defaults
        sample_count = int(results.get("sample_count", 0))
        avg_length = int(results.get("avg_length", 0))
        total_chars = int(results.get("total_chars", 0))
        positive_samples = int(results.get("positive_samples", 0))
        negative_samples = int(results.get("negative_samples", 0))
        neutral_samples = int(results.get("neutral_samples", 0))

    except (AttributeError, TypeError, ValueError) as e:
        print(f"Error processing analysis results: {e}")
        return

    # Display the results
    print("\n" + "=" * 50)
    print("🎯 NLP ANALYSIS RESULTS")
    print("=" * 50)

    print(f"📊 Text Analysis Summary:")
    print(f"   Total Samples: {sample_count}")
    print(f"   Average Length: {avg_length} characters")
    print(f"   Total Characters: {total_chars}")

    print(f"\n😊 Sentiment Distribution:")
    print(f"   Positive: {positive_samples}")
    print(f"   Negative: {negative_samples}")
    print(f"   Neutral: {neutral_samples}")

    if sample_count > 0:
        pos_pct = (positive_samples / sample_count) * 100
        neg_pct = (negative_samples / sample_count) * 100
        print(f"   Positive %: {pos_pct:.1f}%")
        print(f"   Negative %: {neg_pct:.1f}%")

    print("\n✅ NLP analysis completed successfully!")


@zenml_pipeline
def simple_nlp_pipeline(num_samples: int = 10) -> None:
    """Execute the complete NLP pipeline."""

    print("🚀 Starting Simple NLP Processing Pipeline")
    print(f"Samples to process: {num_samples}")

    # Execute pipeline steps
    texts = generate_text_samples(num_samples)
    analysis_results = analyze_texts(texts)

    # Display results - pass the artifact directly to extraction step
    extract_and_display_results(analysis_data=analysis_results)


if __name__ == "__main__":
    # Running the pipeline locally via ZenML's default orchestrator.
    simple_nlp_pipeline(num_samples=10)
