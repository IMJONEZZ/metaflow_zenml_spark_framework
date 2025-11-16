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

from typing import Annotated, Any, Dict, List

# ZenML imports
from zenml import log_metadata, pipeline as zenml_pipeline, step
from zenml.types import HTMLString

# Local imports
from nlp_html_utils import generate_nlp_html_report


@step(enable_cache=True)
def generate_text_samples(num_samples: int = 10) -> Annotated[List[str], "text_samples"]:
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
def analyze_texts(texts: List[str]) -> Annotated[Dict[str, Any], "analysis_results"]:
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
    texts: List[str],
    analysis_data: Dict[str, Any],
) -> Annotated[HTMLString, "html_report"]:
    """Create an HTML visualization of the NLP analysis results and log metadata."""

    # Extract values from analysis results
    sample_count = int(analysis_data.get("sample_count", 0))
    avg_length = int(analysis_data.get("avg_length", 0))
    total_chars = int(analysis_data.get("total_chars", 0))
    positive_samples = int(analysis_data.get("positive_samples", 0))
    negative_samples = int(analysis_data.get("negative_samples", 0))
    neutral_samples = int(analysis_data.get("neutral_samples", 0))

    # Calculate percentages
    pos_pct = (positive_samples / sample_count * 100) if sample_count > 0 else 0.0
    neg_pct = (negative_samples / sample_count * 100) if sample_count > 0 else 0.0
    neutral_pct = (neutral_samples / sample_count * 100) if sample_count > 0 else 0.0

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

    # Log metadata with all the analysis values
    log_metadata(
        metadata={
            "text_analysis": {
                "sample_count": sample_count,
                "avg_length": avg_length,
                "total_chars": total_chars,
            },
            "sentiment_distribution": {
                "positive_samples": positive_samples,
                "negative_samples": negative_samples,
                "neutral_samples": neutral_samples,
                "positive_percentage": round(pos_pct, 2),
                "negative_percentage": round(neg_pct, 2),
                "neutral_percentage": round(neutral_pct, 2),
            }
        },
    )

    return HTMLString(generate_nlp_html_report(
        texts=texts,
        sample_count=sample_count,
        avg_length=avg_length,
        total_chars=total_chars,
        positive_samples=positive_samples,
        negative_samples=negative_samples,
        neutral_samples=neutral_samples,
        pos_pct=pos_pct,
        neg_pct=neg_pct,
        neutral_pct=neutral_pct,
    ))


@zenml_pipeline
def simple_nlp_pipeline(num_samples: int = 10) -> None:
    """Execute the complete NLP pipeline."""

    print("🚀 Starting Simple NLP Processing Pipeline")
    print(f"Samples to process: {num_samples}")

    # Execute pipeline steps
    texts = generate_text_samples(num_samples)
    analysis_results = analyze_texts(texts)

    # Display results - pass the artifact directly to extraction step
    html_report = extract_and_display_results(texts=texts, analysis_data=analysis_results)


if __name__ == "__main__":
    # Running the pipeline locally via ZenML's default orchestrator.
    simple_nlp_pipeline(num_samples=10)
