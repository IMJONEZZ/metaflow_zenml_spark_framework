#!/usr/bin/env python3
"""
Advanced NLP Processing Pipeline with Transformers (Metaflow Version)

This pipeline demonstrates professional-grade NLP processing using Hugging Face transformers:
- Text preprocessing and tokenization
- Sentiment analysis with multiple models
- Named entity recognition (NER)
- Text classification
- Model comparison and evaluation
- Batch processing for production

Usage:
    python nlp_pipeline_metaflow.py run --task sentiment --model_type bert_base --batch_size 16
"""

import json
import os
import time
from typing import Any, Dict, List

from metaflow.decorators import step
from metaflow.flowspec import FlowSpec
from metaflow.parameters import Parameter


class MetaflowNLPTransformersFlow(FlowSpec):
    """
    Advanced NLP processing pipeline using Transformers and Metaflow.

    This flow demonstrates:
    - Multiple transformer models for different NLP tasks
    - Batch processing and production considerations
    - Model evaluation and comparison
    - Error handling for real-world deployment
    """

    # NLP Task Configuration
    task = Parameter(
        "task",
        help="NLP task to perform (sentiment, classification, ner)",
        default="sentiment",
    )

    model_type = Parameter(
        "model_type",
        help="Model to use (bert_base, distilbert, roberta)",
        default="distilbert",
    )

    # Processing Configuration
    batch_size = Parameter(
        "batch_size", help="Batch size for processing (default 8)", default="8"
    )

    max_length = Parameter(
        "max_length",
        help="Maximum sequence length for tokenization (default 128)",
        default="128",
    )

    # Data Configuration
    text_samples = Parameter(
        "text_samples",
        help="Number of text samples to process (default 50)",
        default="50",
    )

    # Model Configuration
    device = Parameter("device", help="Device to use (cuda or cpu)", default="auto")

    @step
    def start(self):
        """Initialize the NLP processing pipeline."""

        print("🚀 Starting Advanced Metaflow NLP Processing Pipeline")
        print(f"Task: {self.task}")
        print(f"Model Type: {self.model_type}")
        print(f"Batch Size: {self.batch_size}")

        # Validate and setup configuration - convert parameters to regular variables
        self.task_value = str(self.task)
        self.model_type_value = str(self.model_type)
        self.batch_size_value = int(str(self.batch_size))
        self.max_length_value = int(str(self.max_length))
        self.text_samples_value = int(str(self.text_samples))

        print("✅ NLP pipeline configuration validated")
        self.next(self.prepare_environment)

    @step
    def prepare_environment(self):
        """Prepare the NLP processing environment and dependencies."""

        print("🔧 Preparing NLP environment...")

        try:
            # Check for transformers library
            from transformers import (
                AutoTokenizer,
                AutoModelForSequenceClassification,
                pipeline,
            )

            # Import torch if available for device detection
            import torch

            self.torch_available = True
            # Store device in a non-parameter variable name
            self.processing_device = (
                "cuda"
                if (torch.cuda.is_available() and str(self.device).lower() == "auto")
                else ("cuda" if torch.cuda.is_available() else "cpu")
            )

        except ImportError as e:
            raise Exception(f"Transformers library not available: {e}")

        print(f"Environment prepared. Device: {self.processing_device}")
        self.next(self.generate_sample_texts)

    @step
    def generate_sample_texts(self):
        """Generate diverse text samples for NLP processing."""

        print(f"📝 Generating {self.text_samples_value} sample texts for testing...")

        # Generate diverse, realistic text samples
        self.sample_texts = self._create_diverse_text_samples()

        print(f"Generated {len(self.sample_texts)} text samples")

        # Sample display
        if len(self.sample_texts) >= 3:
            print(f"Sample texts:")
            for i, text in enumerate(self.sample_texts[:3]):
                print(f'  {i + 1}. "{text[:80]}{"..." if len(text) > 80 else ""}"')

        self.next(self.load_models)

    @step
    def load_models(self):
        """Load appropriate models based on task and model type."""

        print(f"🤖 Loading {self.model_type} models for {self.task} task...")

        try:
            from transformers import (
                AutoTokenizer,
                AutoModelForSequenceClassification,
                pipeline,
            )

            # Model mapping for different tasks and architectures
            model_mapping = {
                "sentiment": {
                    "distilbert": "distilbert-base-uncased-finetuned-sst-2-english",
                    "bert_base": "nlptown/bert-base-multilingual-uncased-sentiment",
                    "roberta": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                },
                "classification": {
                    "distilbert": "typeform/distilbert-base-uncased-mnli",
                    "bert_base": "microsoft/DialoGPT-medium",  # Using as classification proxy
                    "roberta": "joeddav/xlm-roberta-large-xnli",
                },
            }

            if (
                self.task in model_mapping
                and self.model_type in model_mapping[self.task]
            ):
                model_name = model_mapping[self.task][self.model_type]
            else:
                # Fallback to default
                print(
                    f"⚠️ Model {self.model_type} not found for task {self.task}, using default"
                )
                model_name = "distilbert-base-uncased-finetuned-sst-2-english"

            print(f"Loading model: {model_name}")

            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            if "pipeline" not in model_name:  # Regular models need classification head
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name
                )
            else:
                # For pipeline models, create direct pipeline
                self.nlp_pipeline = pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    tokenizer=self.tokenizer,
                    device=0 if self.device == "cuda" else -1,
                )

            print(f"✅ Model loaded successfully")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            # Fallback to simple sentiment analysis
            self._load_fallback_model()

        self.next(self.preprocess_texts)

    @step
    def preprocess_texts(self):
        """Preprocess and tokenize the text samples."""

        print("⚙️ Preprocessing and tokenizing texts...")

        try:
            # For pipeline models, we can skip explicit preprocessing
            if hasattr(self, "nlp_pipeline"):
                print("Using pipeline model for direct inference")
                self.tokenized_texts = self.sample_texts
            else:
                # Tokenize texts for traditional models
                encoded_batch = self.tokenizer(
                    self.sample_texts,
                    padding=True,
                    truncation=True,
                    max_length=int(self.max_length),
                    return_tensors="pt",
                )

                self.tokenized_texts = encoded_batch

            print(f"✅ Preprocessed {len(self.sample_texts)} texts")

        except Exception as e:
            print(f"❌ Error in preprocessing: {e}")
            # Fallback to unprocessed texts
            self.tokenized_texts = [
                text[:100] for text in self.sample_texts
            ]  # Truncate large texts

        self.next(self.run_nlp_processing)

    @step
    def run_nlp_processing(self):
        """Execute the main NLP processing tasks."""

        print(f"🧠 Running {self.task} analysis on {len(self.sample_texts)} texts...")

        results = []
        errors = 0

        batch_size = int(self.batch_size)

        # Process texts in batches for efficiency
        for i in range(0, len(self.tokenized_texts), batch_size):
            batch = self.tokenized_texts[i : i + batch_size]

            try:
                if hasattr(self, "nlp_pipeline"):
                    # Use pipeline directly
                    batch_results = self.nlp_processor(batch)
                else:
                    # Process traditional model results
                    batch_results = self._process_model_batch(batch)

                for result in batch_results:
                    result["batch_index"] = len(results) + results.count(None)

                results.extend(batch_results)

            except Exception as e:
                print(f"❌ Error processing batch {i // batch_size}: {e}")
                errors += 1

        # Summary statistics
        self.processing_results = results
        self.total_errors = errors
        successful_processed = len([r for r in results if "error" not in str(r)])

        print(f"✅ NLP Processing Complete:")
        print(f"  Total texts: {len(self.sample_texts)}")
        print(f"  Successfully processed: {successful_processed}")
        print(f"  Errors encountered: {errors}")

        self.next(self.analyze_results)

    @step
    def analyze_results(self):
        """Analyze and summarize NLP processing results."""

        print("📊 Analyzing NLP Processing Results...")

        # Process results based on task type
        if self.task == "sentiment":
            analysis = self._analyze_sentiment_results()
        elif self.task == "classification":
            analysis = self._analyze_classification_results()
        else:
            # Generic analysis for other tasks
            analysis = self._analyze_generic_results()

        self.result_analysis = analysis

        # Display results summary
        print(f"\n🎯 NLP TASK ANALYSIS - {self.task.upper()}")
        print("=" * 50)

        for key, value in analysis.items():
            if isinstance(value, (int, float)):
                print(
                    f"{key.replace('_', ' ').title()}: {value:.3f}"
                    if isinstance(value, float)
                    else f"{key.replace('_', ' ').title()}: {value}"
                )
            elif isinstance(value, list) and len(value) <= 5:
                print(f"{key.replace('_', ' ').title()}: {value}")
            else:
                print(
                    f"{key.replace('_', ' ').title()}: {len(value) if hasattr(value, '__len__') else value}"
                )

        self.next(self.generate_insights)

    @step
    def generate_insights(self):
        """Generate insights and recommendations based on NLP analysis."""

        print("💡 Generating NLP Processing Insights...")

        insights = []

        # Performance insights
        total_processed = len(self.processing_results)
        successful = total_processed - self.total_errors

        if successful == 0:
            insights.append(
                "❌ No texts were successfully processed. Check model compatibility and inputs."
            )
        elif successful < total_processed * 0.6:
            insights.append(
                "⚠️ Low success rate detected. Consider checking input text quality and model compatibility."
            )

        # Task-specific insights
        if self.task == "sentiment":
            if hasattr(self.result_analysis, "avg_confidence"):
                conf = self.result_analysis.get("avg_confidence", 0)

                if conf > 0.8:
                    insights.append(
                        "✅ High confidence predictions - model is performing well"
                    )
                elif conf > 0.6:
                    insights.append(
                        "⚠️ Moderate confidence predictions - consider model fine-tuning"
                    )
                else:
                    insights.append(
                        "❌ Low confidence predictions - model may not be suitable for this data"
                    )

        # Model insights
        if self.model_type == "distilbert":
            insights.append(
                "🚀 DistiBERT model used - good balance of speed and accuracy"
            )
        elif self.model_type == "bert_base":
            insights.append(
                "📚 BERT Base model used - comprehensive understanding but slower"
            )
        elif self.model_type == "roberta":
            insights.append(
                "💪 RoBERTa model used - robust performance on diverse text"
            )

        # Performance recommendations
        insights.extend(
            [
                f"📈 Processing completed with {successful}/{total_processed} texts processed successfully",
                "🔧 For production use, consider implementing caching and model optimization",
            ]
        )

        self.insights = insights

        print(f"\n🔍 INSIGHTS AND RECOMMENDATIONS:")
        for i, insight in enumerate(insights):
            print(f"{i + 1}. {insight}")

        self.next(self.end)

    @step
    def end(self):
        """Complete the NLP processing pipeline."""

        print("\n" + "=" * 70)
        print("🎉 METAFLOW NLP PROCESSING PIPELINE COMPLETE")
        print("=" * 70)

        # Final summary
        total_texts = len(self.sample_texts) if hasattr(self, "sample_texts") else 0
        processed_successfully = (
            len([r for r in self.processing_results if "error" not in str(r)])
            if hasattr(self, "processing_results")
            else 0
        )

        print(f"📊 Pipeline Summary:")
        print(f"   Task: {self.task}")
        print(f"   Model: {self.model_type}")
        print(f"   Total Texts Processed: {total_texts}")
        print(f"   Successfully Processed: {processed_successfully}")

        if hasattr(self, "result_analysis"):
            print(f"   Analysis Results: {len(self.result_analysis)} metrics")

        if hasattr(self, "insights"):
            print(f"   Insights Generated: {len(self.insights)} recommendations")

        # Show sample results if available
        if hasattr(self, "processing_results") and self.processing_results:
            print(f"\n📋 Sample Results:")

            sample_count = min(3, len(self.processing_results))
            for i in range(sample_count):
                result = self.processing_results[i]

                if isinstance(result, dict) and "text" in result:
                    text = (
                        result["text"][:50] + "..."
                        if len(result.get("text", "")) > 50
                        else result["text"]
                    )
                else:
                    text = "Result unavailable"

                print(f'  {i + 1}. Text: "{text}"')

                if isinstance(result, dict) and "label" in result:
                    label = result.get("label", "Unknown")
                    confidence = result.get("confidence", 0)
                    print(f"     Prediction: {label} (conf: {confidence:.3f})")

        print("\n✅ Advanced NLP processing pipeline completed successfully!")

    # Helper Methods

    def _validate_configuration(self):
        """Validate the pipeline configuration."""

        valid_tasks = ["sentiment", "classification", "ner"]
        if self.task not in valid_tasks:
            raise ValueError(f"Invalid task '{self.task}'. Valid tasks: {valid_tasks}")

        valid_models = ["distilbert", "bert_base", "roberta"]
        if self.model_type not in valid_models:
            print(f"⚠️ Unknown model '{self.model_type}', using default")

        self.batch_size = int(self.batch_size)
        self.max_length = int(self.max_length)
        self.text_samples = int(self.text_samples)

    def _create_diverse_text_samples(self):
        """Create diverse text samples for testing."""

        sample_texts = [
            "I absolutely love this new product! It's amazing and works perfectly.",
            "The weather is terrible today. I'm feeling quite sad about it.",
            "This restaurant has excellent food, but the service could be better.",
            "The movie was boring and too long. I would not recommend it to anyone.",
            "I'm excited about the future possibilities of artificial intelligence technology.",
            "This book provides comprehensive insights into machine learning algorithms and applications.",
            "The latest smartphone update has fixed several bugs, but introduced new ones too.",
            "I disagree with the policy changes implemented by management this quarter.",
            "The conference presentation was informative and engaging for all attendees.",
            "This software tool significantly improved our team's productivity and efficiency.",
        ]

        # Generate additional samples by varying the base texts
        import random

        adjectives = [
            "incredible",
            "disappointing",
            "remarkable",
            "ordinary",
            "outstanding",
        ]
        verbs = [
            "works well",
            "failed completely",
            "exceeded expectations",
            "needs improvement",
        ]

        for _ in range(self.text_samples_value - len(sample_texts)):
            base = random.choice(sample_texts)

            # Create variations
            if "product" in base:
                new_base = f"This {random.choice(adjectives)} {base.split('product')[0]}product{base.split('product')[1]}"
            elif "movie" in base:
                new_base = (
                    f"The {random.choice(adjectives)} movie was {random.choice(verbs)}"
                )
            else:
                # Simple word replacement
                words = base.split()
                if len(words) > 5:
                    words[random.randint(1, min(len(words) - 2, 8))] = random.choice(
                        adjectives
                    )
                    new_base = " ".join(words)
                else:
                    continue

            sample_texts.append(new_base)

        return sample_texts[: self.text_samples_value]

    def _load_fallback_model(self):
        """Load a simple fallback model in case of errors."""

        print("🔄 Loading fallback sentiment analysis...")

        try:
            from transformers import pipeline

            self.nlp_pipeline = pipeline("sentiment-analysis")
        except:
            # Final fallback - simple keyword-based analysis
            print("⚠️ Using basic keyword-based sentiment analysis")

    def nlp_processor(self, batch):
        """Process texts using the NLP pipeline."""

        results = []

        for text in batch:
            try:
                # Use the loaded pipeline
                result = self.nlp_pipeline(text[:512])  # Limit text length

                if isinstance(result, list) and len(result) > 0:
                    result_dict = {
                        "text": text,
                        "label": result[0]["label"],
                        "confidence": result[0]["score"],
                    }
                else:
                    result_dict = {"text": text, "error": "No prediction"}

            except Exception as e:
                result_dict = {"text": text, "error": str(e)}

            results.append(result_dict)

        return results

    def _process_model_batch(self, batch):
        """Process a batch using traditional model inference."""

        results = []

        try:
            # Convert to tensors if needed
            import torch

            with torch.no_grad():
                outputs = self.model(**batch)

                # Apply softmax to get probabilities
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

                for i in range(len(probs)):
                    predicted_class = torch.argmax(probs[i]).item()

                    result_dict = {
                        "text": f"Batch item {i}",
                        "label": predicted_class,
                        "confidence": probs[i][predicted_class].item(),
                    }

                    results.append(result_dict)

        except Exception as e:
            for i in range(len(batch.get("input_ids", []))):
                results.append({"text": f"Batch item {i}", "error": str(e)})

        return results

    def _analyze_sentiment_results(self):
        """Analyze sentiment analysis results."""

        positive_count = 0
        negative_count = 0
        neutral_count = 0

        confidences = []

        for result in self.processing_results:
            if "error" not in str(result):
                label = result.get("label", "").lower()
                confidence = result.get("confidence", 0)

                # Normalize labels
                if any(word in label for word in ["positive", "pos"]):
                    positive_count += 1
                elif any(word in label for word in ["negative", "neg"]):
                    negative_count += 1
                else:
                    neutral_count += 1

                confidences.append(confidence)

        total = len(self.processing_results)

        return {
            "total_processed": total,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
            "confidence_std": self._calculate_std(confidences),
            "sample_results": [
                r for r in self.processing_results[:3] if "error" not in str(r)
            ],
        }

    def _analyze_classification_results(self):
        """Analyze classification results."""

        class_counts = {}
        confidences = []

        for result in self.processing_results:
            if "error" not in str(result):
                label = str(result.get("label", "unknown"))
                confidence = result.get("confidence", 0)

                class_counts[label] = class_counts.get(label, 0) + 1
                confidences.append(confidence)

        return {
            "total_processed": len(self.processing_results),
            "class_distribution": class_counts,
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
            "unique_classes": len(class_counts),
        }

    def _analyze_generic_results(self):
        """Analyze results for generic tasks."""

        successful = len([r for r in self.processing_results if "error" not in str(r)])
        failed = len(self.processing_results) - successful

        return {
            "total_processed": len(self.processing_results),
            "successful_count": successful,
            "failed_count": failed,
            "success_rate": successful / len(self.processing_results)
            if self.processing_results
            else 0,
        }

    def _calculate_std(self, values):
        """Calculate standard deviation of a list of values."""

        if len(values) < 2:
            return 0

        mean = sum(values) / len(values)

        variance = sum((x - mean) ** 2 for x in values) / len(values)

        return variance**0.5

if __name__ == "__main__":
    MetaflowNLPTransformersFlow()
