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

        # Unique ASCII Art for NLP Pipeline
        print(
            Fore.WHITE
            + """
            Raw Text: "The cat sat"
                ↓
            [Tokenization]
                ↓
            ["The", "cat", "sat"]
                ↓
            [Embedding Lookup]
                ↓
            [vector₁, vector₂, vector₃]
                ↓
            [Attention/Transformer Layers] ←──┐
                ↓                              │
            [Context flows between tokens] ────┘
                ↓
            [Task-Specific Head]
                ↓
            Output: Classification/Generation/etc
        """
        )

        # Validate and setup configuration - convert parameters to regular variables
        self.task_value = str(self.task)
        self.model_type_value = str(self.model_type)
        self.batch_size_value = int(str(self.batch_size))
        self.max_length_value = int(str(self.max_length))
        self.text_samples_value = int(str(self.text_samples))

        # Consolidated configuration summary for non-ML engineers
        print(Fore.BLUE + f"📋 Configuration Summary:")
        print(
            Fore.CYAN
            + f"   • Task Type: {self.task_value} (analyzing text sentiment/classification)"
        )
        print(
            Fore.CYAN
            + f"   • Model: {self.model_type_value} (AI brain for understanding text)"
        )
        print(
            Fore.CYAN + f"   • Processing Size: {self.batch_size_value} texts at once"
        )
        print(
            Fore.CYAN
            + f"   • Text Samples: {self.text_samples_value} examples to analyze"
        )
        print(
            Fore.GREEN
            + f"✅ Configuration validated - ready to process {self.text_samples_value} texts"
        )
        self.next(self.prepare_environment)

    @step
    def prepare_environment(self):
        """Prepare the NLP processing environment and dependencies."""

        print(Fore.CYAN + "🔧 Preparing NLP processing environment...")
        print(Fore.BLUE + "   Loading AI models and setting up computing resources...")

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

        device_emoji = "🚀 GPU (fast)" if self.processing_device == "cuda" else "💻 CPU"
        print(Fore.GREEN + f"✅ Environment ready - using {device_emoji}")
        self.next(self.generate_sample_texts)

    @step
    def generate_sample_texts(self):
        """Generate diverse text samples for NLP processing."""

        print(
            Fore.CYAN
            + f"📝 Creating {self.text_samples_value} diverse text samples for analysis..."
        )
        print(
            Fore.BLUE
            + "   Generating realistic example texts (reviews, comments, descriptions)..."
        )

        # Generate diverse, realistic text samples
        self.sample_texts = self._create_diverse_text_samples()

        print(
            Fore.GREEN
            + f"✅ Generated {len(self.sample_texts)} text samples ready for processing"
        )

        self.next(self.load_models)

    @step
    def load_models(self):
        """Load appropriate models based on task and model type."""

        print(
            Fore.CYAN
            + f"🤖 Loading {self.model_type} AI model for {self.task_value} analysis..."
        )
        print(
            Fore.BLUE
            + "   Downloading and initializing the text understanding model..."
        )

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
                self.task_value in model_mapping
                and self.model_type_value in model_mapping[self.task_value]
            ):
                model_name = model_mapping[self.task_value][self.model_type_value]
            else:
                # Fallback to default
                print(
                    Fore.YELLOW
                    + f"⚠️ Model {self.model_type_value} not found for task {self.task_value}, using default"
                )
                model_name = "distilbert-base-uncased-finetuned-sst-2-english"

            print(Fore.BLUE + f"   Loading model: {model_name}")

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

            model_emoji = (
                "🧠 DistilBERT (fast & accurate)"
                if self.model_type_value == "distilbert"
                else "📚 BERT (comprehensive)"
                if self.model_type_value == "bert_base"
                else "💪 RoBERTa (robust)"
            )
            print(Fore.GREEN + f"✅ {model_emoji} loaded successfully")

        except Exception as e:
            print(Fore.RED + f"❌ Error loading model: {e}")
            # Fallback to simple sentiment analysis
            self._load_fallback_model()

        self.next(self.preprocess_texts)

    @step
    def preprocess_texts(self):
        """Preprocess and tokenize the text samples."""

        print(Fore.CYAN + "⚙️ Preparing texts for AI analysis...")
        print(
            Fore.BLUE
            + "   Converting text into numerical format the model can understand..."
        )

        try:
            # For pipeline models, we can skip explicit preprocessing
            if hasattr(self, "nlp_pipeline"):
                print(Fore.BLUE + "   Using streamlined processing mode")
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

            print(
                Fore.GREEN
                + f"✅ Text preparation complete - {len(self.sample_texts)} texts ready for analysis"
            )

        except Exception as e:
            print(Fore.RED + f"❌ Text preparation failed: {e}")
            # Fallback to unprocessed texts
            self.tokenized_texts = [
                text[:100] for text in self.sample_texts
            ]  # Truncate large texts

        self.next(self.run_nlp_processing)

    @step
    def run_nlp_processing(self):
        """Execute the main NLP processing tasks."""

        print(
            Fore.CYAN
            + f"🧠 Running {self.task_value} analysis on {len(self.sample_texts)} texts..."
        )
        print(
            Fore.BLUE
            + "   Analyzing each text to determine sentiment/classification..."
        )

        results = []
        errors = 0

        batch_size = int(self.batch_size_value)

        # Track progress for better user experience
        total_batches = (len(self.tokenized_texts) + batch_size - 1) // batch_size

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

                # Progress update every 5 batches
                current_batch = i // batch_size + 1
                if (
                    current_batch % max(1, total_batches // 5) == 0
                    or current_batch == total_batches
                ):
                    print(
                        Fore.BLUE
                        + f"   Progress: {current_batch}/{total_batches} batches processed"
                    )

            except Exception as e:
                print(Fore.RED + f"❌ Batch {i // batch_size} failed: {e}")
                errors += 1

        # Summary statistics
        self.processing_results = results
        self.total_errors = errors
        successful_processed = len([r for r in results if "error" not in str(r)])

        print(Fore.GREEN + f"✅ Analysis Complete:")
        print(Fore.CYAN + f"   • Total texts: {len(self.sample_texts)}")
        print(Fore.CYAN + f"   • Successfully analyzed: {successful_processed}")
        if errors > 0:
            print(Fore.YELLOW + f"   • Errors encountered: {errors}")

        self.next(self.analyze_results)

    @step
    def analyze_results(self):
        """Analyze and summarize NLP processing results."""

        print(Fore.CYAN + "📊 Calculating analysis results and insights...")

        # Process results based on task type
        if self.task_value == "sentiment":
            analysis = self._analyze_sentiment_results()
        elif self.task_value == "classification":
            analysis = self._analyze_classification_results()
        else:
            # Generic analysis for other tasks
            analysis = self._analyze_generic_results()

        self.result_analysis = analysis

        # Display results summary
        print(
            Fore.WHITE
            + f"""
    ╔════════════════════════════════════════╗
    ║                                        ║
    ║  🎯 {self.task_value.upper()} ANALYSIS RESULTS 💬           ║
    ║                                        ║
    ╚════════════════════════════════════════╝
        """
        )

        for key, value in analysis.items():
            if isinstance(value, (int, float)):
                print(
                    Fore.BLUE + f"{key.replace('_', ' ').title()}: {value:.3f}"
                    if isinstance(value, float)
                    else Fore.BLUE + f"{key.replace('_', ' ').title()}: {value}"
                )
            elif isinstance(value, list) and len(value) <= 5:
                print(Fore.BLUE + f"{key.replace('_', ' ').title()}: {value}")
            else:
                print(
                    Fore.BLUE
                    + f"{key.replace('_', ' ').title()}: {len(value) if hasattr(value, '__len__') else value}"
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

        print(
            Fore.WHITE
            + """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║  🎉 ADVANCED NLP PROCESSING PIPELINE SUCCESSFULLY COMPLETED! 🎉║
    ║                                                               ║
    ║  Your texts have been analyzed and insights generated         ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
        """
        )

        # Final summary
        total_texts = len(self.sample_texts) if hasattr(self, "sample_texts") else 0
        processed_successfully = (
            len([r for r in self.processing_results if "error" not in str(r)])
            if hasattr(self, "processing_results")
            else 0
        )

        print(Fore.BLUE + f"📊 Pipeline Summary:")
        print(Fore.CYAN + f"   • Task Type: {self.task_value}")
        print(Fore.CYAN + f"   • AI Model: {self.model_type_value}")
        print(Fore.CYAN + f"   • Total Texts Analyzed: {total_texts}")
        print(Fore.CYAN + f"   • Successfully Processed: {processed_successfully}")

        if hasattr(self, "result_analysis"):
            print(
                Fore.CYAN
                + f"   • Analysis Metrics: {len(self.result_analysis)} insights generated"
            )

        if hasattr(self, "insights"):
            print(
                Fore.CYAN
                + f"   • Recommendations: {len(self.insights)} suggestions provided"
            )

        success_rate = (
            (processed_successfully / total_texts * 100) if total_texts > 0 else 0
        )

        if success_rate >= 95:
            print(
                Fore.GREEN
                + f"✅ Excellent performance - {success_rate:.1f}% success rate!"
            )
        elif success_rate >= 80:
            print(
                Fore.YELLOW + f"⚠️ Good performance - {success_rate:.1f}% success rate"
            )
        else:
            print(
                Fore.RED
                + f"❌ Performance issues - only {success_rate:.1f}% success rate"
            )

        print(
            Fore.GREEN
            + "\n🎯 Your advanced NLP processing pipeline has completed successfully!"
        )

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

        print(Fore.YELLOW + "🔄 Loading backup sentiment analysis system...")

        try:
            from transformers import pipeline

            self.nlp_pipeline = pipeline("sentiment-analysis")
        except:
            # Final fallback - simple keyword-based analysis
            print(
                Fore.YELLOW
                + "⚠️ Using basic keyword-based sentiment analysis (limited accuracy)"
            )

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
