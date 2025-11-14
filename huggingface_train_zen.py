# Hugging Face Training Pipeline using ZenML
"""
This file implements a training pipeline for Qwen/Qwen3-VL-32B-Thinking- model using ZenML and Hugging Face Transformers. It loads a text dataset, trains on 10 batches, and saves the model checkpoint.
dataset, trains on 10 batches, and saves the model checkpoint.
"""

from typing import Tuple

import torch
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase

# ZenML imports
from zenml import pipeline, step
from zenml.integrations.huggingface.materializers import (
    HFDatasetMaterializer,
    HFPTModelMaterializer,
    HFTokenizerMaterializer,
)


@step(enable_cache=False, output_materializers=HFDatasetMaterializer)
def loading_dataset() -> Dataset:
    """Load and stream HuggingfaceFW/fineweb-edu for 10 batches.

    Returns:
        object: Streamed dataset with only first 10 batches
    """
    from datasets import load_dataset

    print("Loading HuggingfaceFW/fineweb-edu dataset...")

    # Load dataset with streaming to handle large datasets
    # Using a simpler dataset that works for testing - wiki_en from the same org

    try:
        # Try to load a working dataset configuration
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="CC-MAIN-2024-10",
            split="train",
            streaming=True,
        )
        dataset = iter(dataset)
        train_dataset = [
            next(dataset),
            next(dataset),
            next(dataset),
            next(dataset),
            next(dataset),
            next(dataset),
            next(dataset),
            next(dataset),
            next(dataset),
            next(dataset),
        ]  # First 10 examples

        train_dataset = Dataset.from_list(train_dataset)

    except Exception as e:
        print(f"Error loading FineWeb dataset: {e}")
        print("Falling back to a simpler test dataset...")

        # Fallback to a simple text dataset that we know works
        try:
            # Create a simple test dataset with 10 examples
            sample_data = [
                {"text": "This is a test example for training."},
                {"text": "Machine learning is fascinating and complex."},
                {"text": "Natural language processing has advanced greatly."},
                {"text": "Neural networks can learn complex patterns."},
                {"text": "Transformers revolutionized the field of NLP."},
                {"text": "Deep learning requires large amounts of data."},
                {"text": "Computer vision tasks benefit from CNNs."},
                {"text": "Attention mechanisms improve model performance."},
                {"text": "Large language models show emergent abilities."},
                {"text": "AI research continues to push boundaries."},
            ]

            # Create dataset object
            train_dataset = Dataset.from_list(sample_data)
            print(f"Created fallback test dataset with {len(train_dataset)} examples")

        except Exception as e2:
            print(f"Fallback dataset creation failed: {e2}")
            raise RuntimeError("Could not load any suitable dataset")

    print(
        f"Dataset loaded. Here is what it looks like:\n{train_dataset[0]['text'][:100]}..."
    )
    print("Tokenizing the dataset")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Coder-30B-A3B-Instruct")

    def tokenization(example):
        return tokenizer(example["text"])

    train_dataset = train_dataset.map(tokenization, batched=True)

    return train_dataset


@step(
    enable_cache=False,
    enable_step_logs=False,
    environment={
        "CUDA_MODULE_LOADING": "LAZY",
        "CUDA_CACHE_MAXSIZE": "2147483648",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "CUDA_FORCE_PTX_JIT": "1",
        "CUDA_VISIBLE_DEVICES": "0",
    },
)
def train_model(dataset: Dataset) -> str:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    model_name = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    print(f"Loading Model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        max_memory={0: "120GiB"},
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    output_dir = "./checkpoints/qwen3-vl-finetuned"

    print("Setting up training arguments...")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        logging_steps=1,  # Log every step since we only have 10
        save_steps=1000,
        warmup_steps=2,
        logging_dir=f"{output_dir}/logs",
        report_to="none",
        save_total_limit=2,
        max_steps=10,
        learning_rate=5e-5,
        weight_decay=0.01,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=False,  # Use bfloat16 instead for better stability
        bf16=True,  # Add this if your GPU supports it
    )

    # Add data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print("Emptying cache to make sure we're good for the train step")
    torch.cuda.empty_cache()

    print("Creating trainer...")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=data_collator,  # Critical addition
    )

    print("Starting training...")

    try:
        trainer.train()
        print("Training completed successfully!")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        print(f"Model saved to: {output_dir}")
    except Exception as e:
        print(f"Training error: {e}")
        import traceback

        traceback.print_exc()  # Get full error trace
        raise  # Re-raise to see the actual error

    return output_dir


@step(enable_cache=False)
def end_training(checkpoint_path: str) -> None:
    """Final step - report training completion."""
    if checkpoint_path and checkpoint_path == "./checkpoints/qwen3-vl-finetuned":
        print(f"Training pipeline completed successfully!")
        print(f"Final checkpoint saved at: {checkpoint_path}")

        # List contents of checkpoint directory
        import os

        if os.path.exists(checkpoint_path):
            print("\nCheckpoint contents:")
            for item in os.listdir(checkpoint_path):
                print(f"  - {item}")
        else:
            print("Warning: Checkpoint directory not found!")
    else:
        print("Training completed with fallback model.")

    print("HuggingFace Training Pipeline is all done.")


@pipeline
def huggingface_training_pipeline():
    """Complete training pipeline for Hugging Face model."""

    # Load dataset (10 batches)
    dataset = loading_dataset()

    # # Load model and tokenizer
    # model, tokenizer = load_model(after=[dataset])

    # Train model on 10 batches
    checkpoint_path = train_model(dataset=dataset, after=[dataset])

    # Final reporting step
    end_training(checkpoint_path=checkpoint_path)


if __name__ == "__main__":
    """Main execution point."""

    print("=" * 60)
    print("Hugging Face Training Pipeline with ZenML")
    print("Model: Qwen3-VL-32b-thinking")
    print("Dataset: FineWeb-edu (10 batches)")
    print("=" * 60)

    # Run the complete training pipeline
    huggingface_training_pipeline()

    print("\n" + "=" * 60)
    print("Pipeline execution completed!")
    print("Check './checkpoints/' directory for saved model.")
    print("=" * 60)
