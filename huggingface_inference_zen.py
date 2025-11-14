# Hugging Face Inference Pipeline using ZenML
"""
This file implements an inference pipeline that loads a saved checkpoint from local
file system and performs inference on text prompts. Designed for deployment using ZenML.
"""

from typing import Tuple

import torch

# ZenML imports
from zenml import pipeline, step


@step(enable_cache=True)
def load_checkpoint() -> Tuple[object, object]:
    """Load saved checkpoint from local file system for inference.

    Returns:
        Tuple[object, object]: (model, tokenizer) loaded from checkpoint
    """
    import os

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    checkpoint_path = "./checkpoints/qwen3-vl-finetuned"
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)

    print(f"Loading checkpoint from: {checkpoint_path}")

    # Check if checkpoint directory exists
    if not os.path.exists(checkpoint_path):
        error_msg = f"Checkpoint path does not exist: {checkpoint_path}"
        print(f"Error: {error_msg}")

        # List available checkpoints to help debug
        checkpoints_dir = "./checkpoints"
        if os.path.exists(checkpoints_dir):
            print(f"Available items in {checkpoints_dir}:")
            for item in os.listdir(checkpoints_dir):
                print(f"  - {item}")
        else:
            print("Checkpoints directory does not exist.")

        raise FileNotFoundError(error_msg)

    try:
        # Load model from checkpoint
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path, quantization_config=quantization_config
        )

        # Load tokenizer from checkpoint
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

        # Set pad token if not exists
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        print("Checkpoint loaded successfully!")

        return model, tokenizer

    except Exception as e:
        error_msg = f"Error loading checkpoint: {e}"
        print(f"Error: {error_msg}")
        raise RuntimeError(error_msg)


@step(enable_cache=False)
def inference_step(
    model: object, tokenizer: object, prompt: str = "What is artificial intelligence?"
) -> str:
    """Perform inference on given prompt using the loaded model.

    Args:
        model: The loaded model for inference
        tokenizer: The loaded tokenizer
        prompt (str): Input text prompt for generation

    Returns:
        str: Generated response from the model
    """

    if not prompt or not isinstance(prompt, str):
        print("Warning: Invalid prompt provided. Using default.")
        prompt = "What is artificial intelligence?"

    print(
        f"Processing inference for prompt: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'"
    )

    try:
        # Set model to evaluation mode
        model.eval()

        # Tokenize input prompt
        inputs = tokenizer(prompt, return_tensors="pt")

        # Move to same device as model
        device = next(model.parameters()).device
        inputs = inputs.to(device)

        print(f"Input tokenized. Shape: {inputs.shape}")

        # Generate response with controlled parameters
        with torch.no_grad():  # Disable gradients for inference
            outputs = model.generate(
                inputs,
                max_new_tokens=20000,  # Limit response length
                temperature=0.7,  # Control randomness (lower = more focused)
                do_sample=True,  # Enable sampling for variety
                top_p=0.9,  # Nucleus sampling parameter
                top_k=50,  # Limit vocabulary to top k tokens
                pad_token_id=tokenizer.pad_token_id,
                # Quality improvements
                repetition_penalty=1.1,  # Reduce repetitive text
                length_penalty=1.0,  # Neutral length penalty
                # Safety and coherence
                no_repeat_ngram_size=2,  # Avoid repeating 2-grams
            )

        print(f"Generation completed. Output shape: {outputs.shape}")

        # Decode response, skipping special tokens
        raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(
            f"Raw output: '{raw_output[:100]}{'...' if len(raw_output) > 100 else ''}'"
        )

        return raw_output

    except Exception as e:
        error_msg = f"Error during inference: {e}"
        print(f"Inference error: {error_msg}")

        # Return a fallback response
        return f"I apologize, but I encountered an error during inference: {str(e)}"


@step(enable_cache=False)
def format_response(raw_output: str, original_prompt: str) -> str:
    """Format the raw model output for clean response presentation.

    Args:
        raw_output (str): Raw output from the model
        original_prompt (str): The original prompt for context

    Returns:
        str: Formatted, clean response
    """

    try:
        if not raw_output or not isinstance(raw_output, str):
            return "I couldn't generate a response. Please try again."

        # Remove prompt from output to get only generated text
        if raw_output.startswith(original_prompt):
            # Clean response - remove prompt and any extra whitespace
            response = raw_output[len(original_prompt) :].strip()
        else:
            # If prompt not found in output, use raw response
            response = raw_output.strip()

        # Remove any remaining special tokens or artifacts
        import re

        # Clean up common artifacts
        response = re.sub(r"<\|.*?\|>", "", response)  # Remove control tokens
        response = re.sub(r"\s+", " ", response)  # Normalize whitespace

        # Ensure we have a meaningful response
        if len(response.strip()) < 2:
            return "I generated a very short response. The model may need more training or better prompting."

        # Final cleanup
        response = response.strip()

        if not response.endswith((".", "!", "?", ":")):
            # Add period if no proper ending
            response += "."

        print(
            f"Formatted response: '{response[:80]}{'...' if len(response) > 80 else ''}'"
        )

        return response

    except Exception as e:
        error_msg = f"Error formatting response: {e}"
        print(f"Formatting error: {error_msg}")

        # Return original output if formatting fails
        return (
            raw_output.strip()
            if isinstance(raw_output, str)
            else "Response formatting failed."
        )


@step(enable_cache=False)
def log_inference_result(response: str, prompt: str) -> None:
    """Log the inference result for monitoring and debugging.

    Args:
        response (str): The final generated response
        prompt (str): The original input prompt
    """

    print("\n" + "=" * 60)
    print("INFERENCE RESULT")
    print("=" * 60)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print(f"Response Length: {len(response)} characters")
    print("=" * 60)


@pipeline
def huggingface_inference_pipeline(prompt: str = "What is machine learning?"):
    """Complete inference pipeline with deployment-ready endpoints.

    Args:
        prompt (str): Input text prompt for generation

    Returns:
        str: Formatted response from the model
    """

    # Load checkpoint from local file system
    model, tokenizer = load_checkpoint()

    # Perform inference on the prompt
    raw_output = inference_step(model=model, tokenizer=tokenizer, prompt=prompt)

    # Format the response for clean presentation
    formatted_response = format_response(raw_output=raw_output, original_prompt=prompt)

    # Log the result (can be extended for monitoring)
    log_inference_result(response=formatted_response, prompt=prompt)

    return formatted_response


def main() -> None:
    """Main function for standalone execution."""

    print("=" * 60)
    print("Hugging Face Inference Pipeline with ZenML")
    print("Loading from: ./checkpoints/qwen3-vl-finetuned")
    print("=" * 60)

    # Example prompts for testing
    sample_prompts = [
        "What is artificial intelligence?",
        # "Explain the concept of deep learning:",
        # "How does a neural network work?",
    ]

    print("Running inference on sample prompts:")

    for i, prompt in enumerate(sample_prompts, 1):
        print(f"\n--- Inference {i}/{len(sample_prompts)} ---")

        try:
            # Run inference pipeline
            response = huggingface_inference_pipeline(prompt=prompt)

            print(f"Final Response: {response}")

        except Exception as e:
            print(f"Inference failed for prompt '{prompt}': {e}")

        # Separator between inferences
        if i < len(sample_prompts):
            print("-" * 40)

    print("\n" + "=" * 60)
    print("Inference pipeline execution completed!")
    print("=" * 60)


if __name__ == "__main__":
    """Main execution point."""

    # Run standalone inference examples
    main()
