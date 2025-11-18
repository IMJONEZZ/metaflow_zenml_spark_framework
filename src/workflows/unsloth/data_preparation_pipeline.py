# data_preparation.py
import json
import random
from pathlib import Path

import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# Initialize tokenizer for length estimation
enc = tiktoken.encoding_for_model("gpt-4")


def count_tokens(text):
    """Estimate token count"""
    return len(enc.encode(text))


def download_arxiv_data(output_dir="./data", sample_size=50000):
    """Download and process arXiv dataset"""
    print("Downloading arXiv dataset...")

    # Load from HuggingFace (smaller, curated version)
    dataset = load_dataset("scientific_papers", "arxiv", split="train")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_data = []

    for item in tqdm(dataset.select(range(min(sample_size, len(dataset))))):
        abstract = item["abstract"]
        article = item["article"]

        # Filter by length (want substantial papers)
        token_count = count_tokens(article)
        if token_count < 1000 or token_count > 15000:
            continue

        # Create instruction-following format
        # Task 1: Summarization
        summary_example = {
            "instruction": "Summarize the following scientific paper, highlighting key contributions, methodology, and results.",
            "input": article,
            "output": abstract,  # Abstract serves as ground truth summary
        }
        processed_data.append(summary_example)

        # Task 2: Key findings extraction (synthesize from abstract)
        findings = extract_key_findings(abstract)
        if findings:
            findings_example = {
                "instruction": "Extract the key findings and contributions from this scientific paper.",
                "input": article,
                "output": findings,
            }
            processed_data.append(findings_example)

    # Save to JSONL
    output_file = output_dir / "arxiv_processed.jsonl"
    with open(output_file, "w") as f:
        for item in processed_data:
            f.write(json.dumps(item) + "\n")

    print(f"Processed {len(processed_data)} examples")
    print(f"Saved to {output_file}")
    return output_file


def extract_key_findings(abstract):
    """Extract structured findings from abstract (simplified)"""
    # In production, use more sophisticated extraction
    # For now, extract sentences with key phrases
    key_phrases = [
        "we show",
        "we demonstrate",
        "we propose",
        "our results",
        "we find",
        "we present",
    ]

    sentences = abstract.split(". ")
    findings = [
        s for s in sentences if any(phrase in s.lower() for phrase in key_phrases)
    ]

    if len(findings) >= 2:
        return " ".join(findings)
    return None


def create_qa_pairs(output_dir="./data", num_samples=10000):
    """Create question-answering pairs from papers"""
    # This would use a more sophisticated approach in production
    # For demonstration, we'll create template-based QA

    dataset = load_dataset("scientific_papers", "arxiv", split="train")
    output_dir = Path(output_dir)

    qa_data = []

    for item in tqdm(dataset.select(range(num_samples))):
        article = item["article"]
        abstract = item["abstract"]

        # Generate different types of questions
        questions = [
            ("What is the main contribution of this paper?", abstract),
            ("What methodology does this paper use?", extract_methodology(article)),
            ("What are the key results presented?", extract_results(abstract)),
        ]

        for question, answer in questions:
            if answer and len(answer) > 50:  # Filter low-quality answers
                qa_data.append(
                    {"instruction": question, "input": article, "output": answer}
                )

    output_file = output_dir / "arxiv_qa.jsonl"
    with open(output_file, "w") as f:
        for item in qa_data:
            f.write(json.dumps(item) + "\n")

    print(f"Created {len(qa_data)} QA pairs")
    return output_file


def extract_methodology(text):
    """Extract methodology section (simplified)"""
    # Look for methodology indicators
    method_keywords = ["methods", "methodology", "approach", "implementation"]
    sentences = text.split(". ")

    method_sentences = []
    in_method_section = False

    for sentence in sentences:
        if any(keyword in sentence.lower() for keyword in method_keywords):
            in_method_section = True

        if in_method_section:
            method_sentences.append(sentence)
            if len(method_sentences) >= 5:  # Get ~5 sentences
                break

    return ". ".join(method_sentences) if method_sentences else None


def extract_results(abstract):
    """Extract results from abstract"""
    sentences = abstract.split(". ")
    result_keywords = [
        "result",
        "achieve",
        "performance",
        "accuracy",
        "improvement",
        "outperform",
    ]

    results = [s for s in sentences if any(kw in s.lower() for kw in result_keywords)]
    return ". ".join(results) if results else None


def split_train_val_test(input_file, train_ratio=0.8, val_ratio=0.1):
    """Split data into train/val/test sets"""
    with open(input_file, "r") as f:
        data = [json.loads(line) for line in f]

    random.shuffle(data)

    total = len(data)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)

    train_data = data[:train_size]
    val_data = data[train_size : train_size + val_size]
    test_data = data[train_size + val_size :]

    # Save splits
    base_path = Path(input_file).parent

    for split_name, split_data in [
        ("train", train_data),
        ("val", val_data),
        ("test", test_data),
    ]:
        output_file = base_path / f"{split_name}.jsonl"
        with open(output_file, "w") as f:
            for item in split_data:
                f.write(json.dumps(item) + "\n")
        print(f"{split_name}: {len(split_data)} examples")


def format_for_llama(input_file, output_file):
    """Format data for Llama instruction format"""
    with open(input_file, "r") as f:
        data = [json.loads(line) for line in f]

    formatted_data = []

    for item in data:
        # Llama 3.1 instruction format
        formatted = {
            "text": f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a scientific paper analysis assistant. Provide accurate, detailed responses based on the paper content.<|eot_id|><|start_header_id|>user<|end_header_id|>

{item["instruction"]}

Paper content:
{item["input"]}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{item["output"]}<|eot_id|>"""
        }
        formatted_data.append(formatted)

    with open(output_file, "w") as f:
        for item in formatted_data:
            f.write(json.dumps(item) + "\n")

    print(f"Formatted {len(formatted_data)} examples for Llama")


if __name__ == "__main__":
    # Run the complete pipeline
    print("=" * 50)
    print("Scientific Paper Dataset Preparation")
    print("=" * 50)

    # Step 1: Download and process arXiv data
    arxiv_file = download_arxiv_data(sample_size=50000)

    # Step 2: Create QA pairs
    qa_file = create_qa_pairs(num_samples=10000)

    # Step 3: Combine datasets
    print("\nCombining datasets...")
    combined_data = []
    for file in [arxiv_file, qa_file]:
        with open(file, "r") as f:
            combined_data.extend([json.loads(line) for line in f])

    combined_file = Path("./data/combined.jsonl")
    with open(combined_file, "w") as f:
        for item in combined_data:
            f.write(json.dumps(item) + "\n")

    # Step 4: Split into train/val/test
    print("\nSplitting data...")
    split_train_val_test(combined_file)

    # Step 5: Format for Llama
    print("\nFormatting for Llama 3.1...")
    for split in ["train", "val", "test"]:
        format_for_llama(f"./data/{split}.jsonl", f"./data/{split}_formatted.jsonl")

    print("\n" + "=" * 50)
    print("Dataset preparation complete!")
    print("=" * 50)
    print("\nReady files:")
    print("- ./data/train_formatted.jsonl")
    print("- ./data/val_formatted.jsonl")
    print("- ./data/test_formatted.jsonl")
