# train_with_unsloth.py
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

# Configuration
max_seq_length = 8192
model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
output_dir = "./outputs/unsloth-llama70b-scientific"

print("=" * 60)
print("Unsloth Fine-Tuning: Llama 3.1 70B for Scientific Papers")
print("=" * 60)

# Load model with Unsloth optimizations
print("\n[1/5] Loading model with Unsloth...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=torch.bfloat16,
    load_in_4bit=True,
    device_map="auto",
)

print(f"Model loaded. Memory usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Add LoRA adapters
print("\n[2/5] Adding LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
    random_state=42,
)

# Count trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(
    f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)"
)

# Load dataset
print("\n[3/5] Loading dataset...")
dataset = load_dataset(
    "json",
    data_files={
        "train": "./data/train_formatted.jsonl",
        "validation": "./data/val_formatted.jsonl",
    },
)

print(f"Training examples: {len(dataset['train']):,}")
print(f"Validation examples: {len(dataset['validation']):,}")

# Training arguments
print("\n[4/5] Configuring training...")
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    logging_steps=10,
    eval_steps=100,
    save_steps=500,
    save_total_limit=3,
    bf16=True,
    optim="adamw_torch",
    weight_decay=0.01,
    max_grad_norm=1.0,
    report_to="wandb",  # Optional: set to "none" to disable
    run_name="unsloth-llama70b-scientific",
    gradient_checkpointing=True,
    # DGX Spark optimizations
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
)

# Create trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=training_args,
    packing=False,  # Don't pack multiple examples together
)

# Enable native Flash Attention 2 (if available)
model.config.use_cache = False  # Required for gradient checkpointing

print("\n[5/5] Starting training...")
print("-" * 60)

# Train!
trainer.train()

# Save final model
print("\n[✓] Training complete! Saving model...")
model.save_pretrained(f"{output_dir}/final_model")
tokenizer.save_pretrained(f"{output_dir}/final_model")

print(f"\nModel saved to: {output_dir}/final_model")
print("=" * 60)
print("Fine-tuning complete!")
print("=" * 60)
