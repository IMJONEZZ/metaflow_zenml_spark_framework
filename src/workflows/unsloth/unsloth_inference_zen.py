# test_unsloth_model.py
import torch
from unsloth import FastLanguageModel

# Load fine-tuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./outputs/unsloth-llama70b-scientific/final_model",
    max_seq_length=8192,
    dtype=torch.bfloat16,
    load_in_4bit=True,
)

# Enable inference mode (2x faster)
FastLanguageModel.for_inference(model)

# Test
prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a scientific paper analysis assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

What is the main contribution of this paper?

[Paper content here]<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
