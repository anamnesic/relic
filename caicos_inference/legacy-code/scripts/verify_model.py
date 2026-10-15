"""Verify SmolLM2-135M output against our implementation."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

model_path = "C:/Users/luann/Documents/GitHub/caicos_inference/llama.cpp/models/smollm2-135m.gguf"

# Load tokenizer from HuggingFace
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", trust_remote_code=True)

prompt = "The capital of France is"
encoded = tokenizer.encode(prompt)
print(f"Prompt: {prompt!r}")
print(f"Encoded tokens: {encoded}")
print(f"Decoded tokens: {[tokenizer.decode([t]) for t in encoded]}")

# Check specific tokens
for test_str in [" Paris", " Paris.", "Paris", "paris"]:
    t = tokenizer.encode(test_str)
    print(f"  encode({test_str!r}) -> {t[0]} = {tokenizer.decode([t[0]])!r}")

# Check what token ID 7042 is
print(f"\nToken 7042 decode: {tokenizer.decode([7042])!r}")
print(f"Token 912 decode: {tokenizer.decode([912])!r}")
print(f"Token 198 decode: {tokenizer.decode([198])!r}")
print(f"Token 307 decode: {tokenizer.decode([307])!r}")

# Test BOS token
print(f"BOS: {tokenizer.bos_token_id!r}, EOS: {tokenizer.eos_token_id!r}")
bos_enc = tokenizer.encode("<|im_start|>")
print(f"encode('<|im_start|>') -> {bos_enc}")

# Try model inference with pytorch (if we can load from GGUF)
# Actually let's load from HuggingFace directly
print("\nLoading model from HuggingFace...")
model = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM2-135M",
    trust_remote_code=True,
    torch_dtype=torch.float32
)
model.eval()

print(f"Model device: {next(model.parameters()).device}")
print(f"Model dtype: {next(model.parameters()).dtype}")

# Now run inference
inputs = tokenizer(prompt, return_tensors="pt")
print(f"\nInput IDs: {inputs['input_ids'].tolist()}")
print(f"Decoded input: {[tokenizer.decode([t]) for t in inputs['input_ids'][0].tolist()]}")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]  # last token's logits
    probs = torch.softmax(logits, dim=0)
    top5 = torch.topk(probs, 10)
    print("\nTop-10 predicted tokens:")
    for i in range(10):
        tid = top5.indices[i].item()
        print(f"  {tid}: {tokenizer.decode([tid])!r} (prob={top5.values[i].item():.4f}, logit={logits[tid].item():.1f})")
    # Check specific tokens
    for check in [" Paris", " Paris.", "Paris", "paris", "aw"]:
        tids = tokenizer.encode(check)
        if tids:
            tid = tids[0]
            print(f"  {check!r} (token {tid}): logit={logits[tid].item():.1f}, prob={probs[tid].item():.6f}")

print("\nDone!")
