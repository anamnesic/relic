"""Dump hidden states from reference SmolLM2-135M for comparison."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", trust_remote_code=True)

prompt = "The capital of France is"
encoded = tokenizer.encode(prompt)
print(f"Prompt: {prompt!r}")
print(f"Encoded: {encoded}")

print("\nLoading model...")
model = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM2-135M",
    trust_remote_code=True,
    dtype=torch.float32
)
model.eval()

# Get hidden states from the last layer
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
    logits = outputs.logits[0, -1, :]
    hidden_states = outputs.hidden_states  # tuple of (layers + 1 embedding)

print(f"\nNumber of hidden states: {len(hidden_states)} (embedding + {len(hidden_states)-1} layers)")

# Final hidden state (before output projection)
final_hidden = hidden_states[-1][0, -1, :]  # [batch, seq_len, dim] -> last token
print(f"\nFinal hidden state (last token, last layer):")
print(f"  Shape: {final_hidden.shape}")
print(f"  RMS norm: {torch.sqrt(torch.mean(final_hidden**2)).item():.4f}")
print(f"  First 20: {final_hidden[:20].tolist()}")

# Output projection - check output.weight stats
output_weight = model.lm_head.weight if hasattr(model, 'lm_head') else model.output.weight
print(f"\nOutput weight: shape={output_weight.shape}")
print(f"  First row RMS: {torch.sqrt(torch.mean(output_weight[0]**2)).item():.4f}")
print(f"  First 5: {output_weight[0, :5].tolist()}")

# Compute logits manually from hidden state
manual_logits = torch.matmul(final_hidden, output_weight.T)
print(f"\nManual logits RMS: {torch.sqrt(torch.mean(manual_logits**2)).item():.4f}")
print(f"Model logits RMS: {torch.sqrt(torch.mean(logits**2)).item():.4f}")
print(f"Max diff: {torch.max(torch.abs(manual_logits - logits)).item():.6f}")

# Compare key logits
top5 = torch.topk(logits, 10)
print(f"\nTop-10 predicted tokens:")
for i in range(10):
    tid = top5.indices[i].item()
    print(f"  {tid}: {tokenizer.decode([tid])!r} (logit={logits[tid].item():.1f})")

# Check specific tokens
for check in [" the", " Paris", " Paris.", "Paris", "aw", "\n"]:
    tids = tokenizer.encode(check)
    if tids:
        tid = tids[0]
        print(f"  {check!r} (token {tid}): logit={logits[tid].item():.1f}")

# Also dump embedding norms for first token
embeds = model.model.embed_tokens(inputs['input_ids'])
print(f"\nEmbedding RMS norms per token:")
for i in range(embeds.shape[1]):
    rms = torch.sqrt(torch.mean(embeds[0, i]**2)).item()
    print(f"  Token {encoded[i]}: {tokenizer.decode([encoded[i]])!r} RMS={rms:.4f}")

print("\nDone!")
