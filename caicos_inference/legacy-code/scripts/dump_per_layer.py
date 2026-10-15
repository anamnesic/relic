"""Dump per-layer hidden states for comparison."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

print("Loading...")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", trust_remote_code=True, dtype=torch.float32)
model.eval()

prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states

# hidden_states[0] = embedding output
# hidden_states[1..30] = layer outputs (post-residual)
print(f"Number of hidden states: {len(hidden_states)} (embedding + {len(hidden_states)-1} layers)")

# For each token position, dump the hidden state after each layer
for pos in range(inputs['input_ids'].shape[1]):
    tok_id = inputs['input_ids'][0, pos].item()
    tok_str = tokenizer.decode([tok_id])
    print(f"\n=== Position {pos}: token {tok_id} = {tok_str!r} ===")
    # Embedding output
    emb = hidden_states[0][0, pos]
    emb_rms = torch.sqrt(torch.mean(emb**2)).item()
    print(f"  Embedding RMS={emb_rms:.4f} first_5={emb[:5].tolist()}")
    for layer_idx in range(min(2, len(hidden_states)-1)):
        h = hidden_states[layer_idx + 1][0, pos]
        rms = torch.sqrt(torch.mean(h**2)).item()
        print(f"  Layer {layer_idx} RMS={rms:.4f} first_5={h[:5].tolist()}")

# Final position (last token) - all layers
pos = inputs['input_ids'].shape[1] - 1
print(f"\n=== Final position {pos} all layers ===")
for layer_idx in range(len(hidden_states)-1):
    h = hidden_states[layer_idx + 1][0, pos]
    rms = torch.sqrt(torch.mean(h**2)).item()
    if layer_idx == 0 or layer_idx == 14 or layer_idx == 29:
        print(f"  Layer {layer_idx} RMS={rms:.4f} first_10={h[:10].tolist()}")

# Compare output weight
ow = model.lm_head.weight if hasattr(model, 'lm_head') else model.output.weight
print(f"\n=== Output weight ===")
print(f"  Shape: {ow.shape}")
print(f"  First row: {ow[0, :5].tolist()}")

print("\nDone!")
