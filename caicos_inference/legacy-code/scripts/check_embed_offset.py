"""Directly check embedding at offset used by the C code."""
from gguf import GGUFReader
import torch
from transformers import AutoModelForCausalLM

r = GGUFReader(r'C:\Users\luann\Documents\GitHub\caicos_inference\llama.cpp\models\smollm2-135m.gguf')

# Find token_embd.weight
emb_t = None
for t in r.tensors:
    if t.name == 'token_embd.weight':
        emb_t = t
        break

print(f"token_embd.weight: shape={emb_t.shape} dtype={emb_t.data.dtype}")
print(f"shape[0]={emb_t.shape[0]}, shape[1]={emb_t.shape[1]}")

# The C code reads: embedding_data + token_id * n_embd
# where n_embd = 576
# So for token 504: offset = 504 * 576 = 290304
n_embd = 576
token_id = 504
offset = token_id * n_embd

all_data = emb_t.data.flatten()
print(f"offset = {token_id} * {n_embd} = {offset}")
print(f"Total elements: {all_data.shape[0]}")

c_embedding = [float(all_data[offset + d]) for d in range(10)]
print(f"C code embedding (offset={offset}): {c_embedding}")

# Also check if shape [576, 49152] means 576 rows and 49152 cols
# If so, embedding for token T should be all elements where dim1 = T
# i.e., all data[i * 49152 + T] for i in 0..575
n_vocab = emb_t.shape[1]  # 49152 assuming shape is [rows, cols] in numpy convention
alt_offset = token_id  # first element
alt_offset2 = token_id * n_embd  # different interpretation
alt_offset3 = n_embd * token_id  # same as above

# Try different interpretations
print(f"\nInterpretations:")
print(f"  A) data[{token_id} * {n_embd} + d]: data[{offset} + d] = {c_embedding[:5]}")

alt_emb1 = [float(all_data[d * n_vocab + token_id]) for d in range(min(5, n_embd))]
print(f"  B) data[d * {n_vocab} + {token_id}]: = {alt_emb1}")

alt_emb2 = [float(all_data[token_id * n_vocab + d]) for d in range(min(5, n_embd))]
print(f"  C) data[{token_id} * {n_vocab} + d]: = {alt_emb2}")

# Reference
ref = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", trust_remote_code=True, dtype=torch.float32)
ref_st = ref.state_dict()
ref_emb = ref_st['model.embed_tokens.weight']
print(f"\nReference shape: {ref_emb.shape}")
print(f"Reference token 504 embedding: {ref_emb[504, :10].tolist()}")
print(f"Reference f16: {ref_emb[504, :10].to(torch.float16).to(torch.float32).tolist()}")

# Which interpretation matches the reference most closely?
import numpy as np
ref_np = ref_emb[504, :10].numpy()

i_a = np.array(c_embedding[:5])
i_b = np.array(alt_emb1[:5])
i_c = np.array(alt_emb2[:5])

for name, vals in [("A (C code)", i_a), ("B (column)", i_b), ("C (row*49152)", i_c)]:
    diff = np.abs(vals - ref_np[:len(vals)]).mean()
    print(f"  {name}: mean_diff={diff:.6f}")
