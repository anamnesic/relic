"""Explicitly check tensor shapes and compare matmul results."""
from gguf import GGUFReader
import torch
import numpy as np
from transformers import AutoModelForCausalLM

r = GGUFReader(r'C:\Users\luann\Documents\GitHub\caicos_inference\llama.cpp\models\smollm2-135m.gguf')

# Load reference
ref = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", trust_remote_code=True, dtype=torch.float32)
ref_st = ref.state_dict()

# Token 504 embedding
print("=== Embedding Check ===")
emb_t = None
for t in r.tensors:
    if t.name == 'token_embd.weight':
        emb_t = t
        break

print(f"t.data.shape = {emb_t.data.shape}")
print(f"t.shape = {emb_t.shape}")

# Create numpy array from GGUF data
gguf_emb = emb_t.data.astype(np.float32)  # shape should be [49152, 576]
print(f"gguf_emb shape: {gguf_emb.shape}")

ref_emb = ref_st['model.embed_tokens.weight'].numpy()
print(f"ref_emb shape: {ref_emb.shape}")

# Compare token 504
gguf_token504 = gguf_emb[504]
ref_token504 = ref_emb[504]
diff = np.abs(gguf_token504 - ref_token504)
print(f"Token 504 max diff: {diff.max():.6f}, mean diff: {diff.mean():.6f}")
print(f"GGUF first_10: {gguf_token504[:10].tolist()}")
print(f"Ref  first_10: {ref_token504[:10].tolist()}")

# It's F16 precision, confirm
gguf_as_f16 = gguf_emb.astype(np.float16).astype(np.float32)
diff_f16 = np.abs(gguf_token504 - ref_token504)
print(f"F16 roundtrip max diff: {np.abs(gguf_as_f16[504] - ref_token504).max():.6f}")

print("\n=== Q projection weights ===")
qt = None
for t in r.tensors:
    if t.name == 'blk.0.attn_q.weight':
        qt = t
        break

print(f"Q: t.data.shape = {qt.data.shape}")
print(f"Q: t.shape = {qt.shape}")

gguf_q = qt.data.astype(np.float32)
print(f"gguf_q shape: {gguf_q.shape}")

ref_q = ref_st['model.layers.0.self_attn.q_proj.weight'].numpy()
print(f"ref_q shape: {ref_q.shape}")

# Compare first few elements
diff_q = np.abs(gguf_q.flatten() - ref_q.flatten())
print(f"Q max diff: {diff_q.max():.6f}, mean diff: {diff_q.mean():.6f}")
print(f"GGUF Q first_5: {gguf_q.flatten()[:5].tolist()}")
print(f"Ref  Q first_5: {ref_q.flatten()[:5].tolist()}")

# Check dimensions shape more carefully
print(f"\nGGUF Q data[0] to data[10]: {gguf_q.flatten()[:10].tolist()}")
print(f"Ref  Q data[0] to data[10]: {ref_q.flatten()[:10].tolist()}")
print(f"Reversed ref: {ref_q.T.flatten()[:10].tolist()}")

# Does GGUF match ref or ref.T?
rms_diff = np.sqrt(np.mean((gguf_q - ref_q)**2))
rms_diff_t = np.sqrt(np.mean((gguf_q - ref_q.T)**2))
print(f"\nRMS diff vs ref: {rms_diff:.6f}")
print(f"RMS diff vs ref.T: {rms_diff_t:.6f}")

# Check a column
col_idx = 5
gguf_col5 = gguf_q[:, col_idx]
ref_col5 = ref_q[:, col_idx]
print(f"\nQ column {col_idx}: GGUF vs Ref RMS diff: {np.sqrt(np.mean((gguf_col5 - ref_col5)**2)):.6f}")

# Compute expected Q
input_emb = ref_emb[504]  # use reference embedding
norm_weight = None
for t in r.tensors:
    if t.name == 'blk.0.attn_norm.weight':
        norm_weight = t.data.astype(np.float32)
        break
print(f"\nNorm weight shape: {norm_weight.shape}")
print(f"Norm weight first_5: {norm_weight[:5].tolist()}")

# RMS norm
eps = 1e-5
rms = np.sqrt(np.mean(input_emb**2) + eps)
normed = input_emb * norm_weight / rms
print(f"normed RMS: {np.sqrt(np.mean(normed**2)):.4f}")

# Q via reference formula: q = W @ hidden
q_ref = ref_q @ normed
print(f"Q (ref W) RMS: {np.sqrt(np.mean(q_ref**2)):.4f}")
print(f"Q (ref W) first_5: {q_ref[:5].tolist()}")

# Q via GGUF W
q_gguf = gguf_q @ normed
print(f"Q (gguf W) RMS: {np.sqrt(np.mean(q_gguf**2)):.4f}")
print(f"Q (gguf W) first_5: {q_gguf[:5].tolist()}")

# Our C code matmul: q[j] = sum_k normed[k] * W[j, k] (same as W @ normed)
print(f"\nQ (gguf W @ input) matches: {np.allclose(q_gguf, ref_q @ normed, atol=1e-3)}")

# Check if perhaps the weight should be transposed for our matmul
# C code: q[j] = sum_k hidden[k] * b[j * K + k] = b @ hidden
# Where b has dimensions [N, K] = [576, 576]
# b[j, k] = data[j * 576 + k]
# q[j] = sum_k hidden[k] * data[j * 576 + k]
# This is: q = data @ hidden (if data is [576, 576] and hidden is [576])
# Which is same as q_j = sum_k data[j][k] * hidden[k]

# But wait, that IS: q = W @ h with W shape [576, 576]
# Let's verify by comparing
q_via_gguf_W = gguf_q @ normed  # numpy matmul does (576,576) @ (576,) = (576,)
print(f"\nnumpy q_via_gguf_W: RMS={np.sqrt(np.mean(q_via_gguf_W**2)):.4f}, first_5={q_via_gguf_W[:5].tolist()}")

# Now what about our C code's actual output?
c_q_values = np.array([-1.211306, -0.200476, -0.648252, -0.936146, -0.774463])
print(f"\nC code Q first_5: {c_q_values}")
print(f"Python Q first_5 (ref W): {q_ref[:5].tolist()}")
print(f"Python Q first_5 (gguf W): {q_via_gguf_W[:5].tolist()}")

# Are they close?
print(f"\nC vs Python ref W diff: {np.abs(c_q_values - q_ref[:5]).mean():.4f}")
print(f"C vs Python gguf W diff: {np.abs(c_q_values - q_via_gguf_W[:5]).mean():.4f}")

# Try transposed matmul: q = W.T @ h (different interpretation)
q_via_gguf_W_T = gguf_q.T @ normed  # (576,576).T @ (576,) = (576,)
print(f"\nnumpy q_via_gguf_W.T: RMS={np.sqrt(np.mean(q_via_gguf_W_T**2)):.4f}, first_5={q_via_gguf_W_T[:5].tolist()}")

print(f"\nC vs Python gguf W.T diff: {np.abs(c_q_values - q_via_gguf_W_T[:5]).mean():.4f}")
