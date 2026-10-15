"""Load GGUF weights into PyTorch and compute layer 0 output step by step."""
from gguf import GGUFReader
import torch
import numpy as np

# Load GGUF weights
r = GGUFReader(r'C:\Users\luann\Documents\GitHub\caicos_inference\llama.cpp\models\smollm2-135m.gguf')

# Read all tensors
weights = {}
for t in r.tensors:
    if t.data.dtype == np.float16:
        weights[t.name] = torch.from_numpy(t.data.astype(np.float32))
    else:
        weights[t.name] = torch.from_numpy(t.data)

# Parameters
n_embd = 576
n_head = 9
n_kv_head = 3
n_embd_head = 64
eps = 1e-5

# Token 504 embedding
token = 504
# Use token_embd.weight
emb_w = weights['token_embd.weight']  # shape [576, 49152]
print(f"token_embd.weight shape: {emb_w.shape}")

# Output projection (output.weight)
out_w = weights['output.weight']
print(f"output.weight shape: {out_w.shape}")
print(f"output.weight first_5: {out_w[:5].tolist() if out_w.shape[0] == 49152 else out_w.flatten()[:5].tolist()}")

# Embedding for token 504
embedding = emb_w[:, token]  # This depends on layout
print(f"emb_w[:, 504] shape: {embedding.shape}")
print(f"emb_w[:, 504] first_10: {embedding[:10].tolist()}")

# Try embedding = emb_w[token, :]
# First check shape: if [576, 49152] means 576 rows, 49152 cols
# numpy shape[0]=576, shape[1]=49152
# So emb_w[row, col] = data[row * 49152 + col]
# embedding[token] would need row=token, col=0..575
# emb_w[token, :] = data[token * 49152 + d]

emb_b = weights['token_embd.weight'][token, :]
print(f"emb_w[504, :] first_10: {emb_b[:10].tolist()}")

# The C code does: memcpy(emb, embedding_data + token * n_embd, ...)
# In numpy indexing: data[504 * 576 + d] for d in 0..575
# In the numpy tensor shape [576, 49152]:
# data[504 * 576 + d] = data[290304 + d]
# This is element at row 290304//49152=5, col 290304%49152=44544
# = emb_w[5, 44544] for d=0

flat = emb_w.flatten()
c_emb = flat[token * n_embd : token * n_embd + 10]
print(f"C code embedding (flat[{token}*{n_embd}:...]): {c_emb.tolist()}")

# Now check: which layout makes sense?
# In PyTorch/HuggingFace: embed_tokens.weight shape = [49152, 576]
# embedding = embed_tokens.weight[token]
# In GGUF with shape [576, 49152]:
# - numpy convention: [rows, cols], data[row, col] = flat[row*49152 + col]
# - For embedding lookup, we want the 504-th row (in PyTorch sense)
# - If GGUF stores it transposed: GGUF[row, col] = HuggingFace[col, row]
# - So token 504's embedding dim d = GGUF[d, 504] = flat[d * 49152 + 504]

transposed_emb = flat[:n_embd * 49152].reshape(n_embd, 49152)
col_emb = [float(transposed_emb[d, 504]) for d in range(10)]
print(f"Column lookup emb[d, 504]: {col_emb}")

# Reference
from transformers import AutoModelForCausalLM
ref = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", trust_remote_code=True, dtype=torch.float32)
ref_st = ref.state_dict()
ref_emb = ref_st['model.embed_tokens.weight']
print(f"\nReference emb[504]: {ref_emb[504, :10].tolist()}")

# Which matches?
for name, vals in [("C code (flat[504*576+d])", c_emb.tolist()),
                    ("Transposed (flat[d*49152+504])", col_emb),
                    ("emb_w[504,:]", emb_b[:10].tolist())]:
    ref_np = ref_emb[504, :len(vals)].numpy()
    diff = np.abs(np.array(vals) - ref_np).mean()
    print(f"  {name}: mean_diff={diff:.6f}")

# Now run the actual computation with the correct weights
print("\n=== Running computation with GGUF weights ===")

# The correct embedding is the one that matches reference
embedding = ref_emb[504]  # use reference embedding for now

def rms_norm(x, w, eps=1e-5):
    rms = torch.sqrt(torch.mean(x * x) + eps)
    return (x / rms) * w

def matmul_nt_cpu(dst, a, b, M, N, K):
    """CPU matmul matching C code: dst = a @ b (where b is (N, K))"""
    a_np = a.numpy() if isinstance(a, torch.Tensor) else a
    b_np = b.numpy() if isinstance(b, torch.Tensor) else b
    # C code: dst[i*N+j] = sum_k a[i*K+k] * b[j*K+k]  (b is transposed)
    # This is: dst = a @ (b^T) = a @ b.T
    # But the C code transposes b on the fly
    # Equivalent to: dst[i,j] = sum_k a[i,k] * b[j,k]
    # dst = a @ b^T

    # Actually, the C matmul_nt is: dst[i*N+j] = sum_k a[i*K+k] * b[j*K+k]
    # Here `b` is used with index [j*K + k], meaning b is treated as (N, K) and NOT transposed
    # But the access pattern b[j * K + k] with j=0..N-1 means the second dimension is K
    # and the first dimension is N.
    # So b has shape (N, K) in row-major.
    # Result: dst = a @ b^T where a is (M, K) and b is (N, K)
    # Wait, let me re-read: dst[i*N + j] = sum_k a[i*K + k] * b[j*K + k]
    # a is indexed as (M, K): a[i*K + k] rows=i cols=k
    # b is indexed as (N, K): b[j*K + k] rows=j cols=k
    # dst is (M, N): dst[i*N + j] rows=i cols=j
    # This IS: dst[i,j] = sum_k a[i,k] * b[j,k]
    # Which is: dst = a @ b.T where a is (M, K) and b is (N, K)
    # So if a is (1, K) and b is (N, K), dst = a @ b.T
    # That means dst[j] = sum_k a[k] * b[j, k]
    return a_np @ b_np.T

# Layer 0 weights
l0 = {}
prefix = 'blk.0.'
for gguf_name, short_name in [('attn_q.weight', 'q'),
                                ('attn_k.weight', 'k'),
                                ('attn_v.weight', 'v'),
                                ('attn_output.weight', 'o'),
                                ('attn_norm.weight', 'attn_norm'),
                                ('ffn_norm.weight', 'ffn_norm'),
                                ('ffn_gate.weight', 'gate'),
                                ('ffn_up.weight', 'up'),
                                ('ffn_down.weight', 'down')]:
    key = prefix + gguf_name
    if key in weights:
        l0[short_name] = weights[key]
        print(f"  {short_name}: shape={weights[key].shape}")

# Step 1: RMS norm
print("\n=== Step 1: RMS Norm ===")
residual = embedding.numpy()
rms_val = np.sqrt(np.mean(residual**2) + eps)
normed = (residual / rms_val) * l0['attn_norm'].numpy()
print(f"residual RMS: {np.sqrt(np.mean(residual**2)):.4f}")
print(f"normed RMS: {np.sqrt(np.mean(normed**2)):.4f}")
print(f"norm weight RMS: {np.sqrt(np.mean(l0['attn_norm'].numpy()**2)):.4f}")
print(f"first_5_normed: {normed[:5].tolist()}")

# Step 2: Q projection
print("\n=== Step 2: Q Projection ===")
q = matmul_nt_cpu(None, normed.reshape(1, -1), l0['q'].numpy(), 1, n_embd, n_embd)
# Or maybe: q = normed @ l0['q'].T
q2 = normed @ l0['q'].numpy().T
print(f"Q (matmul_nt): RMS={np.sqrt(np.mean(q**2)):.4f} first_5={q.flatten()[:5].tolist()}")
print(f"Q (normed @ W.T): RMS={np.sqrt(np.mean(q2**2)):.4f} first_5=q2[:5].tolist()")

# Try direct multiplication
q3 = l0['q'].numpy() @ normed
print(f"Q (W @ normed): RMS={np.sqrt(np.mean(q3**2)):.4f} first_5=q3[:5].tolist()")
