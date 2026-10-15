"""Full layer 0 computation in Python with GGUF weights, compared to C output."""
from gguf import GGUFReader
import torch
import numpy as np

r = GGUFReader(r'C:\Users\luann\Documents\GitHub\caicos_inference\llama.cpp\models\smollm2-135m.gguf')

weights = {}
for t in r.tensors:
    arr = t.data.astype(np.float32) if t.data.dtype == np.float16 else t.data
    weights[t.name] = arr

n_embd = 576
n_head = 9
n_kv_head = 3
n_embd_head = 64
eps = 1e-5

# Token 504 = "The" 
token_id = 504
emb = weights['token_embd.weight'][token_id]
print(f"Embedding token {token_id}: RMS={np.sqrt(np.mean(emb**2)):.4f}")
print(f"  first_5: {emb[:5].tolist()}")

def rms_norm(x, w, eps=1e-5):
    rms = np.sqrt(np.mean(x**2) + eps)
    return x * w / rms

def silu(x):
    return x / (1 + np.exp(-x))

def matmul_nt(a, b, M, N, K):
    """Our C matmul_nt: dst[i*N+j] = sum_k a[i*K+k] * b[j*K+k]"""
    a_np = a.reshape(M, K)
    b_np = b.reshape(N, K)
    dst = np.zeros((M, N), dtype=np.float32)
    for i in range(M):
        for j in range(N):
            s = 0.0
            for k in range(K):
                s += a_np[i, k] * b_np[j, k]
            dst[i, j] = s
    return dst.flatten()

# === Layer 0 ===
prefix = 'blk.0.'
l0 = {}

# Load all layer 0 weights
l0['attn_norm'] = weights[prefix + 'attn_norm.weight']
l0['ffn_norm'] = weights[prefix + 'ffn_norm.weight']

# QKV projection
q_w = weights[prefix + 'attn_q.weight']
k_w = weights[prefix + 'attn_k.weight']
v_w = weights[prefix + 'attn_v.weight']
o_w = weights[prefix + 'attn_output.weight']

# FFN weights
gate_w = weights[prefix + 'ffn_gate.weight']
up_w = weights[prefix + 'ffn_up.weight']
down_w = weights[prefix + 'ffn_down.weight']

# Check shapes
print(f"\n=== Layer 0 weights ===")
for name, w in [('Q', q_w), ('K', k_w), ('V', v_w), ('O', o_w), 
                ('gate', gate_w), ('up', up_w), ('down', down_w),
                ('attn_norm', l0['attn_norm']), ('ffn_norm', l0['ffn_norm'])]:
    print(f"  {name}: shape={w.shape}")

# === Step 1: RMS Norm ===
residual = emb.copy()
rms = np.sqrt(np.mean(residual**2) + eps)
normed = residual * l0['attn_norm'] / rms
print(f"\n=== Step 1: RMS Norm ===")
print(f"  residual RMS: {np.sqrt(np.mean(residual**2)):.4f}")
print(f"  normed RMS: {np.sqrt(np.mean(normed**2)):.4f}")
print(f"  first_5_resid: {residual[:5].tolist()}")
print(f"  first_5_normed: {normed[:5].tolist()}")
print(f"  first_5_weight: {l0['attn_norm'][:5].tolist()}")

# === Step 2: QKV projections ===
q = q_w @ normed  # shape (576,)
k = k_w @ normed  # shape (192,)
v = v_w @ normed  # shape (192,)
print(f"\n=== Step 2: QKV ===")
print(f"  Q RMS: {np.sqrt(np.mean(q**2)):.4f} first_5: {q[:5].tolist()}")
print(f"  K RMS: {np.sqrt(np.mean(k**2)):.4f} first_5: {k[:5].tolist()}")
print(f"  V RMS: {np.sqrt(np.mean(v**2)):.4f} first_5: {v[:5].tolist()}")

# Store KV in cache (for position 0)
k_cache = k.copy()
v_cache = v.copy()

# === Step 3: Attention (single token, self-attention) ===
S = 1  # one position in cache
q_heads = q.reshape(n_head, n_embd_head)  # (9, 64)
k_heads = k_cache.reshape(n_kv_head, n_embd_head)  # (3, 64)
v_heads = v_cache.reshape(n_kv_head, n_embd_head)  # (3, 64)

# Compute scores: each query head with its corresponding KV head
# GQA: Q has 9 heads, KV has 3 heads, so 3 queries per KV head
scores = np.zeros(n_head, dtype=np.float32)
for h in range(n_head):
    kv_h = h // (n_head // n_kv_head)  # 0,0,0,1,1,1,2,2,2
    s = np.dot(q_heads[h], k_heads[kv_h]) / np.sqrt(n_embd_head)
    scores[h] = s

print(f"\n=== Step 3: Attention ===")
print(f"  scores: {scores.tolist()}")

# Softmax
scores_max = np.max(scores)
scores_exp = np.exp(scores - scores_max)
scores_soft = scores_exp / np.sum(scores_exp)
print(f"  softmax scores: {scores_soft.tolist()}")

# Weighted sum of V
attn_out = np.zeros(n_embd, dtype=np.float32)
for h in range(n_head):
    kv_h = h // (n_head // n_kv_head)
    for d in range(n_embd_head):
        attn_out[h * n_embd_head + d] += scores_soft[h] * v_heads[kv_h, d]

print(f"  attn_out RMS: {np.sqrt(np.mean(attn_out**2)):.4f}")
print(f"  attn_out first_5: {attn_out[:5].tolist()}")

# === Step 4: Output projection ===
o = o_w @ attn_out
print(f"\n=== Step 4: Output Projection ===")
print(f"  O RMS: {np.sqrt(np.mean(o**2)):.4f} first_5: {o[:5].tolist()}")

# === Step 5: Residual ===
hidden = residual + o  # This is the original + O output
print(f"  hidden (after resid+O) RMS: {np.sqrt(np.mean(hidden**2)):.4f}")
print(f"  first_5: {hidden[:5].tolist()}")

# === Step 6: FFN RMS Norm ===
residual2 = hidden.copy()
rms2 = np.sqrt(np.mean(residual2**2) + eps)
ffn_normed = residual2 * l0['ffn_norm'] / rms2
print(f"\n=== Step 6: FFN RMS Norm ===")
print(f"  ffn_norm_in RMS: {np.sqrt(np.mean(residual2**2)):.4f}")
print(f"  normed RMS: {np.sqrt(np.mean(ffn_normed**2)):.4f}")

# === Step 7: FFN ===
gate = gate_w @ ffn_normed
up = up_w @ ffn_normed
silu_gate = silu(gate)
intermediate = silu_gate * up
down = down_w @ intermediate

print(f"\n=== Step 7: FFN ===")
print(f"  gate RMS: {np.sqrt(np.mean(gate**2)):.4f} first_5: {gate[:5].tolist()}")
print(f"  up RMS: {np.sqrt(np.mean(up**2)):.4f} first_5: {up[:5].tolist()}")
print(f"  down RMS: {np.sqrt(np.mean(down**2)):.4f} first_5: {down[:5].tolist()}")
print(f"  silu(gate) RMS: {np.sqrt(np.mean(silu_gate**2)):.4f}")

# === Step 8: FFN Residual ===
hidden = residual2 + down
print(f"\n=== Step 8: Final ===")
print(f"  hidden after FFN+residual RMS: {np.sqrt(np.mean(hidden**2)):.4f}")
print(f"  first_10: {hidden[:10].tolist()}")

# === Also try matmul_nt ===
print(f"\n=== Verify matmul_nt vs numpy ===")
q_nt = matmul_nt(normed, q_w, 1, n_embd, n_embd)
print(f"  matmul_nt Q: RMS={np.sqrt(np.mean(q_nt**2)):.4f} first_5={q_nt[:5].tolist()}")
print(f"  numpy Q:     RMS={np.sqrt(np.mean(q**2)):.4f} first_5={q[:5].tolist()}")
diff = np.abs(q_nt - q).max()
print(f"  max diff: {diff:.6f}")

# Compare with C output
c_vals = np.array([-1.211306, -0.200476, -0.648252, -0.936146, -0.774463])
print(f"\n  C code Q first_5: {c_vals.tolist()}")
print(f"  C vs matmul_nt: {np.abs(c_vals - q_nt[:5]).mean():.4f}")
