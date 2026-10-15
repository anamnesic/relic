"""Check GGUF embedding lookup vs reference."""
from gguf import GGUFReader
import struct

r = GGUFReader(r'C:\Users\luann\Documents\GitHub\caicos_inference\llama.cpp\models\smollm2-135m.gguf')

# Find token_embd.weight
for t in r.tensors:
    if t.name == 'token_embd.weight':
        print(f"token_embd.weight shape={t.shape} dtype={t.data.dtype}")
        # t.shape = [576, 49152] (rows=576, cols=49152)
        n_embd = t.shape[1]  # 576? NO, shape is [576, 49152]
        n_vocab = t.shape[0]  # 576? NO
        
        if len(t.shape) == 2:
            rows, cols = t.shape[0], t.shape[1]
        else:
            rows, cols = t.shape[0], 1
            
        print(f"Rows={rows}, Cols={cols}")
        
        # Method 1: If row=token, col=embd_dim (standard layout)
        token_id = 504
        stride_col = cols  # row stride in elements
        
        emb_method1 = []
        for d in range(5):
            idx = token_id * stride_col + d
            emb_method1.append(float(t.data.flatten()[idx]))
        print(f"Method 1 (data[{token_id}*{stride_col} + d]): first_5={emb_method1}")
        
        # Method 2: If row=embd_dim, col=token (transposed)
        emb_method2 = []
        for d in range(5):
            idx = d * cols + token_id
            emb_method2.append(float(t.data.flatten()[idx]))
        print(f"Method 2 (data[d*{cols} + {token_id}]): first_5={emb_method2}")
        
        break

# Also dump what our code would read (from dequantized tensor)
# We look up embedding via hidden[token * n_embd + d]
# where n_embd=576, and we read the tensor as-is
n_embd = 576

# Read raw file at the embedding tensor offset
with open(r'C:\Users\luann\Documents\GitHub\caicos_inference\llama.cpp\models\smollm2-135m.gguf', 'rb') as f:
    pass  # Too complex to parse manually

# Compare with reference
import torch
from transformers import AutoModelForCausalLM
ref = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", trust_remote_code=True, dtype=torch.float32)
ref_st = ref.state_dict()

# Reference embedding
ref_emb = ref_st['model.embed_tokens.weight']
print(f"\nReference embedding shape: {ref_emb.shape}")
print(f"Reference token 504 first_10: {ref_emb[504, :10].tolist()}")

# Check GGUF output.weight
for t in r.tensors:
    if t.name == 'output.weight':
        print(f"\noutput.weight shape={t.shape} dtype={t.data.dtype}")
        flat = t.data.flatten()[:10]
        print(f"GGUF first_10: {[float(x) for x in flat]}")
        
        # Reference
        ref_ow = ref_st['lm_head.weight']
        print(f"Reference output.weight first_10: {ref_ow[0, :10].tolist()}")
        break

# Check if it's the same model with F16 differences
print("\n=== Comparing output.weight first row ===")
for t in r.tensors:
    if t.name == 'output.weight':
        gguf_vals = [float(x) for x in t.data.flatten()[:5]]
        ref_vals = ref_st['lm_head.weight'][0, :5].tolist()
        print(f"GGUF:  [{', '.join(f'{v:.10f}' for v in gguf_vals)}]")
        print(f"Ref:   [{', '.join(f'{v:.10f}' for v in ref_vals)}]")
        
        # Check if GGUF values are F16 versions of ref values
        import numpy as np
        gguf_as_f16 = np.array(gguf_vals, dtype=np.float16)
        print(f"GGUF as F16: [{', '.join(f'{v:.10f}' for v in gguf_as_f16.astype(np.float32))}]")
        
        ref_np = np.array(ref_vals, dtype=np.float32)
        ref_as_f16 = ref_np.astype(np.float16).astype(np.float32)
        print(f"Ref as F16:  [{', '.join(f'{v:.10f}' for v in ref_as_f16)}]")
        print(f"GGUF ≈ Ref F16? {np.allclose(np.array(gguf_vals, dtype=np.float32), ref_as_f16, atol=1e-3)}")
        break
