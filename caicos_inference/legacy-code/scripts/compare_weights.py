"""Compare GGUF weights vs HuggingFace reference."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading reference model from HuggingFace...")
ref = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", trust_remote_code=True, dtype=torch.float32)
ref.eval()

# Get reference weights
ref_state = ref.state_dict()
print(f"Reference state dict keys ({len(ref_state)}):")
for k in list(ref_state.keys())[:10]:
    print(f"  {k}: {ref_state[k].shape}, {ref_state[k].dtype}")

# Check specific weights
print("\n--- Reference output.weight first row ---")
ow_key = [k for k in ref_state if 'lm_head' in k or 'output' in k]
print(f"  Output keys: {ow_key}")
if ow_key:
    ow = ref_state[ow_key[0]]
    print(f"  First 10: {ow[0, :10].tolist()}")
    print(f"  RMS: {torch.sqrt(torch.mean(ow**2)).item():.4f}")

# Check embedding
emb_key = [k for k in ref_state if 'embed' in k and 'weight' in k and 'position' not in k]
print(f"\n  Embedding keys: {emb_key}")
if emb_key:
    emb = ref_state[emb_key[0]]
    print(f"  Shape: {emb.shape}")
    # Token 504 = "The"
    the_emb = emb[504]
    print(f"  Token 504 first 10: {the_emb[:10].tolist()}")
    print(f"  RMS: {torch.sqrt(torch.mean(the_emb**2)).item():.4f}")

# Check layer 0 weights
print("\n--- Layer 0 weights ---")
for name in ['model.layers.0.self_attn.q_proj.weight',
             'model.layers.0.self_attn.k_proj.weight',
             'model.layers.0.self_attn.v_proj.weight',
             'model.layers.0.self_attn.o_proj.weight',
             'model.layers.0.input_layernorm.weight',
             'model.layers.0.mlp.gate_proj.weight',
             'model.layers.0.mlp.up_proj.weight',
             'model.layers.0.mlp.down_proj.weight',
             'model.layers.0.post_attention_layernorm.weight']:
    if name in ref_state:
        w = ref_state[name]
        print(f"  {name}: shape={w.shape}, first_5={w.flatten()[:5].tolist()}, rms={torch.sqrt(torch.mean(w**2)).item():.4f}")
    else:
        print(f"  {name}: NOT FOUND")

print("\nDone!")
