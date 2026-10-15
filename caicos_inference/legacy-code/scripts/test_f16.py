"""Test HuggingFace model with F16 weights to see if F16 quantization explains errors."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", trust_remote_code=True, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", trust_remote_code=True)

prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt")

# Test in F16
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False, temperature=0.0)
    result_f16 = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"F16: {result_f16}")

# Test in F32
model_f32 = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", trust_remote_code=True, torch_dtype=torch.float32)
with torch.no_grad():
    outputs = model_f32.generate(**inputs, max_new_tokens=5, do_sample=False, temperature=0.0)
    result_f32 = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"F32: {result_f32}")

# Compare logits
with torch.no_grad():
    out_f16 = model(**inputs, output_hidden_states=True)
    logits_f16 = out_f16.logits[0, -1]
    print(f"\nF16 token 260 ( the): {logits_f16[260].item():.2f}")
    print(f"F16 token 7042 ( Paris): {logits_f16[7042].item():.2f}")
    print(f"F16 token 912 (aw): {logits_f16[912].item():.2f}")
    top5 = torch.topk(logits_f16, 5)
    print(f"F16 top-5: {[(t.item(), i.item()) for t,i in zip(top5.values, top5.indices)]}")
    
    out_f32 = model_f32(**inputs, output_hidden_states=True)
    logits_f32 = out_f32.logits[0, -1]
    print(f"\nF32 token 260 ( the): {logits_f32[260].item():.2f}")
    print(f"F32 token 7042 ( Paris): {logits_f32[7042].item():.2f}")
    print(f"F32 token 912 (aw): {logits_f32[912].item():.2f}")
    top5 = torch.topk(logits_f32, 5)
    print(f"F32 top-5: {[(t.item(), i.item()) for t,i in zip(top5.values, top5.indices)]}")
    
    # F16 vs F32 difference
    diff = (logits_f16 - logits_f32).abs().mean()
    print(f"\nF16 vs F32 mean logit diff: {diff.item():.2f}")
