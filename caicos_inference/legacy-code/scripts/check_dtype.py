from gguf import GGUFReader
r = GGUFReader(r'C:\Users\luann\Documents\GitHub\caicos_inference\llama.cpp\models\smollm2-135m.gguf')

t0 = r.tensors[0]
print("t0.name:", t0.name)
print("t0.shape:", t0.shape)
print("t0.data dtype:", t0.data.dtype)
print("t0.n_elements:", t0.n_elements)

dtypes = {}
for t in r.tensors:
    dtypes[t.data.dtype] = dtypes.get(t.data.dtype, 0) + 1
print("Tensor dtype counts:", dtypes)

for tn in ["output.weight", "token_embd.weight", "blk.0.attn_q.weight", "blk.0.attn_norm.weight"]:
    for t in r.tensors:
        if t.name == tn:
            flat = t.data.flatten()[:8]
            print(f"{tn}: shape={t.shape} dtype={t.data.dtype} first_8={[float(x) for x in flat]}")
            break
    else:
        print(f"{tn}: NOT FOUND")

ft = r.fields.get("general.file_type")
print(f"File type data: {ft.data}")
print(f"Total tensors: {len(r.tensors)}")
