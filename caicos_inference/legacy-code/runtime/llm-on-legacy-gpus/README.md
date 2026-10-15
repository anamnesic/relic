# LLM on Legacy GPUs

> Minimal GGUF LLM runtime for OpenCL 1.2 GPUs  
> Running modern transformers on obsolete hardware.

---

## What is this?

LLM on Legacy GPUs is an experimental inference runtime designed to run modern transformer models on extremely old GPUs using OpenCL 1.2.

The project explores how far legacy hardware can be pushed for local AI inference.

Test target:

- AMD Radeon HD 6450
- AMD Caicos
- Terascale 2 GPUs
- 1GB VRAM class hardware

---

## Goals

- Run GGUF models on ancient GPUs
- Build a minimal transformer runtime from scratch
- Understand low-level LLM execution
- Experiment with OpenCL inference
- Make AI accessible on discarded hardware

---

## Current Features

✅ GGUF loader  
✅ GPT-2 tokenizer support  
✅ FP16 tensor loading  
✅ CPU backend  
✅ OpenCL 1.2 backend  
✅ RMSNorm  
✅ RoPE  
✅ GQA attention  
✅ KV cache  
✅ Autoregressive generation  
✅ SmolLM2 inference working

---

## Example

```bash
llm_on_legacy_gpus.exe \
  -m smollm2-135m.gguf \
  -p "The capital of France is" \
  -n 5 -t 0
```

Output:

```text
Paris.
```

---

## Why?

Modern inference frameworks usually assume:

* CUDA
* modern Vulkan
* tensor cores
* large VRAM
* recent GPUs

This project asks:

> Can transformers run on hardware everyone abandoned?

So far:

Yes.

---

## Supported Hardware

Designed around very old GPUs:

* AMD Terascale
* Radeon HD 5000/6000 series
* OpenCL 1.2 devices
* low VRAM GPUs

Also supports CPU fallback mode.

---

## Runtime Architecture

Implemented manually:

* GGUF parsing
* tensor loading
* tokenizer loading
* RMSNorm
* RoPE
* grouped-query attention (GQA)
* KV cache
* SwiGLU feed-forward
* sampling
* autoregressive decoding

No dependency on llama.cpp runtime execution.

---

## OpenCL Backend

Custom OpenCL execution path:

* GPU buffers
* tensor operations
* matmul kernels
* memory management
* fallback execution

Focused on compatibility over peak performance.

---

## Debugging Features

The runtime exposes internal transformer states:

```text
q_rms
k_rms
v_rms
attention RMS
layer hidden norms
top logits
```

Useful for:

* transformer research
* runtime debugging
* low-level AI experimentation
* learning how LLMs work internally

---

## Example Hardware

Current development machine:

```text
GPU: AMD Caicos
VRAM: 1GB
API: OpenCL 1.2
```

---

## Roadmap

* [ ] quantization support
* [ ] Vulkan backend
* [ ] Flash Attention experiments
* [ ] speculative decoding
* [ ] multi-GPU support
* [ ] CUDA backend
* [ ] Metal backend
* [ ] continuous batching
* [ ] MoE support

---

## Build

### Windows

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

---

## Usage

```bash
llm_on_legacy_gpus.exe -m model.gguf -p "Hello"
```

Options:

```text
--list-devices
--platform
--device
--cpu
--max-seq-len
```

---

## Philosophy

This project is not trying to beat llama.cpp.

The goal is:

* experimentation
* education
* accessibility
* low-level understanding
* reviving obsolete hardware

---

## Inspiration

Inspired by:

* llama.cpp
* ggml
* tinygrad
* candle
* llama2.c

---

## License

MIT