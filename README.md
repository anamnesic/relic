# Relic

> **LLM inference for hardware left behind.**
>
> Minimal GGUF LLM runtime for OpenCL 1.2 GPUs. Binary: `relic`.
> Running modern transformers on obsolete hardware.

`Relic` is part of [Anamnesic](https://github.com/anamnesic) — AI for hardware
left behind. It makes "relíquia" (relic) GPUs come back to life for local
inference.

---

## What is this?

**Relic** is an experimental inference runtime designed to run modern
transformer models on extremely old GPUs using OpenCL 1.2.

The project explores how far legacy hardware can be pushed for local AI
inference.

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

Validated on this machine: OpenCL kernels and numeric self-tests pass on the
NVIDIA GTX 1650 and Intel UHD Graphics. GGUF metadata from Ollama `gemma4:e2b`
loads successfully; inference is currently limited to `llama` architecture.

---

## Example

```bash
relic rebuild
pwsh scripts/relic.ps1 run -m smollm2-135m.gguf -p "The capital of France is" -n 5 -t 0
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

## Research Foundation (2026 Theses)

RELIC is architected according to recent literature in consumer-grade LLM inference and data-movement minimization:

* 📄 **[2026 Research Theses & Applied Roadmap](file:///C:/Users/luann/Documents/Anamnesic/relic/docs/RESEARCH_THESES_2026.md)** — In-depth breakdown of 8 papers (ATSInfer, Llamas on the Web, Gated DeltaNet, Fast NF4, Cache-Resident Inference, SAW-INT4, FluxBin, Linear Attention) and their implementation in RELIC.

---

## Roadmap

* [x] OpenCL 1.2 & OpenCL 3.0 Backends
* [x] Multi-Row 8x GEMV & Fused FFN (SwiGLU)
* [x] Hardware Micro-Profiler (`devices.json`)
* [x] Adaptive Memory Engine & Pinned Host Pools
* [x] Distributed Heterogeneous Speculation (NVIDIA + Intel UHD/CPU)
* [ ] ATSInfer Granular Tensor-Level Scheduler (`benefit_per_byte`)
* [ ] Static Memory Planning & Auto-Tuned Kernel Registry
* [ ] Fused Recurrent Operators (Gated DeltaNet / Qwen 3.5)
* [ ] Ultra-low-bit LUT backend (Q2/Q3 for 1GB VRAM)


---

## Build

### Windows

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

Or via the bundled script:

```bash
pwsh scripts/build.ps1
```

---

## Usage

```bash
# OpenCL device inventory
pwsh scripts/relic.ps1 probe

# GGUF metadata/tensors
pwsh scripts/relic.ps1 info model.gguf

# Inference
pwsh scripts/relic.ps1 run -m model.gguf -p "Hello"

# JSON wrapper report around an inference run
pwsh scripts/relic.ps1 bench -m model.gguf -p "Hello" -n 16
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
