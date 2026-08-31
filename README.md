# Relic

> **Maximize LLM inference under a fixed memory budget.**
>
> Minimal, high-performance heterogeneous GGUF LLM runtime for OpenCL 1.2 & 3.0 devices. Binary: `relic`.

`Relic` is part of [Anamnesic](https://github.com/anamnesic) — AI for hardware with constrained memory. It optimizes data movement and hardware tiering across discrete GPUs (NVIDIA/AMD), integrated GPUs (Intel UHD), and host CPUs.

---

## What is this?

**Relic** is an inference runtime designed to maximize local LLM performance under strict memory constraints (1 GB to 4 GB VRAM class hardware, iGPUs, and hybrid heterogeneous topologies).

Target hardware topologies:

- Dedicated GPUs: NVIDIA GeForce GTX 1650, AMD Radeon / Caicos / Terascale
- Integrated GPUs: Intel UHD Graphics / Iris (Zero-Copy Unified Memory)
- Heterogeneous Hybrid: dGPU + iGPU + Host RAM Pinned DMA Pools

---

## Supported Architectures

✅ **Llama / SmolLM2** (Multi-Head Attention & Grouped-Query Attention)  
✅ **Qwen 3.5 / Gated DeltaNet** (Hybrid Linear Recurrent SSM + Full Attention Layers)  

---

## Current Features

✅ GGUF loader & metadata parser  
✅ GPT-2 and Byte-Pair Tokenizer support  
✅ FP16, Q8_0, and Q4_0 tensor loading with on-the-fly repacking  
✅ OpenCL 1.2 & 3.0 backends (NVIDIA CUDA OpenCL, Intel UHD OpenCL, CPU)  
✅ Vectorized RMSNorm & Fused Residual Add-RMSNorm  
✅ Rotary Position Embedding (RoPE)  
✅ GQA Attention & DeltaNet Recurrent Operators  
✅ Static Scratch & Activation Buffer Planning (Zero Hot-Path Allocations)  
✅ Adaptive Tensor Scheduling (`benefit_per_byte` ATSInfer Cost Model)  
✅ Distributed Speculative Decoding with Real Batched Verification  
✅ Min-Heap $O(V \log K)$ CPU Sampler & GPU Argmax Selection  

Validated on this machine: OpenCL compute kernels and layer-by-layer numerical error self-tests pass on both NVIDIA GeForce GTX 1650 and Intel(R) UHD Graphics.

---

## Research Foundation (2026 Theses)

RELIC is architected according to recent literature in consumer-grade LLM inference and data-movement minimization:

* 📄 **[2026 Research Theses & Applied Roadmap](docs/RESEARCH_THESES_2026.md)** — In-depth breakdown of 8 research papers (ATSInfer, Llamas on the Web, Gated DeltaNet, Fast NF4, Cache-Resident Inference, SAW-INT4, FluxBin, Linear Attention) and their implementation in RELIC.

---

## Roadmap

* [x] **OpenCL 1.2 & OpenCL 3.0 Backends** `[Implemented & Validated by Benchmark]`
* [x] **Multi-Row 8x GEMV & Fused FFN (SwiGLU)** `[Implemented & Validated by Benchmark]`
* [x] **Hardware Micro-Profiler (`devices.json`)** `[Implemented & Validated by Benchmark]`
* [x] **Intel UHD Graphics Compute Kernels** `[Implemented & Validated by Benchmark]`
* [x] **Adaptive Memory Engine & Pinned Host Pools** `[Integrated to Hot Path]`
* [x] **ATSInfer Granular Tensor-Level Scheduler (`benefit_per_byte`)** `[Integrated to Hot Path]`
* [x] **Static Memory Planning & Auto-Tuned Kernel Registry** `[Integrated to Hot Path]`
* [x] **Distributed Heterogeneous Speculation with Batched Verification** `[Implemented & Validated by Benchmark]`
* [x] **Fused Recurrent Operators (Gated DeltaNet / Qwen 3.5)** `[Scaffold & Foundation]`
* [ ] **Ultra-low-bit LUT backend (Q2/Q3 for 1GB VRAM)** `[In Progress]`

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
# OpenCL device inventory and hardware micro-profiling
pwsh scripts/relic.ps1 probe
relic.exe --profile

# Run layer-by-layer numerical error tests and self-test
relic_test.exe

# Model metadata inspection
pwsh scripts/relic.ps1 info model.gguf

# Single inference run
pwsh scripts/relic.ps1 run -m model.gguf -p "Hello"

# Rigorous automated benchmark suite with statistical metrics
relic.exe -m model.gguf -p "Hello" --bench --bench-json bench_results.json --bench-csv bench_results.csv
```

---

## License

MIT

