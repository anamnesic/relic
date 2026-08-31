# RELIC Roadmap & Master TODO

> **Relic: Maximize LLM inference under a fixed memory budget.**

---

## 🎯 Architecture Blueprint Overview

```text
                    ┌────────────────────────────┐
                    │         relic CLI          │
                    │ run / bench / profile      │
                    └─────────────┬──────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │       Model Runtime        │
                    │ GGUF / tokenizer / sampler │
                    │ transformer / generation   │
                    └─────────────┬──────────────┘
                                  │
              ┌───────────────────▼───────────────────┐
              │          Adaptive Planner             │
              │                                       │
              │ model profile + hardware profile      │
              │ memory budget + execution costs       │
              │ tensor placement + migration          │
              └───────┬───────────┬───────────┬───────┘
                      │           │           │
            ┌─────────▼───┐ ┌────▼─────┐ ┌───▼─────────┐
            │Memory Engine│ │Scheduler │ │KV Manager   │
            └──────┬──────┘ └────┬─────┘ └────┬────────┘
                   │             │            │
        ┌──────────▼─────────────▼────────────▼──────────┐
        │                Backend API                     │
        └───────┬────────────┬────────────┬───────────────┘
                │            │            │
         NVIDIA (Warp32)   Intel UHD     CPU (AVX2)
```

---

## 📋 Implementation Checklist

### 🏁 Phase 1: Decoupled Core & Abstract Backend API
- [x] **ADR 0003**: Formalize Modular Heterogeneous Runtime Architecture (`docs/adr/0003-modular-heterogeneous-runtime-architecture.md`).
- [x] **Backend Abstraction Interface** (`src/backends/backend.h`):
  - [x] `BackendBuffer` memory handles (Device VRAM, Host Pinned, Shared iGPU).
  - [x] Abstract compute primitives (`gemv_q4_0`, `gemv_q8_0`, `fused_ffn`, `rms_norm`, `rope`, `attention`, `deltanet`, `argmax`, `embed_lookup`).
  - [x] Device telemetry & statistics (`DeviceStats`).
- [x] **Hardware Profiler** (`src/planner/hardware_profile.h` / `src/planner/hardware_profile.cpp`):
  - [x] Measure VRAM capacity, PCIe DMA transfer rate (GB/s), and compute units per device.
  - [x] CLI command `relic --profile` outputting `devices.json`.
- [x] **Adaptive Planner Core** (`src/planner/adaptive_planner.h` / `src/planner/adaptive_planner.cpp`):
  - [x] Read Model Spec + Hardware Profile + Memory Budget.
  - [x] Compute per-tensor placement decisions (`ExecutionPlan`).
- [x] **Memory Engine Foundation** (`src/memory/memory_engine.h` / `src/memory/memory_engine.cpp`):
  - [x] Memory Tier allocation and tensor residency mapping.
- [x] **Adaptive KV Cache Manager** (`src/kv/kv_manager.h` / `src/kv/kv_manager.cpp`):
  - [x] Dynamic multi-layer KV cache allocation with FP16, Q8_0, and Q4_0 support.

---

### ⚡ Phase 2: Sub-Layer Offload & Async Prefetch Pipeline
- [x] **Memory Engine Advanced Pools** (`src/memory/`):
  - [x] `PinnedHostPool`: 64-byte aligned virtual memory allocator for PCIe DMA bursts.
  - [x] `AsyncPrefetcher`: Dual-slot VRAM staging buffers with non-blocking DMA queue (`compute(N) + copy(N+1)`).
- [x] **Sub-Layer Tensor Placement Decisions**:
  - [x] Pin critical Attention Q/K/V/Out matrices in VRAM.
  - [x] Offload bulky FFN weights to Host RAM with async prefetching when model exceeds VRAM budget (>4GB models on 4GB GPUs).
# RELIC Roadmap & Master TODO

> **Relic: Maximize LLM inference under a fixed memory budget.**

---

## 🎯 Architecture Blueprint Overview

```text
                    ┌────────────────────────────┐
                    │         relic CLI          │
                    │ run / bench / profile      │
                    └─────────────┬──────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │       Model Runtime        │
                    │ GGUF / tokenizer / sampler │
                    │ transformer / generation   │
                    └─────────────┬──────────────┘
                                  │
              ┌───────────────────▼───────────────────┐
              │          Adaptive Planner             │
              │                                       │
              │ model profile + hardware profile      │
              │ memory budget + execution costs       │
              │ tensor placement + migration          │
              └───────┬───────────┬───────────┬───────┘
                      │           │           │
            ┌─────────▼───┐ ┌────▼─────┐ ┌───▼─────────┐
            │Memory Engine│ │Scheduler │ │KV Manager   │
            └──────┬──────┘ └────┬─────┘ └────┬────────┘
                   │             │            │
        ┌──────────▼─────────────▼────────────▼──────────┐
        │                Backend API                     │
        └───────┬────────────┬────────────┬───────────────┘
                │            │            │
         NVIDIA (Warp32)   Intel UHD     CPU (AVX2)
```

---

## 📋 Implementation Checklist

### 🏁 Phase 1: Decoupled Core & Abstract Backend API
- [x] **ADR 0003**: Formalize Modular Heterogeneous Runtime Architecture (`docs/adr/0003-modular-heterogeneous-runtime-architecture.md`).
- [x] **Backend Abstraction Interface** (`src/backends/backend.h`):
  - [x] `BackendBuffer` memory handles (Device VRAM, Host Pinned, Shared iGPU).
  - [x] Abstract compute primitives (`gemv_q4_0`, `gemv_q8_0`, `fused_ffn`, `rms_norm`, `rope`, `attention`, `deltanet`, `argmax`, `embed_lookup`).
  - [x] Device telemetry & statistics (`DeviceStats`).
- [x] **Hardware Profiler** (`src/planner/hardware_profile.h` / `src/planner/hardware_profile.cpp`):
  - [x] Measure VRAM capacity, PCIe DMA transfer rate (GB/s), and compute units per device.
  - [x] CLI command `relic --profile` outputting `devices.json`.
- [x] **Adaptive Planner Core** (`src/planner/adaptive_planner.h` / `src/planner/adaptive_planner.cpp`):
  - [x] Read Model Spec + Hardware Profile + Memory Budget.
  - [x] Compute per-tensor placement decisions (`ExecutionPlan`).
- [x] **Memory Engine Foundation** (`src/memory/memory_engine.h` / `src/memory/memory_engine.cpp`):
  - [x] Memory Tier allocation and tensor residency mapping.
- [x] **Adaptive KV Cache Manager** (`src/kv/kv_manager.h` / `src/kv/kv_manager.cpp`):
  - [x] Dynamic multi-layer KV cache allocation with FP16, Q8_0, and Q4_0 support.

---

### ⚡ Phase 2: Sub-Layer Offload & Async Prefetch Pipeline
- [x] **Memory Engine Advanced Pools** (`src/memory/`):
  - [x] `PinnedHostPool`: 64-byte aligned virtual memory allocator for PCIe DMA bursts.
  - [x] `AsyncPrefetcher`: Dual-slot VRAM staging buffers with non-blocking DMA queue (`compute(N) + copy(N+1)`).
- [x] **Sub-Layer Tensor Placement Decisions**:
  - [x] Pin critical Attention Q/K/V/Out matrices in VRAM.
  - [x] Offload bulky FFN weights to Host RAM with async prefetching when model exceeds VRAM budget (>4GB models on 4GB GPUs).
- [x] **Adaptive KV Cache Manager Compression**:
  - [x] Pressure-aware dynamic KV quantization (FP16 $\to$ Q8 $\to$ Q4) under low VRAM conditions.
  - [x] Context budget and token eviction policies for ultra-long contexts.

---

### 🌟 Phase 3: Heterogeneity & Distributed Speculation
- [x] **Heterogeneous Speculative Engine** (`src/speculative/distributed_speculative.h` / `src/speculative/distributed_speculative.cpp`):
  - [x] Target Model Verifier on NVIDIA GTX 1650 (Real Batched Verification).
  - [x] Draft Model Generator on Intel UHD or CPU AVX2.
  - [x] Multi-device target-draft pipelining with verification metrics.
- [x] **Intel UHD OpenCL 3.0 Backend Specialized Compute Kernels** (`src/backends/intel_uhd_backend.h` / `src/backends/intel_uhd_backend.cpp`):
  - [x] Shared host memory zero-copy allocation on 11th Gen UHD Graphics.
  - [x] Full compute kernel suite: `rms_norm`, `add_rms_norm`, `gemv_q4_0`, `gemv_q8_0`, `gemv_q4_0_fused_ffn`, `rope`, `embed_lookup_q4_0`, `argmax`.

---

### 🔬 Phase 4: 2026 Research Foundation Milestones (`docs/RESEARCH_THESES_2026.md`)
- [x] **1. Tensor Planner & Granular Placement** (*ATSInfer - Jul/2026*):
  - [x] Per-tensor benefit score: $\text{benefit\_per\_byte} = \Delta \text{Latency} / \text{VRAM\_Bytes}$.
  - [x] Granular scheduling across Tier 0 (dGPU), Tier 1 (iGPU Shared), and Tier 2 (CPU RAM).
  - [x] Correct VRAM footprint reduction (35.6%) separated from memory traffic reduction.
- [x] **2. Static Memory Planning & Kernel Registry** (*Llamas on the Web - Mai/2026*):
  - [x] Deterministic activation buffer reuse graph (`StaticMemoryPlanner` - zero runtime driver allocations).
  - [x] `DeviceProfile` & `KernelRegistry` with startup micro-benchmark autotuning.
- [x] **3. Layer-by-Layer Numerical Verification Suite** (`src/numerical_verifier.h` / `src/numerical_verifier.cpp`):
  - [x] Exact reference checks for Hidden State, Norm, Gated DeltaNet, Attention, FFN, and Logits.
  - [x] Reports Max Absolute Error, Mean Absolute Error, and Cosine Similarity.
- [x] **4. Sampler & Output Optimization**:
  - [x] $O(V \log K)$ Min-Heap CPU Top-k sampler avoiding $O(V \log V)$ vector reallocations.
  - [x] GPU Argmax selection with scalar readback avoiding 248k float readbacks.
- [x] **5. Rigorous Automated Benchmark Suite** (`src/benchmark_suite.h` / `src/benchmark_suite.cpp`):
  - [x] Cold start, warmup exclusion, prefill, decode, and end-to-end breakdown.
  - [x] Median, p50, p95, and stddev statistics across multiple runs.
  - [x] Stage breakdown: GPU forward, PCIe readback, sampling, tokenizer, speculative verification.
  - [x] Export to JSON (`--bench-json`) and CSV (`--bench-csv`).
  - [x] Automated comparative benchmark table vs llama.cpp / Ollama format.
- [ ] **6. Fused Recurrent Operators for Hybrid Architectures** (*Gated DeltaNet - Jul/2026*):
  - [ ] Mega-fused kernel: State Update $\to$ RMSNorm $\to$ Gating $\to$ Out Projection.
- [ ] **7. In-Kernel KV Quantization & LUT Ultra-Low-Bit** (*SAW-INT4 & FluxBin*):
  - [ ] Inline rotation + quantization in attention kernels.
  - [ ] Experimental LUT-based Q2/Q3 backend for 1GB VRAM hardware.

---

## 📊 Benchmark & Validation Milestones
- [x] Baseline GPU Port: `0.39 tok/s`
- [x] Multi-Row 8x & On-the-Fly Q4 Repack: `22.75 tok/s`
- [x] Fused FFN (Gate + Up + SwiGLU): `24.48 tok/s`
- [x] In-VRAM GPU Embedding Lookup & Skip Prompt Logits: `29.71 tok/s` Prompt
- [x] Phase 1 Decoupled Engine validation (100% tests passed).
- [x] Phase 2 Async Prefetcher & Pinned Host Pool validation (100% tests passed).
- [x] Phase 3 Distributed Speculative Engine validation with Batched Verification (100% tests passed).
- [x] Phase 3 Intel UHD Compute Kernels & Zero-Copy Backend validation (100% tests passed).
- [x] Phase 4 ATSInfer Tensor Planner & Static Memory Plan on 4GB / 1GB VRAM targets.
- [x] Layer-by-layer Numerical Reference Self-Tests (100% tests passed).
