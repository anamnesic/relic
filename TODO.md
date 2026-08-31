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
- [x] **Adaptive Planner Initial Core** (`src/planner/adaptive_planner.h` / `src/planner/adaptive_planner.cpp`):
  - [x] Read Model Spec + Hardware Profile + Memory Budget.
  - [x] Compute per-tensor placement decisions (`ExecutionPlan`).
- [x] **Memory Engine Foundation** (`src/memory/memory_engine.h` / `src/memory/memory_engine.cpp`):
  - [x] Memory Tier allocation and tensor residency mapping.
- [x] **Adaptive KV Cache Manager** (`src/kv/kv_manager.h` / `src/kv/kv_manager.cpp`):
  - [x] Dynamic multi-layer KV cache allocation with FP16, Q8_0, and Q4_0 support.

---

### ⚡ Phase 2: Sub-Layer Offload & Async Prefetch Pipeline
- [ ] **Memory Engine Advanced Pools** (`src/memory/`):
  - [ ] `VramPoolAllocator`: Fast sub-allocation without runtime driver overhead.
  - [ ] `PinnedHostPool`: Zero-copy DMA-mapped host staging buffers.
  - [ ] `AsyncPrefetcher`: Overlap Layer $N$ computation with Layer $N+1$ PCIe DMA transfers (`compute(N) + copy(N+1)`).
- [ ] **Sub-Layer Tensor Placement Execution**:
  - [ ] Pin critical Attention Q/K/V/Out matrices in VRAM.
  - [ ] Offload bulky FFN weights to Host RAM with async prefetching when model exceeds VRAM budget (>4GB models on 4GB GPUs).
- [ ] **Adaptive KV Cache Manager Compression**:
  - [ ] Pressure-aware dynamic KV quantization (FP16 $\to$ Q8 $\to$ Q4) under low VRAM conditions.
  - [ ] Context budget and token eviction policies for ultra-long contexts.

---

### 🌟 Phase 3: Heterogeneity & Distributed Speculation
- [ ] **Intel UHD Backend** (`src/backends/opencl_intel/`):
  - [ ] OpenCL 3.0 shared host memory buffer allocation for Intel 11th Gen UHD Graphics.
- [ ] **Heterogeneous Speculative Engine** (`src/speculative/`):
  - [ ] Target Model Verifier on NVIDIA GTX 1650 (Batched verification).
  - [ ] Draft Model Generator on Intel UHD or CPU AVX2.
  - [ ] Pipelined asynchronous speculative decoding with zero host stalls.

---

### 📊 Benchmark & Validation Milestones
- [x] Baseline GPU Port: `0.39 tok/s`
- [x] Multi-Row 8x & On-the-Fly Q4 Repack: `22.75 tok/s`
- [x] Fused FFN (Gate + Up + SwiGLU): `24.48 tok/s`
- [x] In-VRAM GPU Embedding Lookup & Skip Prompt Logits: `29.71 tok/s` Prompt
- [x] Phase 1 Decoupled Engine validation (100% tests passed).
- [ ] Phase 2 Sub-layer Offload on >4GB models (e.g. 7B/8B Q4 on 4GB GTX 1650).
- [ ] Phase 3 Distributed Speculative Engine beating Ollama (>35 tok/s).
