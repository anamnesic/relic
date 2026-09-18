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
- [x] **6. Fused Recurrent Operators for Hybrid Architectures** (*Gated DeltaNet - Jul/2026*):
  - [x] Fused DeltaNet recurrence operator (`qwen_gated_deltanet_fused`): State Update $\to$ In-Group RMSNorm $\to$ SiLU Gating executed in a single kernel launch.
- [x] **7. In-Kernel KV Quantization & LUT Ultra-Low-Bit** (*SAW-INT4 & FluxBin*):
  - [x] In-kernel FP16 KV cache append (`kv_cache_append_fp16`): Store half-precision K and V on-the-fly, halving cache VRAM footprint and memory bandwidth during attention.
  - [x] Quantized causal attention step (`qwen_full_attention_step_fp16`): On-the-fly dequantization in registers during dot-product attention.
  - [x] In-kernel SAW-INT4 KV Cache (`kv_cache_append_q4` / `qwen_full_attention_step_q4`) for 75% VRAM saving.
  - [x] FluxBin LUT-based ultra-low-bit Q2/Q3 dequantization helpers in OpenCL C 1.2.

---

### 🏛️ Phase 5: Hexagonal Architecture & Deep Refactoring
- [x] **1. InferenceEngine Dependency Inversion & Pure Domain Decoupling**:
  - [x] Remove `#include "opencl_backend.h"` from `src/inference.h`.
  - [x] Inject `std::unique_ptr<ArchitectureDecoder>` directly into `InferenceEngine` or provide an abstract `IDecoderFactory`.
  - [x] Eliminate concrete hardware driver references from the domain application service.
  - [x] Use `ArchitectureDecoder::supports_device_sampling` port query instead of concrete driver checks.
- [x] **2. Unified Quantization Utilities (`src/quantization/quant_utils.h`)**:
  - [x] Centralize FP16 conversion (`float_to_half_bits`, `float_to_half_val`, `half_bits_to_float`).
  - [x] Centralize block and row dequantization for Q8_0, Q4_0, and F32.
  - [x] Deduplicate `model_dequant_rows` across `Qwen35DecoderAdapter`, `model.h`, and `model.cpp`.
- [x] **3. Internal Decomposition of `Qwen35DecoderAdapter` (SRP & Cohesion)**:
  - [x] Extract `Qwen35WeightsManager` (`src/adapters/qwen35_weights_manager.{h,cpp}`): Weight uploading, VRAM placement, Q8_0 $\to$ Q4_0 on-the-fly repacking, and GEMV dispatch.
  - [x] Extract `Qwen35RecurrentBlock` (`src/adapters/qwen35_recurrent_block.{h,cpp}`): Gated DeltaNet SSM Conv1d and recurrent state steps.
  - [x] Extract `Qwen35AttentionBlock` (`src/adapters/qwen35_attention_block.{h,cpp}`): Full self-attention, RoPE, KV cache append, and scaled dot-product.
  - [x] Extract `Qwen35MlpBlock` (`src/adapters/qwen35_mlp_block.{h,cpp}`): SwiGLU FFN gating and down-projection.
- [x] **4. Unified Backend Device Interface**:
  - [x] Align `OpenClBackend` with `Backend` interface from `src/backends/backend.h`.
  - [x] Unify `ClBuffer` under `BackendBuffer`.
- [x] **5. Ubiquitous Domain Language**:
  - [x] Introduce `using NeuralModel = LlamaModel;` domain abstraction.
- [x] **6. Driving CLI Command Controllers (`src/cli/`)**:
  - [x] Extract CLI sub-commands from `src/main.cpp` into dedicated command adapters (`CliOptions`, `BenchmarkCommand`, `InferenceCommand`, `ServerCommand`).
  - [x] Reduce `src/main.cpp` to a clean ~180-line application bootstrap.
  - [x] Add automated unit tests for Driving & Driven Adapters in `src/test_opencl.cpp`.

---

### 🚀 Phase 6: Performance Engineering & Memory Bandwidth Saturation (Target: 35–60 tok/s)
- [x] **1. FFN SwiGLU Scale Factorization**:
  - [x] Factor `d_gate[r]` and `d_up[r]` out of inner loops in `gemv_q4_0_ffn_swiglu`, saving 512 FP multiplications per block per thread.
- [x] **2. Intra-Layer Kernel Fusion**:
  - [x] Fuse Residual Add + RMSNorm into `add_rms_norm` across Attention $\to$ FFN, FFN $\to$ Next Layer, and Final Layer $\to$ Logits.
  - [x] Eliminate 95 kernel launches and global memory round-trips per token across 24 layers.
- [x] **3. Constant Dot-Product Bias Precomputation**:
  - [x] Precompute $S_a = \sum_{k=0}^{31} a_k$ per block in `gemv_q4_0` and `gemv_q4_0_ffn_swiglu` to eliminate per-element $-8.0\text{f}$ subtractions in inner GEMV loops.
- [x] **4. Vectorized 128-Bit Memory Transactions & Register-Resident Dequantization in GEMV Q4_0**:
  - [x] Clustered 128-bit SIMD loads (`vload16`) for quants in `gemv_q4_0` and `gemv_q4_0_ffn_swiglu`, completely decoupling global memory transfers from the inner math pipeline.
  - [x] Fully unroll inner dequantization loops directly in hardware registers, replacing 64 iterative vector loads with register slice extractions (`.s0123`, `.s4567`, `.s89ab`, `.scdef`).
- [x] **5. Dynamic Local Memory Sizing for High Warp Occupancy**:
  - [x] Replace static `local float l_a[6144]` (24 KB) with dynamically sized kernel arguments `local float *l_a` sized to `K * sizeof(float)` in both `gemv_q4_0` and `gemv_q4_0_ffn_swiglu`.
  - [x] Reduce shared memory footprint per workgroup from 24 KB down to 8 KB for $K=2048$, increasing Turing SM 75 thread block occupancy from 2 workgroups up to 6–8 active workgroups per SM.
  - [x] Implement seamless hexagonal adapter support across both `OpenClBackend` and `IntelUhdBackend`.
- [x] **6. Multi-Row 16x Tiled GEMV Q4_0 & 4-Warp Workgroup Scaling**:
  - [x] Scale `gemv_q4_0` workgroup execution to 4 warps (128 threads) processing 16 rows per block (`l_sum[4][32]`).
  - [x] Double shared memory activation reuse across 16 rows, halving global-to-shared memory bandwidth overhead.
  - [x] Increase physical SM hardware occupancy to 768 threads per SM (75% theoretical hardware limit on NVIDIA Turing).
- [x] **7. Fused Recurrent Projections (QKV + Gate GEMV)**:
  - [x] Concatenate `attn_qkv` and `attn_gate` into a unified projection pass ($N = 8192, K = 2048$) in `Qwen35RecurrentBlock`.
  - [x] Eliminate 18 sequential GEMV dispatches and redundant activation loads from global memory per token.
- [x] **8. Multi-Row 16x Tiling in FFN SwiGLU (`gemv_q4_0_ffn_swiglu`)**:
  - [x] Scale workgroup size to 128 threads processing 16 rows per block, reducing FFN workgroups from 768 to 384.
- [x] **9. Batched Prefill GEMM Q4_0 (`gemm_q4_0`)**:
  - [x] 2D-tiled matrix multiplication ($M \times N \times K$) in OpenCL for prompt evaluation chunks, eliminating token-by-token prefill latency.
- [x] **10. Active Speculative Decoding Pipeline in `InferenceEngine`**:
  - [x] Integrate n-gram / draft model speculative step in autoregressive loop with parallel target model batched verification.
- [x] **11. Native K-Quants GEMV (`Q4_K` and `Q6_K`)**:
  - [x] Add OpenCL kernels for direct execution of GGUF k-quants without repacking.
- [x] **12. OpenAI-Compatible SSE Streaming Daemon (`/v1/chat/completions`)**:
  - [x] Implement Server-Sent Events HTTP streaming in `ServerCommand` for drop-in LLM client integration.

---

## 📊 Benchmark & Validation Milestones
- [x] Baseline GPU Port: `0.39 tok/s`
- [x] Multi-Row 8x & On-the-Fly Q4 Repack: `22.75 tok/s`
- [x] Fused FFN (Gate + Up + SwiGLU): `24.48 tok/s`
- [x] In-VRAM GPU Embedding Lookup & Skip Prompt Logits: `29.71 tok/s` Prompt
- [x] Vectorized 128-bit Memory Transactions & Dynamic Local Memory Occupancy: `13.95 tok/s` Decode (+18.6% speedup)
- [x] Multi-Row 16x Tiled GEMV & 4-Warp Concurrency: `14.67 tok/s` Decode (+24.7% cumulative speedup)
- [x] Fused QKV+Gate, Multi-Row 16x FFN SwiGLU & In-Kernel FP16 KV Cache: `15.47 tok/s` Decode (+31.5% cumulative speedup, 19.44 tok/s prefill)
- [x] Phase 1 Decoupled Engine validation (100% tests passed).
- [x] Phase 2 Async Prefetcher & Pinned Host Pool validation (100% tests passed).
- [x] Phase 3 Distributed Speculative Batched Verification validation (100% tests passed).
- [x] Phase 3 Intel UHD Compute Kernels & Zero-Copy Backend validation (100% tests passed).
- [x] Phase 4 ATSInfer Tensor Planner & Static Memory Plan on 4GB / 1GB VRAM targets.
- [x] Layer-by-layer Numerical Reference Self-Tests (100% tests passed).
