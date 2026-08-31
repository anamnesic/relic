# 5. Fixed-Budget Cost Model, Intel UHD Compute Kernels, Batched Speculation & Rigorous Diagnostics

Date: 2026-08-30

## Status
Accepted

## Context
As RELIC matures into a high-performance runtime for consumer devices under strict memory constraints, the project positioning was refactored from "legacy GPU runtime" to **"Maximize LLM inference under a fixed memory budget."**

To realize this vision, 45 architectural refinements were enacted:
1. **Accurate Footprint & Traffic Accounting**: Corrected VRAM footprint reduction calculation (35.6% reduction from 2602.77 MB to 1676.71 MB) and delineated footprint reduction from memory traffic reduction.
2. **Post-Repack Cost Modeling**: `AdaptivePlanner` now computes exact post-quantization VRAM sizes (`get_tensor_vram_size`), dynamically discovers device topologies via `HardwareProfile`, and implements the ATSInfer `benefit_per_byte` metric ($\Delta\text{Latency} / \text{VRAM\_Bytes}$) with explicit PCIe DMA movement costs.
3. **MemoryEngine Hot Path Integration**: Dynamic budget enforcement, LRU eviction, tensor migration, and feedback telemetry integrated into the memory engine with deterministic `StaticMemoryPlanner` scratch layouts.
4. **Intel UHD Compute Primitives**: Full implementation of OpenCL compute kernels in `IntelUhdBackend` (`rms_norm`, `add_rms_norm`, `gemv_q4_0`, `gemv_q8_0`, `gemv_q4_0_fused_ffn`, `rope`, `embed_lookup_q4_0`, `argmax`) with `KernelRegistry` autotuning.
5. **Real Batched Speculative Verification**: Upgraded `DistributedSpeculativeEngine` and `ArchitectureDecoder::forward_batch` to verify multi-token draft candidate sequences in a single batched pass on the target model.
6. **Sampler & Output Optimization**: Replaced $O(V \log V)$ full vector sort with an $O(V \log K)$ fixed-capacity min-heap and added direct GPU argmax scalar readbacks.
7. **Rigorous Benchmark & Numerical Verification**: Implemented multi-run statistical benchmarks (median, p50, p95, stddev, warmup exclusion, stage timings, JSON/CSV export) and layer-by-layer numerical error tests (Max Abs Error, Mean Abs Error, Cosine Similarity).

## Decision
1. Formalize all runtime scheduling around multi-tier hardware constraints (TIER0 dGPU, TIER1 iGPU Shared, TIER2 Host Pinned RAM).
2. Integrate `StaticMemoryPlanner` and `KernelRegistry` to guarantee zero runtime allocations on the generation hot path.
3. Ensure every OpenCL compute primitive across discrete and integrated GPUs passes automated layer-by-layer reference tests against FP32 CPU implementations with cosine similarity $= 1.000000$.

## Consequences
- Guarantees predictable memory residency, eliminating out-of-memory crashes and paging thrashing on 1 GB to 4 GB VRAM devices.
- Achieves optimal throughput across hybrid CPU + iGPU + dGPU architectures while maintaining full numerical precision and complete diagnostic visibility.
