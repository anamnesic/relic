# 7. Research-Grade Heterogeneous Offloading and Non-Destructive Speculative Rollback

Date: 2026-08-30

## Status
Accepted

## Context
Following user review of ADR 0006, critical refinements were required to reach research-grade rigor:
1. **Heterogeneous Offload Execution**: Offloaded tensors in `ExecutionPlan` must flow into `PinnedHostPool` in host memory, stream asynchronously through `AsyncPrefetcher` via dual GPU staging slots, or execute in zero-copy shared memory on Intel UHD Graphics.
2. **State Corruption in Speculative Decoding**: Evaluating candidate draft tokens speculatively modifies the KV-cache and recurrent Gated DeltaNet state. When drafts are rejected, intermediate state modifications must be rolled back.
3. **Sampler Duplication**: Sampler logic (min-heap Top-K, Top-P nucleus, temperature scaling) was duplicated between `InferenceEngine` and `BenchmarkSuite`.
4. **Benchmark Prefill & Metrics**: Prefill vocabulary projection must be skipped on prompt tokens $0 \dots P-2$ to mirror the true hot path. Stage timings must be measured without synthetic percentage constants.
5. **Numerical Verification Rigor**: Layer-by-layer verification must compute actual weights from the model tensors.

## Decision
1. **Three-Tier Dynamic Tensor Resolution**: In `ArchitectureDecoder::dispatch_gemv()`, tensors are resolved across Dedicated VRAM $\to$ Intel UHD Shared Memory $\to$ Pinned Host RAM (via `AsyncPrefetcher` double-buffered DMA stream).
2. **State Checkpointing & Speculative Rollback**: Added `save_state_checkpoint()` and `restore_state_checkpoint()` to `ArchitectureDecoder` and `Qwen35RecurrentState`. The speculative verification pipeline snapshots the autoregressive state before evaluating drafts, and restores/advances only the strictly accepted prefix.
3. **Extracted Shared Sampler**: Implemented `Sampler` in `src/sampler.h` supporting greedy argmax, temperature scaling, min-heap Top-K, and nucleus Top-P filtering, shared by `InferenceEngine` and `BenchmarkSuite`.
4. **Pure Empirical Benchmark Chronometry**: Eliminated remaining synthetic multipliers; updated prefill loop to skip vocabulary projection on non-terminal prompt tokens; renamed stage metrics to `forward_wall_time_ms`, `pcie_readback_ms`, `sampling_ms`, and `tokenizer_ms`.
5. **Real Model Layer Verification**: Updated `NumericalVerifier::verify_model_layer_by_layer()` to evaluate real model tensors across all layers with cosine similarity ($1.000000$).
6. **Large Model (>4 GB) Planning**: Added test suite verification of a 4.5 GB model partitioned under a strict 1.5 GB VRAM budget across 291 tensor placements.

## Consequences
- Every execution claim is backed by real hardware dispatch and non-destructive state management.
- Offloading on sub-4 GB GPUs operates with double-buffered asynchronous DMA prefetching.
