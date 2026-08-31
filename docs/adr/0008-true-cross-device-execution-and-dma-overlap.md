# 8. True Cross-Device Execution, Double DMA Overlap, and Speculative Autoregressive Alignment

Date: 2026-08-31

## Status
Accepted

## Context
Following rigorous architectural critique of ADR 0007, four foundational hardware and semantic requirements were identified to complete the heterogeneous execution proof:
1. **Cross-Context OpenCL Bridge**: A `cl_mem` handle created on the Intel OpenCL platform/context cannot be submitted to an NVIDIA command queue. Cross-device operations must execute on the native Intel platform with host activation bridging or subgraph migration.
2. **GPU KV-Cache Snapshot & Rollback**: Attention steps mutate `gpu_k_caches` and `gpu_v_caches`. Rolling back rejected speculative drafts requires snapshotting and restoring KV-caches alongside SSM and Conv1D state.
3. **Speculative Autoregressive Alignment**: Autoregressive forward evaluates $D_i \to D_{i+1}$. Pre-draft logits verify candidate $D_0$, while draft batch forward output $D_i$ verifies candidate $D_{i+1}$.
4. **Guaranteed Page-Locked Memory**: `PinnedHostPool` must guarantee non-pageable physical RAM allocation using `VirtualLock` on Windows.
5. **Double DMA Overlap Pipeline**: Asynchronous DMA transfers on the dedicated DMA queue must overlap concurrently with GPU compute on the compute queue.

## Decision
1. **True Cross-Context Execution Bridge**: In `ArchitectureDecoder::dispatch_gemv()`, Intel UHD offloaded operations transfer activation data between NVIDIA GPU and Intel iGPU via host staging buffers, running compute kernels natively on `IntelUhdBackend`.
2. **GPU KV-Cache Snapshotting**: Allocated `gpu_k_snapshot_` and `gpu_v_snapshot_` buffers in `Qwen35DecoderAdapter`. `save_state_checkpoint()` and `restore_state_checkpoint()` now snapshot and restore all 4 state types (SSM matrices, Conv1D history, full KV-caches, activations).
3. **Aligned Speculative Verification**: In `InferenceEngine::generate()`, `drafts[0]` is verified against pre-draft `logits`. Upon match, `forward_batch()` is launched and `batched_logits[d]` verifies `drafts[d+1]`.
4. **Page-Locked Pinned Pool**: Updated `PinnedHostPool` to invoke `VirtualLock` (with process working set quota expansion).
5. **Empirical DMA Overlap Instrumentation**: Implemented concurrent double-buffered DMA/compute testing in `test_opencl.cpp` measuring standalone vs overlapped latency.
6. **Dimension-Aware Accounting**: Updated `AdaptivePlanner` to compute uncompressed bytes from dimensions when tensor data vectors are empty in synthetic tests.

## Consequences
- Heterogeneous execution across discrete GPU (GTX 1650) and integrated GPU (Intel UHD) is strictly cross-context compliant.
- Speculative verification guarantees zero state corruption and exact autoregressive alignment.
- Double-buffered DMA streaming achieves concurrent transfer/compute overlap.
