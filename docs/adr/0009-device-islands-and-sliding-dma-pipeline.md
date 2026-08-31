# 9. Device Islands, Sliding DMA Pipeline, and Over-Budget Inference

Date: 2026-08-31

## Status
Accepted

## Context
ADR 0008 established cross-context OpenCL compatibility, KV-cache snapshotting, and speculative verification alignment. However, granular per-GEMV activation bouncing and non-pipelined DMA transfers introduced avoidable overheads:
1. **Per-GEMV Crossing Penalty**: Offloading individual alternating GEMVs between GTX 1650 and Intel UHD incurred excessive activation round-trips over PCIe.
2. **Buffer Allocation Overhead**: Dynamic buffer allocation on Intel UHD during every GEMV caused memory manager churn.
3. **Sequential Staging Stalls**: Pinned host prefetching was executed sequentially rather than pipelining layer $N+1$ DMA transfer concurrently with layer $N$ GPU kernel execution.
4. **KV Rollback Waste**: Copying full sequence KV caches during speculative rollbacks introduced megabyte-scale copies.

## Decision
1. **Device Islands Partitioning**: `AdaptivePlanner` groups contiguous layers into *Device Islands* and introduces `activation_boundary_cost_ms` to penalize device hopping, favoring large continuous layer blocks on secondary devices.
2. **Persistent Intel UHD Buffers**: Pre-allocated fixed activation buffers `uhd_in_buf_` and `uhd_dst_buf_` once at initialization, eliminating all runtime buffer allocations on the iGPU.
3. **Pipelined Hot-Path DMA Overlap**: Implemented double-buffered asynchronous streaming where DMA transfer of layer $N+1$ runs on the dedicated DMA command queue while layer $N$ compute kernels execute on the GPU queue.
4. **Strict Concurrency Assertion**: Validated DMA overlap in `test_opencl.cpp` requiring $T_{\text{overlapped}} < T_{\text{sequential}}$, reporting measured concurrency efficiency.
5. **Zero-Copy Tail KV-Cache Rollback**: Checkpointed sequence position index `n_past` rather than copying bulk KV buffers, achieving $O(1)$ instant speculative rollback.
6. **Constant 128 MB Sliding Staging Window**: Configured `PinnedHostPool` as a 128 MB dual-slot sliding cache backed by `VirtualLock` with transparent logging.

## Consequences
- Heterogeneous partition graphs minimize PCIe communication by forming contiguous multi-layer device islands.
- Speculative verification has negligible rollback latency.
- DMA transfers achieve high concurrency with GPU compute.
