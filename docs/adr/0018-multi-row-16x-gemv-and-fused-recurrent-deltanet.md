# ADR 0018: Multi-Row 16x Tiled GEMV Q4_0 and Fused Recurrent DeltaNet Execution

## Status
Accepted

## Date
2026-09-18

## Context
Following the 128-bit memory transactions and dynamic local memory sizing in ADR 0017 (which achieved 13.95 tok/s), profiling and literature review of linear attention / hybrid state space models (Gated DeltaNet) and GEMV scaling identified the next level of micro-architectural optimizations:

1. **Activation Cache Traffic Redundancy**:
   In `gemv_q4_0`, workgroups were tiled with 2 warps (64 threads) processing 8 rows per workgroup. For a weight matrix with $N = 6144$ rows (e.g. FFN down projection or QKV projection), 768 independent thread blocks were dispatched, each loading the entire activation vector $a$ of length $K=2048$ into shared memory. This meant that the activation vector had to be read from global memory into L1/shared memory 768 times per token.
2. **SM Warp Concurrency Underutilization**:
   On NVIDIA Turing (GTX 1650, sm_75), each SM can accommodate up to 1024 concurrent active threads. With 64 threads per workgroup and 6 workgroups active, only 384 threads were active per SM (37.5% occupancy).
3. **Formalization of Phase 4.6 (Fused Recurrent DeltaNet)**:
   The Gated DeltaNet recurrent block in Qwen 3.5 requires L2 normalization of $Q$ and $K$, scalar state update with decay $g$ and beta $b$, in-group RMSNorm, and SiLU gating. Consolidating this entire pipeline into [`qwen_gated_deltanet_fused`](file:///home/luann/projects/anamnesic/relic/kernels/kernels.cl#L1095) eliminates intermediate VRAM allocations and round-trips.

## Decision

### 1. Multi-Row 16x Tiled GEMV Q4_0 (`gemv_q4_0`)
- Scaled workgroup execution in [`kernels/kernels.cl`](file:///home/luann/projects/anamnesic/relic/kernels/kernels.cl#L308) to **4 warps (128 threads)** processing **16 rows per block** (`base_row = get_group_id(0) * 16 + warp_id * 4`).
- Expanded warp reduction storage to `local float l_sum[4][32]` (consuming only 2 KB of shared memory across all 4 warps).
- Reused the dynamically loaded shared activation vector `l_a` across 16 rows per block, halving the number of thread blocks dispatched (from 768 down to 384 for $N=6144$) and cutting global memory traffic for activations by 50%.
- Increased SM warp occupancy to **768 active threads per SM** (75% of physical hardware maximum).

### 2. Secondary Driven Adapter Synchronization
- Synchronized workgroup sizing across both secondary hardware drivers:
  - [`OpenClBackend::gemv_q4_0`](file:///home/luann/projects/anamnesic/relic/src/opencl_backend.cpp#L486): Dispatches with `local = 128` and `n_groups = (N + 15) / 16`.
  - [`IntelUhdBackend::gemv_q4_0`](file:///home/luann/projects/anamnesic/relic/src/backends/intel_uhd_backend.cpp#L293): Dispatches with `local = 128` and `n_groups = (N + 15) / 16`.
- Maintained strict bounds checking (`if (row < N)`) on writes and safe pointer aliasing (`(row < N) ? b + offset : b`) to ensure memory safety on arbitrary matrix dimensions.

### 3. Hexagonal Architecture Preservation
- Core domain interfaces ([`Backend`](file:///home/luann/projects/anamnesic/relic/src/backends/backend.h), [`ArchitectureDecoder`](file:///home/luann/projects/anamnesic/relic/src/decoder.h), [`InferenceEngine`](file:///home/luann/projects/anamnesic/relic/src/inference.h)) remain completely unchanged and decouple domain business logic from thread block geometry.

## Verification & Results

1. **Throughput Scaling**:
   - Generation latency decreased from 1792.27 ms down to **1704.47 ms** for 25 tokens.
   - Generation throughput increased to **14.67 tok/s** (+24.7% cumulative speedup from 11.76 tok/s baseline).
   - Total runtime reduced to 2030.19 ms.

2. **Numerical Integrity**:
   - `relic_test` verification suite passed 100%:
     - `[layer.norm.rms_norm]` CosSim: `1.000000` -> PASS
     - `[layer.norm.add_rms_norm]` CosSim: `1.000000` -> PASS
     - `[layer.recurrent.deltanet_conv1d]` CosSim: `1.000000` -> PASS
     - `[layer.ffn.swiglu]` CosSim: `1.000000` -> PASS
     - `[model.layer.00..03.hidden_state]` CosSim: `1.000000` -> PASS
     - `Driving Adapter (CliOptions Parsing)` -> PASS
     - `Driven Adapter (Backend Interface & BackendBuffer Polymorphism)` -> PASS

3. **Output Text Verification**:
   - Prompt: `"The capital of France is"`
   - Output: `" Paris.\n\n<think>\n\n</think>\n\nYes, that is correct. **Paris** is the capital city of France. It is"`
   - Exact semantic match and zero token degradation.
