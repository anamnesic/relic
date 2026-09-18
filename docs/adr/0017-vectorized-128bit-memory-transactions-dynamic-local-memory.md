# ADR 0017: Vectorized 128-Bit Memory Transactions and Dynamic Local Memory Sizing for High Warp Occupancy

## Status
Accepted

## Date
2026-09-18

## Context
Following the bias precomputations in ADR 0016, profiling on the target hardware (NVIDIA GeForce GTX 1650 Mobile, Turing sm_75, 4 GB GDDR5, 14 SMs with 64 KB shared memory each) identified two architectural bottlenecks in GEMV Q4_0:

1. **Shared Memory Static Allocation Bottleneck**:
   In `kernels/kernels.cl`, [`gemv_q4_0`](file:///home/luann/projects/anamnesic/relic/kernels/kernels.cl#L308) statically allocated `local float l_a[6144]`, consuming 24,576 bytes (24 KB) of shared memory per workgroup regardless of layer dimensions.
   Because the GTX 1650 SM shared memory capacity is 64 KB, 24.5 KB per workgroup strictly restricted occupancy to at most 2 workgroups (128 threads / 4 warps) per SM. For $K=2048$ layers (which constitute the majority of projections in Qwen 2.5/3.5), 16 KB of local memory was wasted per workgroup.

2. **Scattered Global Memory Vector Loads**:
   In both `gemv_q4_0` and `gemv_q4_0_ffn_swiglu`, quantized nibbles were loaded in 4 separate 32-bit `vload4` instructions interleaved inside the inner arithmetic loop across 8–16 rows. This resulted in up to 64 independent global memory load instructions per block, causing pipeline stalls and preventing the hardware memory controller from issuing coalesced 128-bit transactions (`LDG.E.128`).

## Decision

### 1. Dynamic Local Memory Sizing
- Refactored [`gemv_q4_0`](file:///home/luann/projects/anamnesic/relic/kernels/kernels.cl#L308) and [`gemv_q4_0_ffn_swiglu`](file:///home/luann/projects/anamnesic/relic/kernels/kernels.cl#L469) to receive `local float *l_a` as an OpenCL kernel argument.
- In both [`OpenClBackend`](file:///home/luann/projects/anamnesic/relic/src/opencl_backend.cpp) and [`IntelUhdBackend`](file:///home/luann/projects/anamnesic/relic/src/backends/intel_uhd_backend.cpp), dynamically configured the shared memory allocation via `clSetKernelArg(..., (size_t)K * sizeof(float), nullptr)`.
- For $K=2048$, shared memory consumption dropped from 24 KB to 8 KB per workgroup, multiplying potential SM thread block occupancy from 2 up to 6–8 active workgroups per SM.

### 2. Clustered 128-Bit SIMD Memory Transactions (`vload16`)
- Replaced iterative 32-bit `vload4` global loads with clustered 128-bit `vload16` transactions at the top of each block:
  ```c
  uchar16 qb0 = vload16(0, qs0);
  uchar16 qb1 = vload16(0, qs1);
  uchar16 qb2 = vload16(0, qs2);
  uchar16 qb3 = vload16(0, qs3);
  ```
- Unrolled the dequantization and dot-product pipeline completely in private registers using OpenCL vector slice selectors (`.s0123`, `.s4567`, `.s89ab`, `.scdef`).
- Clustered all 16 vector loads in `gemv_q4_0_ffn_swiglu` (`qg[0..7]` and `qu[0..7]`) to maximize memory-level parallelism (MLP) on the GDDR5 bus.

### 3. Hexagonal Architecture Preservation
- **Domain Core**: Clean separation preserved; `ArchitectureDecoder` and `InferenceEngine` remain device-agnostic.
- **Driven Secondary Adapters**: Both `OpenClBackend` and `IntelUhdBackend` implement the dynamic local memory interface without leaking hardware details to callers.

## Verification & Results

1. **Throughput Speedup**:
   - Generation latency decreased from 2126.50 ms down to 1792.27 ms for 25 tokens.
   - Decode generation throughput increased from **11.76 tok/s to 13.95 tok/s (+18.6% speedup)**.

2. **Numerical Fidelity**:
   - All 9 test suites in `relic_test` passed 100%:
     - `[layer.norm.rms_norm]` CosSim: `1.000000` -> PASS
     - `[layer.norm.add_rms_norm]` CosSim: `1.000000` -> PASS
     - `[layer.recurrent.deltanet_conv1d]` CosSim: `1.000000` -> PASS
     - `[layer.ffn.swiglu]` CosSim: `1.000000` -> PASS
     - `[model.layer.00..03.hidden_state]` CosSim: `1.000000` -> PASS
     - `Driving Adapter (CliOptions Parsing)` -> PASS
     - `Driven Adapter (Backend Interface & BackendBuffer Polymorphism)` -> PASS

3. **Text Generation Integrity**:
   - Prompt: `"The capital of France is"`
   - Output: `" Paris.\n\n<think>\n\n</think>\n\nYes, that is correct. **Paris** is the capital city of France. It is"`
   - Exact match with zero degradation.
