# ADR 0016: GEMV Constant Bias Precomputation and Hexagonal Adapter Unit Testing

## Status
Accepted

## Date
2026-09-18

## Context
Following the unifications in ADR 0015, profiling identified two remaining opportunities:

1. **Quantization Bias Redundancy in Matrix-Vector Multiplication**:
   In standard GGML Q4_0 quantization, weights $w_i$ are represented as $w_i = d \cdot (q_i - 8)$ where $q_i \in [0..15]$.
   Both [`gemv_q4_0`](file:///home/luann/projects/anamnesic/relic/kernels/kernels.cl#L378) and [`gemv_q4_0_ffn_swiglu`](file:///home/luann/projects/anamnesic/relic/kernels/kernels.cl#L532) computed `convert_float4(qb & 0x0F) - 8.0f` for every single byte of every quantized row.
   In `gemv_q4_0` (4 rows per group), this executed 128 scalar subtractions per 32-element block per thread. In `gemv_q4_0_ffn_swiglu` (16 rows: 8 gate + 8 up), this executed 512 scalar subtractions per block per thread.
2. **Hexagonal Driving & Driven Adapter Test Gap**:
   While the core domain had extensive numerical self-tests in `src/numerical_verifier.cpp`, the newly created driving CLI adapters (`src/cli/`) and the polymorphic backend bridge lacked automated regression assertions in `src/test_opencl.cpp`.

## Decision

### 1. Algebraic Identity for Constant Bias Elimination
- Rewrote the block dot-product using the algebraic expansion:
  $$\sum_{k=0}^{31} a_k (q_k - 8) = \left(\sum_{k=0}^{31} a_k q_k\right) - 8 \sum_{k=0}^{31} a_k$$
- Precomputed the scalar sum of activations $S_a = \sum_{k=0}^{31} a_k$ once per block directly in registers during the vector load phase (requiring only 7 additions).
- Inside the inner loop over all rows, accumulated pure positive integer dot products `dot(a_lo, convert_float4(qb & 0x0F))` without subtracting $8.0\text{f}$.
- At the end of the block, applied the precomputed constant bias once per row:
  $$( \text{block\_acc} - 8.0\text{f} \cdot S_a ) \cdot d$$
- Eliminated 128 subtractions per block in `gemv_q4_0` and 512 subtractions per block in `gemv_q4_0_ffn_swiglu`.

### 2. Automated Testing for Hexagonal Driving & Driven Adapters
- Added automated regression testing in [`src/test_opencl.cpp`](file:///home/luann/projects/anamnesic/relic/src/test_opencl.cpp):
  - **Driving Adapter Test**: Validates that `CliOptions` parses mock command line arguments (`-m`, `-p`, `-n`, `-t`, `-k`, `--budget-mb`, `--server`, `--client`) into strongly typed domain parameters with zero side-effects.
  - **Driven Adapter Test**: Validates polymorphic interaction through abstract `Backend&` and `BackendBuffer` handles (`query_stats()`, `allocate()`, `upload()`, `download()`, and round-trip data integrity).

## Verification & Results
1. **Numerical Fidelity**:
   - `relic_test` reports 100% tests passed:
     - `[layer.norm.rms_norm]` CosSim: `1.000000` -> PASS
     - `[layer.norm.add_rms_norm]` CosSim: `1.000000` -> PASS
     - `[layer.recurrent.deltanet_conv1d]` CosSim: `1.000000` -> PASS
     - `[layer.ffn.swiglu]` CosSim: `1.000000` -> PASS
     - `[model.layer.00..03.hidden_state]` CosSim: `1.000000` -> PASS
     - `Driving Adapter (CliOptions Parsing)` -> PASS
     - `Driven Adapter (Backend Interface & BackendBuffer Polymorphism)` -> PASS
2. **Text Generation**:
   - Prompt: `"The capital of France is"`
   - Output: `" Paris.\n\n<think>\n\n</think>\n\nYes, that is correct. **Paris** is the capital city of France. It is"`
   - Perfect semantic fidelity and zero token degradation.
