# ADR 0021: Single-Pass Recurrent Projection Fusion

## Status
Accepted

## Date
2026-09-19

## Context

ADR 0019 fused the QKV and gate projections used by each Qwen 3.5 recurrent
layer. The recurrent path nevertheless continued to dispatch `ssm_alpha` and
`ssm_beta` as two independent GEMVs. All four projections consume the same
hidden-state vector, so the separate launches repeated activation staging and
queue submission for two very small output matrices.

For the 24-layer Qwen 3.5 topology with full attention every fourth layer, 18
layers use the recurrent path. The old schedule therefore paid for 36 avoidable
GEMV submissions per generated token. The arithmetic and weight traffic are
small relative to the main projection, but launch latency and repeated
activation-cache population remain visible costs in an autoregressive,
batch-one workload.

The measured 15.47 tok/s result from ADR 0019 should therefore not be treated
as a proven hardware ceiling. The remaining dominant limit is still Q4 weight
bandwidth, but dispatch overhead can be reduced further.

## Decision

When `attn_qkv`, `attn_gate`, `ssm_alpha`, and `ssm_beta` have compatible
quantization types and the alpha/beta matrices match the architecture shape,
Relic concatenates their rows into one resident tensor named
`attn_qkv_gate_alpha_beta.weight`.

The recurrent block dispatches a single `gemv_q4_0` over the combined matrix.
QKV stays at the beginning of the output buffer; gate, alpha, and beta slices
are copied asynchronously to their existing device buffers. No host readback or
queue-wide synchronization is introduced.

Compatibility is intentionally conservative:

- fusion is enabled only when all four tensors exist and share a type;
- alpha and beta must each contain `linear_value_heads * n_embd` elements;
- incompatible models retain the existing QKV+gate fusion and independent
  alpha/beta GEMVs;
- the recurrent scratch buffer reserves two additional value-head vectors.

## Performance Effect

For the target 24-layer Qwen 3.5 layout, the optimized schedule removes two
kernel submissions from each of 18 recurrent layers, or 36 GEMV submissions per
decoded token. It also avoids loading the hidden activation into local memory
twice per recurrent layer. The dot products and quantized weight bytes remain
unchanged, preserving numerical behavior.

The change was subsequently measured on the target GTX 1650 using the same
Qwen 3.5 2B GGUF and the benchmark suite's three measured runs after one
warmup. Median decode increased from the ADR 0019 baseline of 15.47 tok/s to
16.797 tok/s, an 8.58% improvement. The verification rerun measured 16.813,
16.797, and 16.797 tok/s with a standard deviation of 0.009 tok/s.

## Verification

- The complete CMake build succeeds for `relic`, `relic_test`, and
  `relic_dump_model`.
- `git diff --check` reports no whitespace errors.
- End-to-end GTX 1650 benchmark: 16.797 tok/s median decode and 17.048 tok/s
  median prompt ingestion for 16 generated tokens.
- The benchmark kept 1,676.71 MB of model weights resident in VRAM and reported
  no PCIe logit readback time.
- The CPU-only PoCL test path remains unsuitable for GPU numerical validation;
  GTX execution and the CUDA microbenchmark provide the target-device checks.

## Consequences

The batch-one recurrent schedule has less host/driver overhead and remains
backward-compatible. Further material decode gains are expected to require a
measured GPU-guided change—such as subgroup reduction specialization or a
different Q4 GEMV geometry—because the main projection and FFN kernels are now
primarily constrained by quantized weight bandwidth.
