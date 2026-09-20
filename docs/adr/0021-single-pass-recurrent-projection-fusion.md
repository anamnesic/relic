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

This is a latency optimization rather than a claim of a new measured token
rate. The development host exposes only a PoCL CPU device and cannot reproduce
the GTX 1650 benchmark environment. A statistically comparable GPU run is
required before recording a new tok/s milestone.

## Verification

- The complete CMake build succeeds for `relic`, `relic_test`, and
  `relic_dump_model`.
- `git diff --check` reports no whitespace errors.
- The local `relic_test` executable cannot complete on the available PoCL CPU:
  the monolithic OpenCL program uses FP16 kernels unsupported by that device,
  kernel creation fails, and the pre-existing test harness later exits with a
  segmentation fault. This environment limitation prevents numerical and
  performance claims for the new GPU path.

## Consequences

The batch-one recurrent schedule has less host/driver overhead and remains
backward-compatible. Further material decode gains are expected to require a
measured GPU-guided change—such as subgroup reduction specialization or a
different Q4 GEMV geometry—because the main projection and FFN kernels are now
primarily constrained by quantized weight bandwidth.
