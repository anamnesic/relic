# ADR 0024: Web-Search-First Error Protocol and Reference-Driven Debugging

## Status
Accepted

## Date
2026-10-06

## Context

RELIC is a ground-up GGUF runtime: it re-implements quantization, RoPE,
attention, the Gated DeltaNet recurrence and the OpenCL kernels that run them.
Any of those can be wrong in a way that is invisible in isolation and only
shows up as fluent-looking garbage at decode time. A wrong `Q6_K` dequant,
an unscaled key vector, or a missing `ssm_norm` gate all produce output text
rather than a crash, so "it runs" is not evidence of correctness.

Two earlier debugging sessions made the same two mistakes repeatedly:

1. **Fixing a suspected defect from first principles.** A plausible-looking
   mismatch between RELIC and `llama.cpp` was patched without first checking
   whether the behavior was in fact specified (or already known-bad upstream).
   Some of those patches were correct; several were cosmetic fixes to code
   that was not the actual fault, which cost a debugging cycle each.
2. **Trusting an unvalidated control path.** `relic --cpu` was treated as a
   trusted reference while bisecting the OpenCL path. It was not: the CPU
   fallback never performed the token-embedding lookup, so `hidden` was
   never populated and `sample_token` degenerated to a constant. The `--cpu`
   results from that period prove nothing and had to be discarded.

Meanwhile the project had no oracle for intermediate states — only the final
decoded string — so divergence could not be localized to a layer.

## Decision

1. **Every error gets a web search before it gets a patch.** For each build
   error, runtime error, NaN, or numerical divergence, the agent must first
   search for an existing solution: upstream `llama.cpp`/GGML issues and PRs,
   Khronos/OpenCL specification text, GGUF/GGML format documentation, or prior
   discussion of the same symptom. The source is cited in the commit or
   comment, and compared against the current code before any edit. If the
   search finds nothing, that is recorded before implementing. This is written
   down in `AGENTS.md` so it survives the session.
2. **`llama.cpp` is the oracle, not intuition.** Ground truth comes from the
   pinned build in `/tmp/opencode/llama.cpp`:
   - `llama-cli -p ... -n 16 --temp 0 --no-jinja -st --simple-io` for the
     reference decode;
   - `llama-eval-callback -m <model> -p <prompt> -ngl 0` for a per-tensor dump
     (tensor name, `sum`, first/last elements) that lets RELIC's hidden state
     be compared **layer by layer** instead of only at the logits;
   - `ggufmeta.py` for GGUF metadata (the tooling to read v3 typed metadata
     correctly is small enough that guessing at type tags is never justified).
3. **A control path must be validated before it is used as a control.** The
   CPU fallback is only a legitimate bisect reference once it reproduces the
   reference decode for a model that does not use it. Debug-only instrumentation
   (`RELIC_DEBUG_TOKENS`, `RELIC_DEBUG_STATE`) is added behind environment
   variables so validation does not perturb normal runs.
4. **Findings are recorded even when they are exonerations.** RoPE, `Q4_1`,
   `Q4_K`, `Q5_K`, `Q6_K` block layouts, `get_scale_min_k4`, `gemv_q6_k`,
   `build_gdn_l2_norm`, conv wiring and layer-type assignment were each
   audited against `llama.cpp` sources and found to match; the ones that did
   not match are tracked as defects. Recording "this was checked and is fine"
   is what stops the same file being re-audited on the next pass.

## Consequences

* Debugging becomes a two-command loop — dump the reference tensor, print the
  same tensor in RELIC — instead of repeated edits-and-rerun cycles.
* The first use of the protocol immediately produced a concrete result: the
  embedding row for token 760 decodes to `sum = 0.072410` in `llama.cpp` and
  `sum = 0.445659` in RELIC, localizing the Qwen3.5 failure to the `Q6_K`
  row dequantization that feeds both the input embedding and the `lm_head`
  `gemv_q6_k`, and consistent with SmolLM2 (`Q8_0` embedding) decoding
  correctly.
* `relic --cpu` cannot be used as evidence until the missing embedding lookup
  and the incomplete DeltaNet step (no `softplus(ssm_a·(alpha+dt))`, no
  `sigmoid(beta)`, no key L2 norm, no `ssm_norm`/gate) are fixed or the path
  is explicitly documented as non-reference.
* The instrumentation is cheap (an `n_embd` readback per layer, disabled by
  default) and reusable for the next architecture added to the registry.
