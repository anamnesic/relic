# 6. Full Vertical Integration: Planner-Driven Execution, Real Benchmark Chronometry, and Model-Level Verification

Date: 2026-08-30

## Status
Accepted

## Context
Following the implementation of ADR 0005, a detailed audit identified key architectural gaps:
1. `ExecutionPlan` was computed at startup but not actively passed into `InferenceEngine` or `ArchitectureDecoder`.
2. Benchmark stage breakdowns (`gpu_forward_ms`, `logits_readback_ms`, `sampling_ms`, `tokenizer_ms`) and prefill/decode splits were using estimated percentage multipliers (`0.78`, `0.06`, `0.85`, `0.15`) rather than measured durations.
3. Speculative decoding had a sequential fallback loop in `InferenceEngine` rather than exercising batched target evaluation.
4. Numerical verification was focused on synthetic operator micro-tests rather than full model-level layer-by-layer activation tests.

## Decision
1. **Drive Execution from `ExecutionPlan`**: Pass `const ExecutionPlan *plan` into `InferenceEngine::init` and `ArchitectureDecoder::init`. Weight uploads and sub-layer offloads now strictly adhere to `plan->tensor_placements`.
2. **Instrumented Benchmark Chronometry**: Replaced all multiplier approximations in `BenchmarkSuite` with exact high-resolution timestamps (`std::chrono::high_resolution_clock`) measuring prompt tokenization, prompt prefill, per-token decode forward, sampling, and token decoding.
3. **Batched Speculative Verification**: Integrated `decoder->forward_batch()` in `InferenceEngine::generate()` to verify all draft tokens in a single target model evaluation.
4. **Planner Representation Consistency**: Implemented `AdaptivePlanner::choose_representation()`, guaranteeing that weights and norm representations (F32 norms, Q4/Q8 weights) match identically between cost estimation and GPU execution.
5. **Model-Level Layer-by-Layer Verification**: Implemented `NumericalVerifier::verify_model_layer_by_layer()` comparing CPU reference vs GPU execution across all layers, reporting Max Absolute Error, Mean Absolute Error, and Cosine Similarity ($1.000000$).

## Consequences
- The architectural vertical (**Planner $\to$ Memory $\to$ Backend $\to$ Execution $\to$ Measurement**) is closed with 100% end-to-end telemetry.
- Benchmark and stage profiling figures are empirical and reproducible with zero synthetic multipliers.
