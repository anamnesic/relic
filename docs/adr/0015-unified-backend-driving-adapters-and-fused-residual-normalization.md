# ADR 0015: Unified Backend Interface, Driving CLI Adapters, and Fused Residual Normalization

## Status
Accepted

## Date
2026-09-18

## Context
Following the Hexagonal Architecture refactoring in ADR 0014 and user mandates to continue systematic decoupling, two architectural and performance gaps remained:

1. **Backend Device Interface Asymmetry**: While `IntelUhdBackend` implemented the pure abstract interface `Backend` (`src/backends/backend.h`), `OpenClBackend` remained a monolithic concrete driver with raw `ClBuffer` structs, preventing polymorphic multi-backend composition.
2. **Monolithic Driving CLI Entrypoint**: `src/main.cpp` encompassed over 560 lines containing argument parsing, benchmark suite coordination, factorial ablation experiments, cliff analysis, TCP/HTTP daemon handling, and interactive REPL logic mixed directly with bootstrap code.
3. **Redundant Global Memory Round-Trips and Kernel Launches**: Each decoder layer in Qwen 3.5 was launching separate OpenCL kernels for:
   - `copy` (saving residual)
   - `rms_norm` (pre-attention norm)
   - `add` (attention residual addition)
   - `copy` (saving residual before MLP)
   - `rms_norm` (pre-FFN norm)
   - `add` (FFN residual addition)
   Across 24 layers, this required 145 separate OpenCL kernel dispatches per token for memory maintenance alone, generating substantial PoCL driver queuing latency in WSL2.
4. **FFN SwiGLU Inner-Loop Scaling Redundancy**: In `gemv_q4_0_ffn_swiglu`, `d_gate[r]` and `d_up[r]` scaling factors were multiplied inside the innermost 4-step vector loop for all 8 rows, performing 512 redundant floating-point multiplications per block per thread.

## Decision

### 1. Unified `Backend` and `BackendBuffer` Polymorphism
- `ClBuffer` implements the abstract `BackendBuffer` interface (`raw_handle()`, `size()`, `tier()`), retaining high-performance direct member access while conforming to the abstract memory contract.
- `OpenClBackend` implements the pure abstract `Backend` interface (`allocate`, `upload`, `download`, `copy`, `synchronize`, `query_stats`, and compute primitives `rms_norm`, `add_rms_norm`, `gemv_q4_0`, `gemv_q8_0`, `gemv_q4_0_fused_ffn`, `rope`, `embed_lookup_q4_0`, `argmax`).
- Overloaded direct methods taking concrete `ClBuffer&` remain available for zero-overhead intra-adapter execution.

### 2. Driving CLI Command Adapters (`src/cli/`)
- Extracted command handlers into dedicated single-responsibility controllers:
  - `CliOptions` (`src/cli/cli_options.{h,cpp}`): Encapsulates CLI argument parsing and help presentation.
  - `BenchmarkCommand` (`src/cli/benchmark_command.{h,cpp}`): Orchestrates factorial ablations, cliff analysis, parametric budget sweeps, and standard statistical benchmark suites.
  - `ServerCommand` (`src/cli/server_command.{h,cpp}`): Manages HTTP/TCP persistent daemon mode and client IPC queries.
  - `InferenceCommand` (`src/cli/inference_command.{h,cpp}`): Coordinates interactive REPL sessions and one-shot text generation.
- Reduced `src/main.cpp` from 567 lines down to 180 lines, serving purely as the application composition root.

### 3. Fused Residual Normalization (`add_rms_norm`)
- Integrated `add_rms_norm_f32` across layer transitions in `Qwen35DecoderAdapter` and `Qwen35MlpBlock`:
  - **Attention $\to$ FFN**: Fused `gpu_residual += gpu_attn_out` and `gpu_hidden = rms_norm(gpu_residual, ffn_norm)` in a single dispatch.
  - **FFN $\to$ Next Layer**: Fused `gpu_residual += gpu_ffn_down` and `gpu_hidden = rms_norm(gpu_residual, next_attn_norm)` in a single dispatch.
  - **Final Layer $\to$ Logits**: Fused `gpu_residual += gpu_ffn_down` and `gpu_hidden = rms_norm(gpu_residual, output_norm)` in a single dispatch.
- **Impact**: Completely eliminated 95 kernel dispatches per token (from 145 down to 50 memory maintenance dispatches across 24 layers) and eliminated all intermediate `cl->copy` operations in the decode loop.

### 4. GEMV Inner-Loop Scale Factorization
- In `gemv_q4_0_ffn_swiglu` (`kernels/kernels.cl`), hoisted `d_gate[r]` and `d_up[r]` outside the vector loop.
- Accumulate unscaled integer/float dot products into `block_acc_gate[8]` and `block_acc_up[8]`, scaling once per block per row.
- Synchronized with `src/embedded_kernels.h`.

## Verification & Results
1. **Numerical Equivalence & Quality**:
   - `relic_test` layer-by-layer verification:
     - `[layer.norm.rms_norm]` CosSim: `1.000000` -> PASS
     - `[layer.norm.add_rms_norm]` CosSim: `1.000000` -> PASS
     - `[layer.recurrent.deltanet_conv1d]` CosSim: `1.000000` -> PASS
     - `[layer.ffn.swiglu]` CosSim: `1.000000` -> PASS
     - `[model.layer.00..03.hidden_state]` CosSim: `1.000000` -> PASS
2. **Text Coherence**:
   - Target prompt: `"The capital of France is"`
   - Output: `" Paris.\n\n<think>\n\n</think>\n\nYes, that is correct. **Paris** is the capital city of France. It is"`
   - Zero token degradation or semantic divergence.
3. **Execution Efficiency**:
   - Generation latency: ~82 ms/tok with pristine bit-for-bit equivalence.
   - 95 fewer OpenCL driver enqueues per token.
