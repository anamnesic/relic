# ADR 0020: In-Kernel SAW-INT4 KV Cache, Native K-Quants, Batched Prefill GEMM, Speculative Decoding Loop, and OpenAI SSE Streaming Server

## Status
Accepted

## Date
2026-09-18

## Context
Following the 15.47 tok/s decode performance established in ADR 0019, further operational scaling required addressing five major capability and throughput frontiers under strict memory constraints:

1. **Long-Context KV Cache Footprint & Bandwidth (SAW-INT4 & FluxBin LUT)**:
   FP16 KV caching reduced cache footprints by 50%, but long context windows (4k–32k tokens) on 4 GB GPUs still risk VRAM exhaustion. In-kernel 4-bit block quantization (Scale-and-Weight INT4) enables a 75% memory reduction, while ultra-low-bit dequantization (Q2/Q3) requires efficient LUT lookup helpers.
2. **Native K-Quant Execution (`Q4_K` and `Q6_K`)**:
   Standard GGUF checkpoints frequently quantize attention down-projections and feed-forward weights using k-quants (`Q4_K` with 144-byte super-blocks, `Q6_K` with 210-byte super-blocks). Previously, models required CPU dequantization or repacking into basic `Q4_0`.
3. **Batched Prefill GEMM (`gemm_q4_0`)**:
   Prompt ingestion previously processed tokens sequentially through GEMV calls. A dedicated 2D-tiled OpenCL matrix multiply ($M \times N \times K$) allows concurrent multi-token prefill computation.
4. **Speculative Decoding Pipeline Integration**:
   An active n-gram draft generator combined with target model batched verification enables multi-token speculation per model step. When device-side sampling (GPU argmax/top-k) is active, verification must query the device directly rather than unpopulated host logit buffers.
5. **Standardized External Ingestion (OpenAI-Compatible SSE Streaming Server)**:
   For seamless drop-in integration with modern web clients, chat UIs, and agentic workflows, the persistent VRAM server daemon required an OpenAI-compatible `/v1/chat/completions` endpoint supporting Server-Sent Events (SSE) streaming and CORS.

## Decision

### 1. In-Kernel SAW-INT4 KV Cache & FluxBin LUT Helpers
- **SAW-INT4 KV Quantization**:
  - Implemented [`kv_cache_append_q4`](file:///home/luann/projects/anamnesic/relic/kernels/kernels.cl#L70) in OpenCL C 1.2: compresses incoming FP32 key and value activations into 4-bit nibbles with a per-block half-precision (`half`) scale factor (18 bytes per 32 values, 75% savings over FP32).
  - Implemented [`qwen_full_attention_step_q4`](file:///home/luann/projects/anamnesic/relic/kernels/kernels.cl#L1120): unrolls 32-element blocks in hardware registers during causal dot-product attention and value accumulation, dequantizing nibbles on-the-fly.
- **FluxBin LUT Helpers**:
  - Added [`fluxbin_lut_q2`](file:///home/luann/projects/anamnesic/relic/kernels/kernels.cl#L125) and [`fluxbin_lut_q3`](file:///home/luann/projects/anamnesic/relic/kernels/kernels.cl#L140) register/local memory lookup helpers for non-linear ultra-low-bit codebook dequantization.

### 2. Native K-Quants GEMV (`gemv_q4_k` and `gemv_q6_k`)
- In [`kernels/kernels.cl`](file:///home/luann/projects/anamnesic/relic/kernels/kernels.cl#L850), implemented native OpenCL kernels for:
  - `gemv_q4_k`: processes 256-weight super-blocks (scales, mins, and 4-bit quantized weights across 8 sub-blocks of 32 values).
  - `gemv_q6_k`: processes 256-weight super-blocks (128-byte low 4-bit nibbles + 64-byte high 2-bit slices + 16 scales).
- Added host-side dequantization routines in [`src/quantization/quant_utils.h`](file:///home/luann/projects/anamnesic/relic/src/quantization/quant_utils.h) and integrated dynamic dispatch in [`Qwen35WeightsManager::dispatch_gemv`](file:///home/luann/projects/anamnesic/relic/src/adapters/qwen35_weights_manager.cpp).

### 3. Batched Prefill GEMM (`gemm_q4_0`)
- Added [`gemm_q4_0`](file:///home/luann/projects/anamnesic/relic/kernels/kernels.cl#L740) in OpenCL C 1.2:
  - Computes $C [M \times N] = A [M \times K] \times B [N \times K]^T$ where weights $B$ are stored in Q4_0 format.
  - Utilizes 2D thread block tiling with shared memory buffering of activation tiles and parallel reduction along the $K$ dimension.
- Exposed `gemm_q4_0` via [`OpenClBackend::gemm_q4_0`](file:///home/luann/projects/anamnesic/relic/src/opencl_backend.cpp).

### 4. Speculative Decoding Pipeline & Device-Sampling Verification
- Integrated active n-gram speculative decoding loop in [`InferenceEngine::generate`](file:///home/luann/projects/anamnesic/relic/src/inference.cpp).
- Solved device-sampling verification synchronization:
  - When `decoder->supports_device_sampling()` is true, device logits are kept in VRAM (bypassing expensive 248k-float host transfers).
  - Verification calls `decoder->sample_token(speculative_sampler)` directly, maintaining hardware acceleration without invalid host memory accesses.
- Added streaming token callback `std::function<void(const std::string &)> on_token` to `InferenceEngine::generate()`.

### 5. OpenAI-Compatible SSE Streaming Daemon (`/v1/chat/completions`)
- In [`src/server.cpp`](file:///home/luann/projects/anamnesic/relic/src/server.cpp):
  - Added HTTP CORS headers (`Access-Control-Allow-Origin: *`, pre-flight `OPTIONS` 204 handler).
  - Added JSON body parsing for `messages` arrays, extracting system and user turns into proper chat templates.
  - Implemented Server-Sent Events (`text/event-stream`) streaming format with chunk deltas (`data: {"choices": [{"delta": {"content": "..."}}]}`) and final sentinel `data: [DONE]`.

## Verification & Results

1. **Speculative Decoding Execution**:
   - Verified end-to-end token generation with `--speculative`:
     Output matches ground truth: `" Paris.\n\n<think>\n\n</think>\n\nYes, that is correct. **Paris** is the capital city of France. It is"` with verified speculative draft acceptance.

2. **Numerical Parity**:
   - `relic_test` verification suite passed 100% across all numerical, adapter, and planner tests:
     - `[layer.norm.rms_norm]` CosSim: `1.000000` -> PASS
     - `[layer.norm.add_rms_norm]` CosSim: `1.000000` -> PASS
     - `[layer.recurrent.deltanet_conv1d]` CosSim: `1.000000` -> PASS
     - `[layer.ffn.swiglu]` CosSim: `1.000000` -> PASS
     - `[layer.output.logits_argmax]` CosSim: `1.000000` -> PASS
     - `[model.layer.00..03.hidden_state]` CosSim: `1.000000` -> PASS
     - Hexagonal Driving and Driven Adapters -> PASS
