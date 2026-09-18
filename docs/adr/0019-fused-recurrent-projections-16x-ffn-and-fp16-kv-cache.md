# ADR 0019: Fused Recurrent Projections, Multi-Row 16x Tiled FFN SwiGLU, and In-Kernel FP16 KV Cache

## Status
Accepted

## Date
2026-09-18

## Context
Following the 16x tiling optimization of `gemv_q4_0` in ADR 0018 (which reached 14.67 tok/s), profiling identified three critical memory bandwidth and instruction latency bottlenecks:

1. **Recurrent Activation Reloads**:
   In recurrent layers (layers 0..23 excluding full attention intervals), `attn_qkv` ($N=6144, K=2048$) and `attn_gate` ($N=2048, K=2048$) multiplied the identical hidden state activation vector `gpu_hidden`. Executing them as two distinct GEMV dispatches required launching two kernel waves and reloading `gpu_hidden` into L1/shared memory twice per recurrent layer (36 dispatches per token).
2. **FFN SwiGLU Sub-Group Underutilization**:
   `gemv_q4_0_ffn_swiglu` was tiled with 64 threads (2 warps) computing 8 rows of Gate and 8 rows of Up per block. For $N=6144$ FFN intermediate dimension, this dispatched 768 blocks, each reloading the input activation into shared memory.
3. **KV Cache VRAM Footprint & Bandwidth Overhead**:
   In full-attention layers, Key and Value tensors were stored as raw 32-bit floats (`sizeof(float)`), requiring large cache allocations and transferring double the necessary memory bandwidth during the causal dot-product attention step.

## Decision

### 1. Fused Recurrent Projections (QKV + Gate GEMV)
- In [`Qwen35WeightsManager`](file:///home/luann/projects/anamnesic/relic/src/adapters/qwen35_weights_manager.cpp), during model upload, row data for `blk.L.attn_qkv.weight` and `blk.L.attn_gate.weight` are concatenated into a single unified tensor `blk.L.attn_qkv_gate.weight` ($N = 8192, K = 2048$) and uploaded directly to GPU VRAM with on-the-fly Q4 repacking.
- Upload of redundant separate tensors is skipped, preserving VRAM residency and memory footprint.
- In [`Qwen35RecurrentBlock`](file:///home/luann/projects/anamnesic/relic/src/adapters/qwen35_recurrent_block.cpp), a single `cl_->gemv_q4_0` call evaluates all 8192 rows simultaneously. The gate slice (indices 6144..8191) is extracted via an in-VRAM DMA copy, completely eliminating 18 kernel launches per token.

### 2. Multi-Row 16x Tiled FFN SwiGLU (`gemv_q4_0_ffn_swiglu`)
- Scaled workgroup execution in [`kernels/kernels.cl`](file:///home/luann/projects/anamnesic/relic/kernels/kernels.cl#L500) to **4 warps (128 threads)** organized into two 64-thread sub-groups (`sub_id = tid / 64`, `sub_tid = tid % 64`).
- Each block computes **16 rows** (8 rows for Gate and 8 rows for Up per sub-group), sharing a single dynamically sized `l_a` buffer (8 KB) and local reduction storage `l_gate[2][8][64]` and `l_up[2][8][64]` (16 KB total shared memory per block).
- Halved the number of dispatched FFN workgroups from 768 down to 384, doubling activation cache reuse and increasing SM warp occupancy.
- Synchronized workgroup sizing across [`OpenClBackend::gemv_q4_0_ffn_swiglu`](file:///home/luann/projects/anamnesic/relic/src/opencl_backend.cpp) and [`IntelUhdBackend::gemv_q4_0_fused_ffn`](file:///home/luann/projects/anamnesic/relic/src/backends/intel_uhd_backend.cpp).

### 3. In-Kernel FP16 KV Quantization & Attention
- Implemented pure OpenCL C 1.2 IEEE-754 bitcast conversions (`fp32_to_fp16` and `fp16_to_fp32`) requiring zero hardware extension dependencies.
- Added [`kv_cache_append_fp16`](file:///home/luann/projects/anamnesic/relic/kernels/kernels.cl#L32) to compress and append newly generated keys and values directly to FP16 caches in VRAM on-the-fly.
- Added [`qwen_full_attention_step_fp16`](file:///home/luann/projects/anamnesic/relic/kernels/kernels.cl#L990) to dequantize FP16 keys and values in hardware registers during causal attention dot-product and value accumulation.
- Halved KV cache buffer sizes from `sizeof(float)` to `sizeof(uint16_t)` in [`Qwen35DecoderAdapter`](file:///home/luann/projects/anamnesic/relic/src/adapters/qwen35_decoder_adapter.cpp), saving 50% KV cache VRAM footprint and reducing memory bandwidth demand.

### 4. Hexagonal Architecture Decoupling
- Domain logic in `InferenceEngine` and port interfaces remain untouched.
- The three sub-block domain adapters (`Qwen35WeightsManager`, `Qwen35RecurrentBlock`, `Qwen35AttentionBlock`) encapsulate their respective micro-architectural enhancements.

## Verification & Results

1. **Throughput Scaling**:
   - Prompt evaluation throughput increased to **19.44 tok/s** (29 tokens in 1492.07 ms).
   - Generation throughput increased from 14.67 tok/s to **15.47 tok/s** (25 tokens in 1615.77 ms).
   - Cumulative decode speedup: **+31.5%** over the initial baseline (11.76 tok/s).
   - 25-token end-to-end execution completed in **1994.18 ms** (sub-2-seconds).

2. **Numerical Parity**:
   - `relic_test` verification suite passed 100%:
     - `[layer.norm.rms_norm]` CosSim: `1.000000` -> PASS
     - `[layer.norm.add_rms_norm]` CosSim: `1.000000` -> PASS
     - `[layer.recurrent.deltanet_conv1d]` CosSim: `1.000000` -> PASS
     - `[layer.ffn.swiglu]` CosSim: `1.000000` -> PASS
     - `[layer.output.logits_argmax]` CosSim: `1.000000` -> PASS
     - `[model.layer.00..03.hidden_state]` CosSim: `1.000000` -> PASS
     - `Driving Adapter (CliOptions Parsing)` -> PASS
     - `Driven Adapter (Backend Interface & BackendBuffer Polymorphism)` -> PASS
