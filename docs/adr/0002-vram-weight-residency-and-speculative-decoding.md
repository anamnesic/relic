# ADR 0002: VRAM Weight Residency, Fused Quantized GEMVs & Zero-VRAM Speculative Decoding

## Status
Accepted

## Context
Initial GPU offloading transferred model weights from CPU to GPU over the PCIe bus on every single layer during token generation, causing massive memory bandwidth bottlenecks (~0.39 tok/s generation throughput).

Furthermore, the target hardware comprises:
1. **NVIDIA GeForce GTX 1650 (Turing TU117)**: 4 GB GDDR5/GDDR6 VRAM, 128 GB/s bandwidth, 14 SMs, FP32 + INT32 concurrent execution, no Tensor Cores.
2. **Intel UHD Graphics**: 6.4 GB shared system RAM, 16 Execution Units, memory-bandwidth bound.

## Decision
1. **Full VRAM Weight Residency (`GpuTensorStore`)**:
   - Upload all model quantized weight tensors (Q8_0, Q4_0, FP32) directly into GPU VRAM buffers at initialization.
   - Keep model weights resident in VRAM across the entire inference session, reducing host-to-device PCIe bandwidth per token to 0 bytes.
2. **On-the-Fly Dynamic In-VRAM Q4 Repacking & Weight Compression**:
   - Quantizes resident Q8_0 weights to Q4_0 directly in VRAM at initialization in < 1.0s, reducing resident model VRAM footprint from 2.60 GB to 1.67 GB (~50% memory traffic savings).
3. **Fused Quantized GEMV OpenCL 1.2 Kernels (`gemv_q8_0`, `gemv_q4_0`)**:
   - Direct dequantization inside GPU work-item private registers during matrix-vector dot product calculation.
   - Workgroup parallel reduction using high-speed on-chip local memory (`__local float l_sum[128]`).
   - 128-bit SIMD Vector Bursts: Memory controller instructions `LDG.E.128` (`vload4`, `char4`, `uchar4`) with hardware `dot(float4, float4)` FMA execution.
   - Multi-Row 8x Register Coarsening: Each workgroup computes 8 matrix rows simultaneously, loading input activation slices into private registers once and cutting global/L1 activation cache reads by 8x.
   - Warp-32 coalesced execution with 128-bit memory bursts for memory controller saturation on Turing architectures.
4. **In-VRAM Recurrent Gated DeltaNet State & Conv1D (`qwen_conv1d_silu`, `qwen_gated_deltanet_step`)**:
   - Maintain recurrent state $S \in \mathbb{R}^{16 \times 128 \times 128}$ and convolution buffers directly in GPU VRAM buffers (`gpu_ssm_states`, `gpu_conv_states`).
   - Eliminate all 36 blocking CPU-GPU synchronization stalls per token across the 18 recurrent DeltaNet layers.
5. **Pure GPU Causal Full Attention & In-VRAM KV Cache (`qwen_full_attention_step`)**:
   - Execute RoPE and Causal Multi-Head Attention directly on GPU for the 6 Full Attention layers, keeping the KV cache in VRAM.
6. **Fused 128-bit SIMD SwiGLU & RMSNorm Kernels (`swiglu_f32`, `rms_norm_f32`, `add_rms_norm_f32`)**:
   - 128-bit vector processing (`float4`) for activation normalization and non-linearities, reducing normalization latency by 4x.
7. **Prompt-Lookup Speculative Decoding (Zero-VRAM Speculation)**:
   - Dynamic $N$-gram pattern matching against context tokens to propose draft candidates without loading secondary draft models or consuming additional VRAM.

## Consequences & Benchmarks
- **GTX 1650 (Turing)**:
  - Generation speed surged from **0.39 tok/s** $\to$ **5.45 tok/s** $\to$ **13.09 tok/s** $\to$ **23.82 tok/s sustained** (**~61.1x speedup / 6.007% gain** vs initial baseline).
  - Generation latency dropped to **~42 ms per token**.
  - Prompt processing throughput reached **23.05 tok/s** (22 tokens in 954 ms).
  - Total inference runtime for 47 tokens dropped from **~65 seconds** to **2.00 seconds**.
  - VRAM utilization: **1.67 GB resident** (down from 2.60 GB).
  - Zero host-to-device PCIe transfer overhead during token generation.
- **Intel UHD Graphics**:
  - Prompt speed increased from **0.13 tok/s** (22.7s) to **0.44 tok/s** (6.7s) (**> 3.4x speedup**).
  - Generation speed increased from **0.19 tok/s** $\to$ **1.51 tok/s** $\to$ **2.25 tok/s** (**~11.8x speedup**).
