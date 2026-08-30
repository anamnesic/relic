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
2. **Fused Quantized GEMV OpenCL 1.2 Kernels (`gemv_q8_0`, `gemv_q4_0`)**:
   - Direct dequantization inside GPU work-item private registers during matrix-vector dot product calculation.
   - Workgroup parallel reduction using high-speed on-chip local memory (`__local float l_sum[128]`).
   - Multi-Row 4x Register Coarsening: Each workgroup computes 4 matrix rows simultaneously, loading input activation slices into private registers once and cutting global/L1 activation cache reads by 4x.
   - Warp-32 coalesced execution with 128-bit memory bursts for memory controller saturation on Turing architectures.
3. **In-VRAM Recurrent Gated DeltaNet State & Conv1D (`qwen_conv1d_silu`, `qwen_gated_deltanet_step`)**:
   - Maintain recurrent state $S \in \mathbb{R}^{16 \times 128 \times 128}$ and convolution buffers directly in GPU VRAM buffers (`gpu_ssm_states`, `gpu_conv_states`).
   - Eliminate all 36 blocking CPU-GPU synchronization stalls per token across the 18 recurrent DeltaNet layers.
4. **Pure GPU Causal Full Attention & In-VRAM KV Cache (`qwen_full_attention_step`)**:
   - Execute RoPE and Causal Multi-Head Attention directly on GPU for the 6 Full Attention layers, keeping the KV cache in VRAM.
5. **Fused SwiGLU Activation Kernel (`swiglu_f32`)**:
   - Element-wise fusion of SiLU and multiplication to eliminate redundant VRAM round-trips in FFN layers.
6. **Prompt-Lookup Speculative Decoding (Zero-VRAM Speculation)**:
   - Dynamic $N$-gram pattern matching against context tokens to propose draft candidates without loading secondary draft models or consuming additional VRAM.

## Consequences & Benchmarks
- **GTX 1650 (Turing)**:
  - Generation speed surged from **0.39 tok/s** $\to$ **5.45 tok/s** $\to$ **9.39 tok/s** $\to$ **12.98 tok/s sustained** (**~33.3x speedup / 3.228% gain** vs initial baseline).
  - Generation latency dropped to **~77 ms per token**.
  - Prompt processing throughput reached **9.91 tok/s**.
  - VRAM utilization: **2.60 GB resident** within the 4.0 GB physical VRAM limit.
  - Zero host-to-device PCIe transfer overhead during token generation.
- **Intel UHD Graphics**:
  - Prompt speed increased from **0.13 tok/s** (22.7s) to **0.44 tok/s** (6.7s) (**> 3.4x speedup**).
  - Generation speed increased from **0.19 tok/s** $\to$ **1.51 tok/s** $\to$ **2.25 tok/s** (**~11.8x speedup**).
