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
   - 4x loop unrolling with 128-bit memory bursts for memory controller saturation on Turing architectures.
3. **In-VRAM Recurrent Gated DeltaNet State & Conv1D (`qwen_conv1d_silu`, `qwen_gated_deltanet_step`)**:
   - Maintain recurrent state $S \in \mathbb{R}^{16 \times 128 \times 128}$ and convolution buffers directly in GPU VRAM buffers (`gpu_ssm_states`, `gpu_conv_states`).
   - Eliminate all 36 blocking CPU-GPU synchronization stalls per token across the 18 recurrent DeltaNet layers.
4. **Fused SwiGLU Activation Kernel (`swiglu_f32`)**:
   - Element-wise fusion of SiLU and multiplication to eliminate redundant VRAM round-trips in FFN layers.
5. **Prompt-Lookup Speculative Decoding (Zero-VRAM Speculation)**:
   - Dynamic $N$-gram pattern matching against context tokens to propose draft candidates without loading secondary draft models or consuming additional VRAM.

## Consequences & Benchmarks
- **GTX 1650 (Turing)**:
  - Generation speed reached **5.35 tok/s sustained** (**~14x speedup / 1.370% gain** vs original 0.39 tok/s).
  - VRAM utilization: **2.60 GB resident** within the 4.0 GB physical VRAM limit.
  - Zero CPU synchronization stalls during DeltaNet recurrence.
- **Intel UHD Graphics**:
  - Prompt speed increased from **0.13 tok/s** (22.7s) to **0.32 tok/s** (9.3s) (**> 2.4x speedup**).
  - Generation speed reached **1.51 tok/s** (**~8x speedup** vs original 0.19 tok/s).
- Zero host-to-device PCIe transfer overhead during token generation.
