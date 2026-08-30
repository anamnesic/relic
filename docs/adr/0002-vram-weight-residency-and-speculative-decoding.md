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
3. **Fused SwiGLU Activation Kernel (`swiglu_f32`)**:
   - Element-wise fusion of SiLU and multiplication to eliminate redundant VRAM round-trips in FFN layers.
4. **Prompt-Lookup Speculative Decoding (Zero-VRAM Speculation)**:
   - Dynamic $N$-gram pattern matching against context tokens to propose draft candidates without loading secondary draft models or consuming additional VRAM.

## Consequences & Benchmarks
- **GTX 1650 (Turing)**:
  - Generation speed improved from **0.39 tok/s** to **4.97 tok/s sustained** (**~13x speedup / 1.275% gain**).
  - VRAM utilization: **2.60 GB resident** within the 4.0 GB physical VRAM limit.
- **Intel UHD Graphics**:
  - Generation speed improved from **0.19 tok/s** to **1.32 tok/s** (**~7x speedup / 694% gain**).
- Zero host-to-device PCIe transfer overhead during token generation.
