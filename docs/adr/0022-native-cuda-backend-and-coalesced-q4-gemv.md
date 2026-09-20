# ADR 0022: Native CUDA Backend and Coalesced Q4 GEMV

## Status
Accepted

## Date
2026-09-19

## Context

The GTX 1650 is visible to WSL through NVIDIA's CUDA bridge, but it is not
enumerated by the Linux OpenCL ICD. The installed native Linux NVIDIA OpenCL
library does not bridge to the WSL virtual GPU. Consequently, the OpenCL Relic
binary can only see the PoCL CPU device even though CUDA identifies the physical
GTX 1650 (compute capability 7.5, 4 GB VRAM).

Running a Windows OpenCL binary through WSL interoperability remains possible,
but it adds a second compiler environment and does not provide a native Linux
deployment. Relic therefore requires CUDA as the NVIDIA backend on WSL while
retaining OpenCL for Intel, AMD, and native OpenCL installations.

## Decision

Relic now builds an optional native `CudaBackend` whenever CMake detects
`nvcc`. The backend implements:

- CUDA stream-backed allocation, upload, download, and device copies;
- Q4_0 and Q8_0 matrix-vector products;
- fused Q4_0 gate/up/SwiGLU;
- RMSNorm and fused residual-add/RMSNorm;
- RoPE, Q4_0 embedding lookup, and device argmax;
- CUDA device discovery through `relic --list-devices`;
- a reproducible `relic_cuda_bench` microbenchmark with CPU numerical checks.

The Q4_0 kernel maps one output row to a 16-thread half-warp. Each lane loads
one packed Q4 byte and the matching two FP32 activations. This makes both input
streams contiguous and uses width-16 shuffle reduction without shared memory or
block-wide barriers.

The implementation was compared with the mature `ggml-cuda` approach already
present in `references/llama.cpp`. An experimental Q8 activation plus `dp4a`
variant was measured and rejected: on this Turing mobile GPU it achieved only
25–31 GB/s and added quantization error up to 1.07, while the FP32 activation
kernel was both faster and numerically tighter. No additional project or binary
dependency was downloaded.

## Measured Results

Hardware and software:

- NVIDIA GeForce GTX 1650, compute capability 7.5, 4 GB;
- Windows driver 610.62 via WSL CUDA;
- CUDA Toolkit 12.4 / nvcc 12.4.131;
- release build for `sm_75`;
- 500 warmup iterations and 2,000 measured iterations.

| Q4_0 shape (N x K) | Mean latency | Effective bandwidth | Max abs error |
|---|---:|---:|---:|
| 8192 x 2048 | 0.190 ms | 49.83 GB/s | 0.000092 |
| 6144 x 2048 | 0.143 ms | 49.74 GB/s | 0.000092 |
| 2048 x 6144 | 0.151 ms | 47.09 GB/s | 0.000145 |

The original four-row-per-warp CUDA prototype measured 19.86, 22.07, and
21.39 GB/s for the same respective shapes. The coalesced kernel is therefore
2.20x to 2.50x faster in effective bandwidth. Five additional 1,000-iteration
runs reproduced approximately 47–50 GB/s after the GPU left its idle power
state.

These are real kernel measurements on the target GTX, not an estimated token
rate. End-to-end tok/s is not claimed until the Qwen and Llama architecture
adapters are fully switched from direct OpenCL handles to the backend-neutral
buffer interface.

The existing OpenCL path was also exercised on the same physical GTX through
PoCL's CUDA device. The full `relic_test` numerical and architecture suite
passed, and the Qwen 3.5 2B end-to-end benchmark measured 16.797 tok/s median
decode. That end-to-end number validates ADR 0021's projection fusion; it does
not yet measure the native CUDA backend introduced here.

## Consequences

CUDA is now a first-class compiled backend and its critical primitive can be
measured directly on the target GPU from WSL. The remaining integration work is
to remove direct `cl_mem`/`cl_command_queue` use from the architecture adapters
so the complete model graph can select CUDA at runtime. Until then, CUDA
microbenchmarks are valid but the existing end-to-end inference command remains
on the OpenCL/CPU paths.
