# Caicos XT — Target Hardware & Feasibility Notes

> Baseline profiling and feasibility study for running modern transformers on
> obsolete hardware via OpenCL 1.2, custom GGUF runtimes and heavily quantized
> models. Formerly the `surveyor` repository, absorbed into RELIC.

---

## 🎯 Target Hardware & Baseline Profiling

| Field | Value |
|---|---|
| GPU Line | AMD Radeon HD 7000 series / R5 OEM |
| PCI ID | `VEN_1002&DEV_6778` |
| Silicon Chip | Caicos XT / Terascale 2 Architecture |
| Common Names | Radeon HD 7470 / 8470, R5 235 / 310 (OEM) |
| Driver Version | Crimson 15.201.1151.1008 (Legacy WHQL) |
| Dedicated VRAM | 1024 MB (1 GB) DDR3 |
| Max Allocatable Buffer | 512 MB per OpenCL context |
| OpenCL Profile | OpenCL 1.2 Full Profile (No SVM, No subgroups) |

Runtime-detected profiles for currently supported devices live in
[`devices.json`](../../devices.json) (`relic --profile`).

---

## 🔬 Technical Directions & Feasibility Notes

Because legacy chips like Caicos XT lack modern compute pathways (ROCm or
OpenCL 2.0+), mainstream LLM engines fail out-of-the-box:

1. **Standard `llama.cpp` Failure Analysis:**
   - Attempting to compile and run vanilla `llama.cpp` with standard OpenCL
     backends on Caicos XT triggers:
     ```text
     OpenCL C 2.0 or above is required; device reports 1.2
     ```
   - In addition, missing FP16 extensions (`cl_khr_fp16`) prevent standard
     half-precision tensor execution without emulation or software
     decompression.

2. **Workaround Pathways:**
   - Hand-rolled OpenCL 1.2 kernels with FP32/int8 quantization unpack.
   - Pinned host memory pooling with DMA transfers to respect the 512 MB max
     buffer limitation.
   - Custom GGUF runtime with static activation buffer reuse — implemented in
     RELIC itself.

---

## 📚 Related Documents

- [2026 Research Theses & Applied Roadmap](../RESEARCH_THESES_2026.md)
- [ADR 0010 — Performance-Preserving Working Set Discovery](../adr/0010-performance-preserving-working-set-discovery.md)
