# ADR 0001: Hexagonal Architecture for Multi-Model Decoders & OpenCL Acceleration

## Status
Accepted

## Context
The inference engine required support for multiple model families (Llama transformer architectures and Qwen 3.5 hybrid Gated DeltaNet / linear attention architectures) across heterogeneous hardware backends (NVIDIA GeForce GTX 1650 dedicated GPU, Intel UHD integrated GPU, and CPU reference fallback).

Coupling model architectures directly to hardware kernel dispatch created rigid dependencies and duplicated execution code.

## Decision
1. **Hexagonal Ports & Adapters Architecture**:
   - Defined `ArchitectureDecoder` as the core domain port (`init`, `reset`, `forward`).
   - Implemented `LlamaDecoderAdapter` and `Qwen35DecoderAdapter` as architecture adapters.
   - Built a factory `create_decoder(spec, backend)` to decouple architectural execution from engine runtime.
2. **OpenCL 1.2 Hardware Abstraction**:
   - Implemented `OpenClBackend` using pure OpenCL C 1.2 without proprietary vendor extensions.
   - Decoupled platform and device selection (supporting multi-platform CUDA and Intel OpenCL runtimes).

## Consequences
- **Positive**: Clean separation of concerns; adding new architectures (Mamba, RWKV, DeepSeek) requires only implementing a new adapter.
- **Positive**: Both CPU reference mode and GPU hardware acceleration share the exact same inference engine workflow.
- **Trade-offs**: Small virtual function call overhead per forward pass (negligible compared to matrix multiplications).
