# 4. Architecture Foundation: Dynamic Tensor Scheduling & 2026 Literature Grounding

Date: 2026-08-30

## Status
Accepted

## Context
Following the implementation of Phase 1, Phase 2, and Phase 3 of RELIC (incorporating OpenCL 1.2 multi-row GEMV, fused FFN, in-VRAM embedding lookup/argmax, pinned host memory prefetching, and distributed heterogeneous speculation), a comprehensive survey of 2026 state-of-the-art research was conducted.

The literature confirms that LLM inference under tight memory constraints (1 GB - 4 GB) on consumer laptops is fundamentally data-movement bound rather than arithmetic compute bound. Recent 2026 breakthroughs—specifically **ATSInfer** (arXiv:2607.10183), **Llamas on the Web** (Microsoft Research), **Gated DeltaNet Optimization** (arXiv:2607.16831), **Fast NF4** (arXiv:2604.02556), **Cache-Resident LLM Inference** (arXiv:2606.25353), **SAW-INT4** (arXiv:2604.19157), and **FluxBin** (arXiv:2608.15602)—provide critical empirical foundations for RELIC's evolution.

## Decision
Adopt the 2026 Research Foundation Roadmap as formalized in docs/RESEARCH_THESES_2026.md with the following architectural commitments:

1. **Sub-Layer Tensor Granular Scheduling (ATSInfer)**:
   - Transition from coarse layer-by-layer offloading to fine-grained per-tensor placement driven by the enefit_per_byte metric:
     \text{benefit\_per\_byte} = \frac{\text{Latency}_{\text{CPU}} - \text{Latency}_{\text{GPU}}}{\text{Tensor Size (bytes)}}
   - Prioritize VRAM residency for tensors providing maximum latency reduction per byte of allocated GDDR.

2. **Static Memory Planning & Kernel Registry (Llamas on the Web)**:
   - Eliminate runtime dynamic driver allocations by pre-computing a deterministic activation buffer reuse graph during model initialization.
   - Establish a DeviceProfile and KernelRegistry that micro-benchmarks kernel variants during system probe (
elic --profile) to select optimal workgroup layouts, subgroup shuffles, or tiled local-memory paths.

3. **Mega-Operator Fusion for Recurrent States (Gated DeltaNet MLSys 2026)**:
   - Merge the state update, normalization, gating, and output projection of hybrid linear attention architectures (Qwen 3.5) into a single OpenCL dispatch, minimizing global memory round-trips.

4. **Residency-as-a-Principle Execution (Cache-Resident LLM)**:
   - Formulate memory access overhead as a first-class optimization objective in the execution planner, ensuring zero intermediate host transfers during decode iterations.

5. **Integrated In-Kernel KV Quantization & Ultra-Low-Bit Support (SAW-INT4 & FluxBin)**:
   - Fuse rotation and quantization directly into attention kernels under memory pressure.
   - Introduce experimental LUT-based ultra-low-bit kernels (Q2/Q3) targeting 1 GB to 2 GB VRAM legacy devices.

## Consequences
- Elevates RELIC beyond a simple OpenCL compatibility port into an adaptive, profile-driven heterogeneous runtime that automatically derives optimal execution schedules for any arbitrary host configuration.
- Establishes a formal scientific basis and benchmark validation checklist for Phase 4 milestones.\n