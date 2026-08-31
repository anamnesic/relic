# 3. Architecture Blueprint: Constrained Heterogeneous LLM Runtime

Date: 2026-08-30

## Status
Accepted

## Context
As LLM models continue to scale, consumer-grade and workstation laptops (e.g. Acer Nitro 5, ThinkPads, Dell G-Series) featuring 4 GB dedicated GPUs (NVIDIA GeForce GTX 1650, GTX 1050 Ti, Quadro T1000) face the "4GB VRAM Wall", where models exceeding 4 GB (such as 7B and 8B models requiring 4.8 - 6.0 GB in quantized formats) fail to run on standard CUDA/Metal runtimes without catastrophic performance degradation.

Simultaneously, standard runtimes treat the Integrated GPU (e.g. Intel UHD Graphics 11th Gen with 16 Execution Units and up to 6.4 GB dynamically shared system memory via OpenCL 3.0) as invisible, leaving massive graphics compute and shared memory bandwidth unutilized.

RELIC transitions from a legacy retro-hardware experiment to a specialized, production-grade **Constrained Heterogeneous LLM Runtime**.

## Decision
Adopt the following modular 4-pillar architectural blueprint:

```text
RELIC
│
├── Execution Engine
│   ├── NVIDIA Backend (CUDA / OpenCL 1.2+ Warp-32 Coalesced)
│   ├── Intel GPU Backend (OpenCL 3.0 UHD/Iris Shared Memory)
│   └── CPU Backend (AVX2 / AVX-512 Fallback)
│
├── Adaptive Memory Engine          ★ Core Differentiator
│   ├── Tensor Profiler (Criticality and access-frequency weighting)
│   ├── Tiered Placement (Tier 0 dGPU ➔ Tier 1 iGPU Shared ➔ Tier 2 Host RAM)
│   ├── Pinned Host RAM (Zero-copy direct DMA transfers)
│   ├── Async Transfers & Prefetching (PCIe latency hiding via pipeline)
│   └── Dynamic Migration (Demand paging and cold tensor eviction)
│
├── Adaptive Context Engine
│   ├── Quantized KV Cache (FP16 ➔ Q8_0 ➔ Q4_0 ➔ Q2_K)
│   ├── KV Eviction & Sliding Window Attention
│   ├── Context Budgeting
│   └── Pressure-Aware Memory Eviction Policy
│
└── Speculative Engine (Heterogeneous)
    ├── Target Engine ➔ NVIDIA GTX 1650 / Quadro T1000 (Fast verification)
    └── Draft Engine  ➔ CPU AVX2 / Intel UHD (Background draft generation)
```

### Memory Tier Hierarchy:
1. **Tier 0 (Dedicated GDDR VRAM - 4 GB)**:
   - Bandwidth: 128 GB/s.
   - Hosts hot parameters (Attention QKV, initial recurrent layers, dense MLP projections).
2. **Tier 1 (Integrated GPU Shared Memory - up to 6.4 GB)**:
   - Bandwidth: 25 GB/s.
   - Hosts contiguous overflow layer blocks with zero-copy shared host pointers.
3. **Tier 2 (System Host RAM - 16 GB)**:
   - Bandwidth: 25 - 50 GB/s.
   - Hosts cold residual layers, vocabulary embeddings, and draft model states.
4. **Tier 3 (Memory-Mapped Storage - NVMe SSD)**:
   - Bandwidth: 3 - 7 GB/s.
   - On-demand streaming for ultra-large models (14B+).

## Consequences
- Enables execution of models that exceed physical dGPU VRAM (e.g. 7B/8B Q4 models) on standard 4 GB laptops without crashing or degrading to single-threaded CPU speeds.
- Preserves ultra-low latency on models that fit in VRAM (Qwen 3.5 2B sustaining **23.89 - 24.48 tok/s**).
- Maximizes hardware utilization across 100% of the host machine's compute silicon concurrently.
