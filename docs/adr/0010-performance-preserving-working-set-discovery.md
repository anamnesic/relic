# 10. Performance-Preserving Working Set Discovery (MPVB) and Factorial Ablation

Date: 2026-08-31

## Status
Accepted

## Context
Following the initial over-budget proof in ADR 0009, experimental rigor required:
1. Multi-run statistical confidence (mean, median, p50, p95, stddev) to separate signal from clock/thermal jitter.
2. Disambiguating planned weight budgets from actual physical OpenCL/GPU allocations (activations, SSM states, Conv1D, KV cache, DMA staging).
3. Explicit breakdown across physical tiers: Dedicated GPU VRAM, Intel UHD shared memory, and Host Pinned DMA streaming.
4. Correcting the autoregressive speculative verification alignment ($D_0$ verified by pre-draft logits, $D_{i+1}$ by draft $i$ forward).
5. Defining and discovering the **Minimum Performance-Preserving VRAM Budget (MPVB)**.

## Decision
1. **Speculative Autoregressive Alignment**: Corrected the verification pipeline in `InferenceEngine::generate()`, ensuring draft $D_0$ is evaluated against pre-draft logits and draft $D_{i+1}$ against `batched_logits[i]`.
2. **Tier-Disaggregated Accounting**: `AdaptivePlanner` now separates `gtx_vram_weights_bytes`, `intel_uhd_weights_bytes`, and `pinned_streamed_weights_bytes`, while projecting `actual_gtx_allocation_peak_bytes` including KV caches, SSM state matrices, and staging buffers.
3. **Statistical Multi-Run Sweep**: Built `--sweep` and `--ablation` modes into `relic.exe`, evaluating 7 fine-grained budget increments (`3500`, `2000`, `1750`, `1500`, `1250`, `1000`, `750` MB) over multi-run samples reporting p50, p95, and stddev.
4. **Automated MPVB Discovery**: Programmatically discover the Minimum Performance-Preserving VRAM Budget:
   $$\text{MPVB}_{95\%} = \min \left\{ B \in \mathcal{B} \;\middle|\; \frac{\text{Throughput}(B)}{\text{Throughput}(\text{Baseline})} \ge 0.95 \right\}$$

## Empirical Results (Real 2.74 GB Qwen Model)
- **Baseline Speed (3500 MB Budget)**: $25.54\text{ tok/s}$ ($2039.8\text{ MB}$ peak allocation)
- **1750 MB Budget**: $25.56\text{ tok/s}$ ($100.1\%$ throughput)
- **1500 MB Budget**: $25.39\text{ tok/s}$ ($99.4\%$ throughput, stddev: $0.12$)
- **Memory Cliff**: Observed between $1500\text{ MB}$ ($25.39\text{ tok/s}$) and $1250\text{ MB}$ ($8.02\text{ tok/s}$).
- **$\text{MPVB}_{\ge 95\%}$**: Exactly **$1500\text{ MB}$**, achieving $57.1\%$ VRAM capacity reduction and $21.9\%$ weight residency reduction with zero statistically meaningful degradation.

## Consequences
- The heterogeneous runtime thesis is backed by a defensible metric ($\text{MPVB}$) and empirical multi-run statistical distributions.
