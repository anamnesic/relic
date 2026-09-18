# 14. Strict Hexagonal Architecture Refactoring, Persistent VRAM Daemon, and Low-Latency OpenCL JIT Precompilation

Date: 2026-09-18

## Status
Accepted

## Context
Following the stabilization of the RELIC engine and memory cliff analysis, three key challenges emerged:
1. **Architectural Coupling (SRP Violation)**: `src/decoder.cpp` had grown into a monolithic ~1,500-line file entangling core domain orchestration, CPU reference mathematical routines, LLaMA multi-head attention, and Qwen3.5 hybrid DeltaNet SSM / full-attention operations.
2. **Cold-Start & Reload Penalty**: Every inference invocation re-parsed GGUF tensors, re-ran the Adaptive Planner, and transferred 1.68 GB of model weights across the PCIe bus into GPU VRAM (~10–15 s latency per query).
3. **Prompt JIT Overhead & Kernel Scheduling**: OpenCL kernel JIT recompilations and small-kernel launch latencies caused excessive prompt evaluation delay (~14.7 s for 5 prompt tokens).

## Decision

### 1. Strict Hexagonal Architecture (Ports and Adapters)
We restructured the engine following the Ports and Adapters architectural pattern:
- **Core Domain & Services**:
  - `InferenceEngine` (`src/inference.{h,cpp}`): Orchestrates tokenization, execution plans, prompt processing, and token sampling loop.
  - `ArchitectureSpec`, `LlamaModel`, `AdaptivePlanner`, `HardwareProfile`: High-level domain entities completely independent of hardware APIs.
- **Driving (Primary) Ports & Adapters**:
  - `src/main.cpp`: Standalone CLI driver for one-shot runs, benchmarking, and diagnostics.
  - `src/server.{h,cpp}`: Persistent HTTP socket server and client CLI driver for zero-reload daemon operation.
  - `src/test_opencl.cpp`: Automated verification test suite driving the domain and hardware adapters.
- **Pure Math Domain**:
  - `src/cpu/cpu_ops.h`: Reusable, header-only reference CPU kernels (`matmul_nt_cpu`, `rms_norm_cpu`, `rope_cpu`, `add_cpu`, `silu_cpu`) under namespace `relic::cpu`.
- **Driven (Secondary) Ports & Adapters**:
  - `ArchitectureDecoder` (`src/decoder.h`): Abstract decoder port defining the contract for neural architectures (`init`, `reset`, `forward`, `warm_up`, `sample_token`, `save_state_checkpoint`, `restore_state_checkpoint`).
  - `LlamaDecoderAdapter` (`src/adapters/llama_decoder_adapter.{h,cpp}`): Concrete driven adapter for standard LLaMA attention architectures.
  - `Qwen35DecoderAdapter` (`src/adapters/qwen35_decoder_adapter.{h,cpp}`): Concrete driven adapter for Qwen 3.5 hybrid DeltaNet SSM + full-attention architectures.
  - `src/decoder.cpp`: Pure IoC factory delegating instantiation based on `ArchitectureKind`.
  - `OpenClBackend`, `IntelUhdBackend`: Hardware compute adapters implementing device memory management and GPU dispatches.

### 2. Persistent VRAM Daemon Mode (`--server` / `--client`)
- Added daemon mode (`./build/relic -m <model> --server <port>`) maintaining model weights and execution plans resident in GPU VRAM across requests.
- Integrated standard endpoints:
  - `POST /generate`: Accepts JSON payload `{"prompt": "...", "n_tokens": 50}` and streams back generated tokens.
  - `GET /health`: Returns residency status and engine health.
  - `POST /reset`: Resets KV-caches and recurrent state buffers without reallocating VRAM tensors.
- Added client interface (`./build/relic --client <port> -p "..."`) and standard HTTP `curl` support.

### 3. OpenCL JIT Precompilation & Activation Cache Optimizations
- Implemented monolithic precompilation of all OpenCL kernels at engine startup with `-cl-fast-relaxed-math` and `-cl-mad-enable`, caching binaries in `OpenClBackend::dev.program` to eliminate repeated JIT delays during prompt evaluation.
- Optimized `gemv_q4_0_ffn_swiglu` kernel with local activation caching (`l_a[6144]`) and 2-warp occupancy doubling, cutting decode time.
- Corrected DeltaNet recurrent state transitions and full-attention rotary embeddings, eliminating token degradation and achieving exact mathematical parity.

## Empirical Validation

### 1. Test Suite & Numerical Verification
```text
=== Relic OpenCL 1.2 Self-Test ===
Platform: Portable Computing Language
  Device 0: NVIDIA GeForce GTX 1650 (GPU, sm_75, 4095 MB VRAM)
PASS: OpenCL initialized
  rms_norm_f32, matmul_f32, matmul_f32_nt, rope_f32, silu_f32: OK
  matmul result: PASS
  rms_norm result: PASS
  Qwen3.5 recurrent state: PASS
  Qwen3.5 decoder port & forward pass: PASS
  HardwareProfile probe: PASS (2 devices detected)
  AdaptivePlanner plan: PASS (PCIe Traffic: 100.0% red, DMA Overlap: 100.0%)
  AsyncPrefetcher Double DMA Overlap: PASS (Eff: 186.5%)
  [layer.norm.rms_norm]             CosSim: 1.000000 -> PASS
  [layer.norm.add_rms_norm]         CosSim: 1.000000 -> PASS
  [layer.recurrent.deltanet_conv1d] CosSim: 1.000000 -> PASS
  [layer.ffn.swiglu]                CosSim: 1.000000 -> PASS
  [model.layer.00-03.hidden_state]  CosSim: 1.000000 -> PASS
=== All tests PASSED ===
```

### 2. End-to-End Coherence & Latency Benchmark
- Model: `qwen3.5-2b.gguf` (2.2B parameters, Q8_0 weights with on-the-fly Q4 repacking).
- Hardware: NVIDIA GeForce GTX 1650 Mobile (4 GB GDDR5, 128 GB/s).
- Prompt: `"The capital of France is"`
- Result:
  - Prompt evaluation: 5 tokens in 420 ms (down from 14,700 ms cold JIT overhead).
  - Generation: 25 tokens in 2402 ms (~10.4 tok/s).
  - Coherence output: `" Paris.\n<think>\n</think>\nYes, that is correct. **Paris** is the capital city of France. It is"`.
- VRAM Residency Daemon: Subsequent queries via `--client` execute with 0 ms weight reload time.

## Consequences
- The codebase satisfies clean Hexagonal Architecture guidelines: adapters are isolated, domain logic has zero dependency on low-level OpenCL calls, and factory instantiation is decoupled.
- Users can run long-running persistent daemon services for interactive applications without repeated cold-start penalties.
