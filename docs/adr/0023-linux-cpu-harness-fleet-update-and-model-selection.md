# ADR 0023: Linux CPU Test Harness, Fleet Update and Low-VRAM Model Selection

## Status
Accepted

## Date
2026-10-06

## Context

Three things changed at once:

1. **The Caicos XT target is gone.** The 1 GB Terascale 2 card that motivated
   RELIC's OpenCL 1.2 kernel constraints was retired from the lab. The
   feasibility notes it produced are preserved in
   `docs/hardware/CAICOS-XT.md` as historical constraints, but it can no longer
   be used as a validation device.
2. **The current fleet is two 4 GB NVIDIA cards** — a GeForce GTX 1650 and a
   T1000 — which moves RELIC's design point from "1 GB, no FP16, OpenCL 1.2
   only" to "4 GB, modern NVIDIA, CUDA-capable". Per ADR 0022 those cards are
   reachable through the native CUDA backend, not through a Linux OpenCL ICD.
3. **No accelerator is available on the build/CI host.** It is a 4-vCPU Intel
   Xeon VM with 15 GB RAM, no `/dev/dri`, no NVIDIA driver and no OpenCL ICD
   preinstalled, so `find_package(OpenCL REQUIRED)` fails out of the box.

RELIC still needs a regression gate that runs on every change: the numerical
self-test (`relic_test`), the hardware profiler (`relic --profile`) and the
benchmark suite (`relic --bench`) must be executable somewhere without a GPU.

## Decision

1. **Stand up a CPU harness on Linux using PoCL.** Install
   `ocl-icd-libopencl1`, `ocl-icd-opencl-dev`, `opencl-headers` and
   `pocl-opencl-icd`. PoCL exposes the host CPU as an OpenCL C 1.2 device, which
   is exactly the legacy feature level RELIC was written against, so kernel
   compilation and the numerical verifier exercise the same code paths as real
   legacy hardware. The build is unchanged — CMake already guards Windows-only
   code behind `#ifdef _WIN32` with `sysinfo`/`mmap` fallbacks.
2. **Treat CPU mode as a first-class validation target**, not a fallback.
   `relic --cpu` and the automatic `OpenCL init failed, falling back to CPU`
   path become the CI gate; GPU-specific behavior (VRAM budgeting, DMA overlap,
   CUDA backend) is validated separately on the lab cards.
3. **Record the fleet in-tree** in `docs/hardware/FLEET.md`, so device-specific
   decisions always reference a committed inventory instead of tribal memory.
4. **Pin a reference model for constrained-inference work**: `Qwen3.5-2B` at
   `Q4_0`. Chosen because (a) arch `qwen35` is a native RELIC architecture with
   dedicated adapters, (b) `Q4_0` is in the supported tensor type set, (c) at
   1.16 GB it fits a 4 GB card with room for KV cache while still being a
   quality-competitive 2026 model, and (d) it is small enough for the CPU
   harness to complete a full benchmark pass in reasonable time.

## Consequences

* `relic_test` has one environment-dependent failure on this host:
  `PinnedHostPool allocation (VirtualLock page-locked)` fails because the
  container hard limit for locked memory is 8 MB (`ulimit -Hl` = 8192) and
  cannot be raised without privileges. This is an environmental limit, not a
  numerical defect — page-locked memory is a throughput optimization for the
  PCIe/DMA path, which does not exist on a CPU device. The pool must degrade to
  unlocked memory and report the degraded state rather than failing the suite.
* CI now has a real regression signal (numerical error + CosSim per layer)
  without needing a GPU runner.
* GPU-specific regressions will *not* be caught by this harness; the CUDA
  backend path stays covered only by lab runs on the GTX 1650 / T1000.
* The Caicos-specific constraints (512 MB max alloc, no `cl_khr_fp16`, no SVM)
  remain valid as *worst-case design bounds* even though the device itself is
  gone — they are what forces the OpenCL C 1.2 baseline.

## Alternatives Considered

* **Skip OpenCL entirely and provide a pure-C++ CPU backend.** Rejected: it
  would fork the kernel surface and stop testing the OpenCL path that ships to
  real devices.
* **Validate only on the GTX 1650 / T1000.** Rejected: no GPU is present on the
  host that builds and tests every change, so this would make the gate
  optional.
