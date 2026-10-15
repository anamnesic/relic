# Caicos XT OpenCL Experiment

This is the local experiment path for the AMD Caicos XT on Windows.

## Machine result

`clinfo` detects the GPU, but the device does not expose the features required by the current `llama.cpp` OpenCL backend:

- device name: `Caicos`
- global memory: 1024 MB
- max single allocation: 512 MB
- device OpenCL C: `OpenCL C 1.2`
- device OpenCL version: `OpenCL 1.2 AMD-APP (1800.11)`
- `cl_khr_fp16`: not present
- SVM: not present

The current OpenCL backend is built around Adreno/Intel-class assumptions and requires OpenCL C 2.0+ plus FP16 kernels. Caicos has neither. A host build can link, but GPU-only LLM execution with this backend is blocked before model load.

## Build-only path

Windows has `OpenCL.dll`, but this machine does not have `OpenCL.lib` installed with the SDK. The repo script generates an import library from the system DLL and then builds the OpenCL target:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build-caicos-opencl.ps1
```

For a faster compile-only sanity check, build just the OpenCL library:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build-caicos-opencl.ps1 -Target ggml-opencl
```

To only configure and verify the build files without compiling:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build-caicos-opencl.ps1 -ConfigureOnly
```

The script configures:

- `GGML_OPENCL=ON`
- `GGML_OPENCL_USE_ADRENO_KERNELS=OFF`
- `GGML_OPENCL_TARGET_VERSION=300`
- local Khronos headers from `llama.cpp/opencl-headers`
- generated import lib at `llama.cpp/build-caicos-ninja/opencl-import/OpenCL.lib`
- Ninja generator with MSVC environment imported from Visual Studio Build Tools
- future-dated CMake metadata is normalized before configure; otherwise Ninja can keep triggering `cmake --regenerate-during-build` on this checkout

`GGML_OPENCL_TARGET_VERSION=300` is used so the host source can compile against the Khronos headers and Windows OpenCL loader. It does not upgrade the Caicos device. Runtime kernel compilation still sees the device as OpenCL C 1.2.

## Local test result

These commands were verified on this machine:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build-caicos-opencl.ps1 -Target ggml-opencl -Parallel 4
powershell -ExecutionPolicy Bypass -File .\scripts\build-caicos-opencl.ps1 -Target llama-cli -Parallel 4
```

Both targets build. The resulting executable is:

```text
llama.cpp\build-caicos-ninja\bin\llama-cli.exe
```

Runtime device enumeration still fails before a model is loaded:

```powershell
$env:GGML_OPENCL_PLATFORM='0'
$env:GGML_OPENCL_DEVICE='0'
.\llama.cpp\build-caicos-ninja\bin\llama-cli.exe --list-devices
```

Observed result:

```text
Available devices:
ggml_opencl: OpenCL C 2.0 or above is required; device reports 1.2
ggml_opencl: OpenCL C 2.0 or above is required; device reports 1.2
```

Only vocabulary/test GGUF files are present locally under `llama.cpp\models`, so there is no local real model to run after device enumeration.

## Practical conclusion

The realistic result for this hardware is:

- OpenCL platform and GPU discovery: yes.
- `ggml-opencl` host library compilation: yes.
- `llama-cli` executable build: yes.
- `llama-cli --list-devices` with OpenCL: no usable device; the backend rejects OpenCL C 1.2.
- `llama.cpp` GPU-only inference on Caicos XT: no, not with the current backend.

Making this work would require a separate OpenCL 1.2, no-FP16 path: f32 or integer-only kernels, no subgroup dependency, and memory planning around the 512 MB max allocation limit. That is a backend port, not a CMake flag.
