param(
  [string]$BuildDir = "build-caicos",
  [string]$Generator = "Visual Studio 17 2022",
  [string]$Architecture = "x64"
)

$ErrorActionPreference = "Stop"

$sdkRoot = "C:\Program Files (x86)\Windows Kits\10"
$includeRoot = Join-Path $sdkRoot "Include"
$libRoot = Join-Path $sdkRoot "Lib"

$sdkVersion = Get-ChildItem $includeRoot -Directory |
  Where-Object { Test-Path (Join-Path $_.FullName "um\OpenCL.h") } |
  Sort-Object Name -Descending |
  Select-Object -First 1

if (-not $sdkVersion) {
  throw "OpenCL headers were not found under $includeRoot. Install the Windows SDK OpenCL development headers before building."
}

$openClLib = Join-Path $libRoot (Join-Path $sdkVersion.Name "um\$Architecture\OpenCL.lib")
if (-not (Test-Path $openClLib)) {
  throw "OpenCL import library was not found at $openClLib. Install the Windows SDK OpenCL development libraries before building."
}

cmake -S . -B $BuildDir -G $Generator -A $Architecture `
  -DBUILD_SHARED_LIBS=OFF `
  -DGGML_NATIVE=OFF `
  -DGGML_OPENCL=ON `
  -DGGML_OPENCL_USE_ADRENO_KERNELS=OFF

cmake --build $BuildDir --config Release