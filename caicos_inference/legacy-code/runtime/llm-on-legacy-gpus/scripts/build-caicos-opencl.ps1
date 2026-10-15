param(
    [string] $BuildDir = "$PSScriptRoot\..\llama.cpp\build-caicos-ninja",
    [string] $Target = "llama-cli",
    [string] $Configuration = "Release",
    [int] $Parallel = 4,
    [switch] $ConfigureOnly
)

$ErrorActionPreference = "Stop"

function Find-VsDevCmd {
    $known = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
    if (Test-Path $known) {
        return $known
    }

    $found = Get-ChildItem "C:\Program Files*\Microsoft Visual Studio" -Recurse -Filter VsDevCmd.bat -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if (!$found) {
        throw "Could not find VsDevCmd.bat. Install Visual Studio Build Tools with the MSVC toolchain."
    }
    return $found.FullName
}

function Import-VsDevEnvironment {
    $vsDevCmd = Find-VsDevCmd
    $envLines = cmd /s /c "`"$vsDevCmd`" -arch=x64 -host_arch=x64 >nul && set"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to initialize the Visual Studio build environment."
    }

    foreach ($line in $envLines) {
        $idx = $line.IndexOf("=")
        if ($idx -le 0) {
            continue
        }
        $name = $line.Substring(0, $idx)
        $value = $line.Substring($idx + 1)
        Set-Item -Path "Env:$name" -Value $value
    }
}

function Normalize-FutureCMakeTimestamps {
    param(
        [string] $SourceDir
    )

    $now = Get-Date
    $items = Get-ChildItem $SourceDir -Recurse -Include CMakeLists.txt,*.cmake,CMakePresets.json -ErrorAction SilentlyContinue |
        Where-Object {
            $_.FullName -notlike "*\build-caicos-*\" -and
            $_.FullName -notlike "*\build-caicos-*\\*" -and
            $_.LastWriteTime -gt $now
        }

    foreach ($item in $items) {
        $item.LastWriteTime = $now
    }
}

$repoRoot = Resolve-Path "$PSScriptRoot\.."
$llamaDir = Join-Path $repoRoot "llama.cpp"
$headersDir = Join-Path $llamaDir "opencl-headers"
$openclLib = & "$PSScriptRoot\make-opencl-import-lib.ps1" -OutputDir (Join-Path $BuildDir "opencl-import")

if (!(Get-Command ninja -ErrorAction SilentlyContinue)) {
    throw "ninja was not found on PATH. Install Ninja or make it available before running this script."
}

Import-VsDevEnvironment
Normalize-FutureCMakeTimestamps -SourceDir $llamaDir

cmake -S $llamaDir -B $BuildDir `
    -G Ninja `
    -DCMAKE_BUILD_TYPE="$Configuration" `
    -DCMAKE_SUPPRESS_REGENERATION=ON `
    -DGGML_OPENCL=ON `
    -DGGML_OPENCL_TARGET_VERSION=300 `
    -DGGML_OPENCL_USE_ADRENO_KERNELS=OFF `
    -DGGML_OPENCL_EMBED_KERNELS=ON `
    -DGGML_CUDA=OFF `
    -DGGML_VULKAN=OFF `
    -DGGML_HIP=OFF `
    -DGGML_METAL=OFF `
    -DGGML_NATIVE=OFF `
    -DGGML_BUILD_TESTS=OFF `
    -DGGML_BUILD_EXAMPLES=OFF `
    -DGGML_LLAMAFILE=ON `
    -DBUILD_SHARED_LIBS=OFF `
    -DOpenCL_INCLUDE_DIR="$headersDir" `
    -DOpenCL_LIBRARY="$openclLib"

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if ($ConfigureOnly) {
    exit 0
}

cmake --build $BuildDir --target $Target --parallel $Parallel
exit $LASTEXITCODE
