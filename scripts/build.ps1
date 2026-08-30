param(
    [string] $BuildDir = "$PSScriptRoot\..\build",
    [string] $Configuration = "Release",
    [int] $Parallel = 4,
    [switch] $ConfigureOnly
)

if (-not (Test-Path "$PSScriptRoot\..\third_party\OpenCL-Headers\CL\opencl.h") -and -not (Test-Path "$PSScriptRoot\..\..\llama.cpp\opencl-headers\CL\opencl.h")) {
    Write-Host "OpenCL headers not found at expected location"
    Write-Host "Set OpenCL headers manually or install OpenCL SDK"
}

$ProjectRoot = Resolve-Path "$PSScriptRoot\.."

# Load MSVC include/lib paths when this script is run outside a Developer Shell.
if (-not (Get-Command cl -ErrorAction SilentlyContinue) -or -not $env:INCLUDE) {
    $vcvars = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    if (Test-Path $vcvars) {
        cmd /c "`"$vcvars`" >nul && set" | ForEach-Object {
            if ($_ -match "^([^=]+)=(.*)$") {
                Set-Item -Path "Env:$($matches[1])" -Value $matches[2]
            }
        }
    }
}

if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    $cmakeDir = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"
    if (Test-Path (Join-Path $cmakeDir "cmake.exe")) {
        $env:Path = "$cmakeDir;$env:Path"
    }
}

# Create build directory
New-Item -ItemType Directory -Path $BuildDir -Force | Out-Null

# Windows graphics drivers ship OpenCL.dll but commonly omit OpenCL.lib. Build
# a local import library from the driver exports when no SDK import library is
# installed, so MSVC can link the runtime without a vendor SDK.
$OpenCLImportLib = Join-Path $BuildDir "OpenCL.lib"
if (-not (Test-Path $OpenCLImportLib) -and (Test-Path "$env:WINDIR\System32\OpenCL.dll")) {
    $OpenCLDef = Join-Path $BuildDir "OpenCL.def"
    $exports = & dumpbin /exports "$env:WINDIR\System32\OpenCL.dll" | ForEach-Object {
        if ($_ -match "^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]+\s+(\S+)$") { $matches[1] }
    }
    if ($exports.Count -gt 0) {
        @("LIBRARY OpenCL.dll", "EXPORTS") + $exports | Set-Content -Encoding ascii $OpenCLDef
        & lib "/def:$OpenCLDef" "/machine:x64" "/out:$OpenCLImportLib" | Out-Null
    }
}

# Configure
Push-Location $BuildDir
try {
    $CmakeArgs = @(
        "-DCMAKE_BUILD_TYPE=$Configuration"
    )

    # Prefer vendored official Khronos headers, then the historical llama.cpp copy.
    $HeaderCandidates = @(
        "$PSScriptRoot\..\third_party\OpenCL-Headers",
        "$PSScriptRoot\..\..\llama.cpp\opencl-headers"
    )
    foreach ($headers in $HeaderCandidates) {
        if (Test-Path "$headers\CL\opencl.h") {
            $OpenCLHeaderDir = Resolve-Path $headers
            $CmakeArgs += "-DOpenCL_INCLUDE_DIR=$OpenCLHeaderDir"
            break
        }
    }

    # Try to find OpenCL library
    $PossibleLibPaths = @(
        $OpenCLImportLib,
        "$env:WINDIR\System32\OpenCL.dll",
        "$env:WINDIR\SysWOW64\OpenCL.dll",
        "C:\Program Files (x86)\IntelSWTools\OpenCL\sdk\lib\x64\OpenCL.lib",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\lib\x64\OpenCL.lib"
    )

    foreach ($lib in $PossibleLibPaths) {
        if (Test-Path $lib) {
            $CmakeArgs += "-DOpenCL_LIBRARY=$lib"
            break
        }
    }

    cmake $ProjectRoot @CmakeArgs

    if (-not $ConfigureOnly) {
        cmake --build . --config $Configuration -j $Parallel
    }
}
finally {
    Pop-Location
}
