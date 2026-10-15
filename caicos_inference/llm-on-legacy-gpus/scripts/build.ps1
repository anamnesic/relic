param(
    [string] $BuildDir = "$PSScriptRoot\..\build",
    [string] $Configuration = "Release",
    [int] $Parallel = 4,
    [switch] $ConfigureOnly
)

if (-not (Test-Path "$PSScriptRoot\..\..\llama.cpp\opencl-headers\CL\opencl.h")) {
    Write-Host "OpenCL headers not found at expected location"
    Write-Host "Set OpenCL headers manually or install OpenCL SDK"
}

$ProjectRoot = Resolve-Path "$PSScriptRoot\.."

# Create build directory
New-Item -ItemType Directory -Path $BuildDir -Force | Out-Null

# Configure
Push-Location $BuildDir
try {
    $CmakeArgs = @(
        "-DCMAKE_BUILD_TYPE=$Configuration"
    )

    # Try to find OpenCL
    if (Test-Path "$PSScriptRoot\..\..\llama.cpp\opencl-headers") {
        $OpenCLHeaderDir = Resolve-Path "$PSScriptRoot\..\..\llama.cpp\opencl-headers"
        $CmakeArgs += "-DOpenCL_INCLUDE_DIR=`"$OpenCLHeaderDir`""
    }

    # Try to find OpenCL library
    $PossibleLibPaths = @(
        "$env:WINDIR\System32\OpenCL.dll",
        "$env:WINDIR\SysWOW64\OpenCL.dll",
        "C:\Program Files (x86)\IntelSWTools\OpenCL\sdk\lib\x64\OpenCL.lib",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\lib\x64\OpenCL.lib"
    )

    foreach ($lib in $PossibleLibPaths) {
        if (Test-Path $lib) {
            $CmakeArgs += "-DOpenCL_LIBRARY=`"$lib`""
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
