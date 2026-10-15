param(
    [string] $DllPath = "$env:WINDIR\System32\OpenCL.dll",
    [string] $OutputDir = "$PSScriptRoot\..\llama.cpp\build-caicos-vs2\opencl-import"
)

$ErrorActionPreference = "Stop"

function Find-Tool {
    param(
        [string] $Name
    )

    $fromPath = Get-Command $Name -ErrorAction SilentlyContinue
    if ($fromPath) {
        return $fromPath.Source
    }

    $vsRoot = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC"
    $tool = Get-ChildItem $vsRoot -Recurse -Filter $Name -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -like "*\bin\Hostx64\x64\*" } |
        Select-Object -First 1

    if (!$tool) {
        throw "Could not find $Name. Install Visual Studio Build Tools with MSVC."
    }

    return $tool.FullName
}

if (!(Test-Path $DllPath)) {
    throw "OpenCL DLL not found: $DllPath"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$dumpbin = Find-Tool "dumpbin.exe"
$lib = Find-Tool "lib.exe"
$defPath = Join-Path $OutputDir "OpenCL.def"
$libPath = Join-Path $OutputDir "OpenCL.lib"

$exports = & $dumpbin /EXPORTS $DllPath |
    ForEach-Object {
        if ($_ -match "^\s+\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]+\s+(cl[A-Za-z0-9_]+)\s*$") {
            $Matches[1]
        }
    } |
    Sort-Object -Unique

if (!$exports -or $exports.Count -eq 0) {
    throw "Could not parse OpenCL exports from $DllPath"
}

$def = @("LIBRARY OpenCL.dll", "EXPORTS") + ($exports | ForEach-Object { "    $_" })
Set-Content -Path $defPath -Value $def -Encoding ASCII

& $lib /def:$defPath /machine:x64 /out:$libPath | Out-Host
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Output $libPath
