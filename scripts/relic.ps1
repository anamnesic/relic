param(
    [Parameter(Position = 0)]
    [ValidateSet("probe", "info", "run", "bench")]
    [string] $Command = "run",

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $RuntimeArgs
)

$ProjectRoot = Resolve-Path "$PSScriptRoot\.."
$BuildCandidates = @(
    (Join-Path $ProjectRoot "build\Release\relic.exe"),
    (Join-Path $ProjectRoot "build\relic.exe"),
    (Join-Path $ProjectRoot "build\Release\caicos_rt.exe"),
    (Join-Path $ProjectRoot "build\caicos_rt.exe")
)
$Runtime = $BuildCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $Runtime) {
    throw "Relic runtime not found. Build first with: pwsh scripts/build.ps1"
}

function Find-DumpModel {
    $candidates = @(
        (Join-Path $ProjectRoot "build\Release\relic_dump_model.exe"),
        (Join-Path $ProjectRoot "build\relic_dump_model.exe")
    )
    return $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
}

switch ($Command) {
    "probe" {
        & $Runtime --list-devices @RuntimeArgs
        exit $LASTEXITCODE
    }
    "info" {
        $dumpModel = Find-DumpModel
        if (-not $dumpModel) {
            throw "relic_dump_model executable not found. Build first with: pwsh scripts/build.ps1"
        }
        & $dumpModel @RuntimeArgs
        exit $LASTEXITCODE
    }
    "run" {
        & $Runtime @RuntimeArgs
        exit $LASTEXITCODE
    }
    "bench" {
        $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
        $output = & $Runtime @RuntimeArgs 2>&1 | Out-String
        $exitCode = $LASTEXITCODE
        $stopwatch.Stop()
        [ordered]@{
            schema_version = "relic.bench.wrapper.v1"
            command = "bench"
            status = if ($exitCode -eq 0) { "ok" } else { "error" }
            executable = $Runtime
            arguments = $RuntimeArgs
            duration_ms = [Math]::Round($stopwatch.Elapsed.TotalMilliseconds, 3)
            exit_code = $exitCode
            output = $output.TrimEnd()
        } | ConvertTo-Json -Depth 4
        exit $exitCode
    }
}
