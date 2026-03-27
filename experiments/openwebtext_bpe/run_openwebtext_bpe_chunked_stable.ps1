param(
    [string]$InputPath = "data/openwebtext_sampled_2_5GB.txt",
    [int]$VocabSize = 32000,
    [int]$ChunkSizeMb = 128,
    [int]$ReportEveryChunks = 4,
    [int]$ReportEveryMerges = 1000,
    [string]$OutputDir = "outputs/openwebtext_bpe_chunked",
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Resolve-AbsolutePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PathValue,
        [Parameter(Mandatory = $true)]
        [string]$BaseDir,
        [switch]$MustExist
    )

    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        $resolved = [System.IO.Path]::GetFullPath($PathValue)
    }
    else {
        $resolved = [System.IO.Path]::GetFullPath((Join-Path $BaseDir $PathValue))
    }

    if ($MustExist -and -not (Test-Path $resolved)) {
        throw "Path does not exist: $resolved"
    }

    return $resolved
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptDir "..\.."))
$pythonExe = Resolve-AbsolutePath -PathValue ".venv\Scripts\python.exe" -BaseDir $repoRoot -MustExist
$trainerScript = Resolve-AbsolutePath -PathValue "experiments\openwebtext_bpe\run_openwebtext_bpe_chunked.py" -BaseDir $repoRoot -MustExist
$inputAbs = Resolve-AbsolutePath -PathValue $InputPath -BaseDir $repoRoot -MustExist
$outputAbs = Resolve-AbsolutePath -PathValue $OutputDir -BaseDir $repoRoot

New-Item -ItemType Directory -Force -Path $outputAbs | Out-Null

$writeProbePath = Join-Path $outputAbs ".__write_probe__"
Set-Content -Path $writeProbePath -Value "ok" -Encoding UTF8
Remove-Item $writeProbePath -Force

$null = & $pythonExe -c "import sys; from cs336_basics.tokenizer_optimized import build_word_freq_from_text; print(sys.version)"
if ($LASTEXITCODE -ne 0) {
    throw "Python environment validation failed."
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $outputAbs ("run_{0}.full.log" -f $timestamp)
$manifestPath = Join-Path $outputAbs ("run_{0}.manifest.json" -f $timestamp)
$statusPath = Join-Path $outputAbs ("run_{0}.status.json" -f $timestamp)

$inputItem = Get-Item $inputAbs
$commandArgs = @(
    $trainerScript,
    "--input-path", $inputAbs,
    "--vocab-size", $VocabSize,
    "--chunk-size-mb", $ChunkSizeMb,
    "--report-every-chunks", $ReportEveryChunks,
    "--report-every-merges", $ReportEveryMerges,
    "--output-dir", $outputAbs
)

$manifest = [ordered]@{
    timestamp = $timestamp
    repo_root = $repoRoot
    python_executable = $pythonExe
    trainer_script = $trainerScript
    input_path = $inputAbs
    input_size_bytes = $inputItem.Length
    output_dir = $outputAbs
    vocab_size = $VocabSize
    chunk_size_mb = $ChunkSizeMb
    report_every_chunks = $ReportEveryChunks
    report_every_merges = $ReportEveryMerges
    command = @($pythonExe) + $commandArgs
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -Path $manifestPath -Encoding UTF8

Write-Host "Repo root: $repoRoot"
Write-Host "Python: $pythonExe"
Write-Host "Input: $inputAbs"
Write-Host ("Input size: {0:N0} bytes" -f $inputItem.Length)
Write-Host "Output dir: $outputAbs"
Write-Host "Manifest: $manifestPath"
Write-Host "Log: $logPath"

if ($ValidateOnly) {
    $status = [ordered]@{
        status = "validated"
        timestamp = (Get-Date).ToString("s")
        manifest_path = $manifestPath
        input_path = $inputAbs
        output_dir = $outputAbs
    }
    $status | ConvertTo-Json -Depth 5 | Set-Content -Path $statusPath -Encoding UTF8
    Write-Host "Validation complete. No training was started."
    exit 0
}

$startTime = Get-Date
try {
    & $pythonExe @commandArgs 2>&1 | Tee-Object -FilePath $logPath
    $exitCode = $LASTEXITCODE
}
catch {
    $exitCode = 1
    $_ | Out-String | Tee-Object -FilePath $logPath -Append | Out-Null
}

$status = [ordered]@{
    status = if ($exitCode -eq 0) { "completed" } else { "failed" }
    exit_code = $exitCode
    started_at = $startTime.ToString("s")
    finished_at = (Get-Date).ToString("s")
    manifest_path = $manifestPath
    log_path = $logPath
    expected_outputs = @(
        (Join-Path $outputAbs "owt_chunked_vocab.pkl"),
        (Join-Path $outputAbs "owt_chunked_merges.pkl"),
        (Join-Path $outputAbs "owt_chunked_bpe_summary.json")
    )
}
$status | ConvertTo-Json -Depth 5 | Set-Content -Path $statusPath -Encoding UTF8

if ($exitCode -ne 0) {
    throw "Chunked BPE training failed. See log: $logPath"
}

Write-Host "Training completed successfully."
Write-Host "Status: $statusPath"
