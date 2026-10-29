# run_all.ps1 — one-pass verification of the whole NeuroContinuum-1 project.
#
# Windows equivalent of the top-level `make` (no make required).  Runs, in
# order, and stops at the first failure:
#
#   1. package smoke test        (imports + a short end-to-end run)
#   2. four novel blocks         (self-checking assertions)
#   3. continual demo  --n 400   (learning ON, real regime change)
#   4. continual demo  --n 400   (frozen  -> the A/B baseline)
#   5. architecture + energy benchmark vs CPU/GPU
#   6. RTL simulation            (both testbenches; use -Wave for a VCD)
#
# Usage, from the repo root:
#   powershell -ExecutionPolicy Bypass -File run_all.ps1
#   powershell -ExecutionPolicy Bypass -File run_all.ps1 -Wave

param(
    [switch]$Wave
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$steps = @(
    @{ name = "smoke test";    cmd = "python"; args = @("-m", "simulator.examples._smoke") },
    @{ name = "novel blocks";  cmd = "python"; args = @("-m", "simulator.examples.test_novel_features") },
    @{ name = "demo (learn)";  cmd = "python"; args = @("-m", "simulator.examples.demo_continual", "--n", "400") },
    @{ name = "demo (frozen)"; cmd = "python"; args = @("-m", "simulator.examples.demo_continual", "--n", "400", "--no-learn") },
    @{ name = "benchmark";     cmd = "python"; args = @("-m", "architecture.benchmark") }
)

$failed = 0
$sw = [System.Diagnostics.Stopwatch]::StartNew()

foreach ($s in $steps) {
    Write-Host ""
    Write-Host ("=" * 70)
    Write-Host (">>> [{0}]  {1} {2}" -f $s.name, $s.cmd, ($s.args -join " ")) -ForegroundColor Cyan
    Write-Host ("=" * 70)
    & $s.cmd @($s.args)
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $($s.name) (exit $LASTEXITCODE)" -ForegroundColor Red
        $failed++
        break
    }
}

if ($failed -eq 0) {
    Write-Host ""
    Write-Host ("=" * 70)
    Write-Host ">>> [rtl]  rtl\run_tests.ps1" -ForegroundColor Cyan
    Write-Host ("=" * 70)
    $rtlArgs = @("-ExecutionPolicy", "Bypass", "-File",
                 (Join-Path $root "rtl\run_tests.ps1"))
    if ($Wave) { $rtlArgs += "-Wave" }   # -File passes strings; don't send "False"
    & powershell @rtlArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: rtl tests (exit $LASTEXITCODE)" -ForegroundColor Red
        $failed++
    }
}

$sw.Stop()
Write-Host ""
if ($failed -eq 0) {
    Write-Host ("ALL STEPS PASSED in {0:n1}s  -> reports in architecture\results\ and rtl\work\" `
        -f $sw.Elapsed.TotalSeconds) -ForegroundColor Green
    exit 0
} else {
    Write-Host ("{0} STEP(S) FAILED after {1:n1}s" -f $failed, $sw.Elapsed.TotalSeconds) -ForegroundColor Red
    exit 1
}
