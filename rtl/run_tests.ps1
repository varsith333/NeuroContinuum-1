# run_tests.ps1 — compile + run the NeuroContinuum-1 RTL testbenches (Windows)
#
# This is the Windows-friendly equivalent of `make` in this folder.  It needs
# Icarus Verilog (iverilog + vvp).  Usage, from the repo root or this folder:
#
#   powershell -ExecutionPolicy Bypass -File rtl\run_tests.ps1           # both TBs
#   powershell -ExecutionPolicy Bypass -File rtl\run_tests.ps1 -Wave     # + wave.vcd
#
# Prints TB_LIF_PASS / TB_CORE_PASS on success and exits non-zero on failure.

param(
    [switch]$Wave,
    [switch]$CoreOnly
)

$ErrorActionPreference = "Stop"
$rtl = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $rtl

# ---- locate iverilog/vvp (PATH first, then the common winget/C: installs) ---
function Find-Tool([string]$name) {
    $cmd = Get-Command $name -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($dir in @("C:\iverilog\bin", "$env:LOCALAPPDATA\Programs\iverilog\bin",
                       "C:\Program Files\iverilog\bin")) {
        $p = Join-Path $dir "$name.exe"
        if (Test-Path $p) { return $p }
    }
    return $null
}
$IVERILOG = Find-Tool "iverilog"
$VVP      = Find-Tool "vvp"
if (-not $IVERILOG -or -not $VVP) {
    Write-Host "ERROR: iverilog/vvp not found. Install with:  winget install Icarus.Verilog" -ForegroundColor Red
    exit 2
}
Write-Host "Using: $IVERILOG"

New-Item -ItemType Directory -Force -Path work | Out-Null

$SRC = @("lif_neuron.v", "bcm_plasticity.v", "stochastic_neuron.v",
         "first_spike_decoder.v", "fault_monitor.v", "synapse_remap.v", "neuro_core.v")

$failed = 0

function Run-TB([string]$tb, [string]$out, [string]$passTag, [switch]$dump) {
    $args = @("-g2012") + $(if ($dump) { @("-DDUMP") } else { @() }) + @("-o", $out, $tb) + $script:SRC
    & $script:IVERILOG @args
    if ($LASTEXITCODE -ne 0) {
        Write-Host "COMPILE FAILED: $tb" -ForegroundColor Red
        $script:failed++
        return
    }
    & $script:VVP $out | Tee-Object -Variable log
    $ok = ($log -join "`n") -match $passTag
    if ($LASTEXITCODE -eq 0 -and $ok) {
        Write-Host "  -> $passTag found" -ForegroundColor Green
    } else {
        Write-Host "  -> $passTag MISSING" -ForegroundColor Red
        $script:failed++
    }
}

if (-not $CoreOnly) { Run-TB "tb/tb_lif_neuron.v" "work\lif.vvp"  "TB_LIF_PASS" }
Run-TB "tb/tb_neuro_core.v" "work\core.vvp" "TB_CORE_PASS" -dump:$Wave

if ($Wave) { Write-Host "Waveform written to $rtl\wave.vcd (open with GTKWave)" }

if ($failed -eq 0) {
    Write-Host "ALL RTL TESTS PASSED" -ForegroundColor Green
    exit 0
} else {
    Write-Host "$failed RTL TEST(S) FAILED" -ForegroundColor Red
    exit 1
}
