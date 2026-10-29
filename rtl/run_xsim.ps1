# run_xsim.ps1 — run the NeuroContinuum-1 RTL testbenches with Vivado's xsim.
#
# This is the Vivado-toolchain equivalent of run_tests.ps1 (which uses the
# lightweight Icarus Verilog).  Same self-checking testbenches, same pass
# tokens (TB_LIF_PASS / TB_CORE_PASS) — running on a second independent
# simulator is what keeps the RTL portable.
#
# Usage, from the repo root or the rtl folder:
#   powershell -ExecutionPolicy Bypass -File rtl\run_xsim.ps1
#
# For the GUI (waveforms, project view) see rtl\create_project.tcl.

param(
    [switch]$CoreOnly
)

$ErrorActionPreference = "Stop"
$rtl = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $rtl

# ---- locate Vivado (PATH first, then the usual install roots) ---------------
function Find-VivadoBin {
    $cmd = Get-Command xvlog.bat -ErrorAction SilentlyContinue
    if ($cmd) { return (Split-Path -Parent $cmd.Source) }
    foreach ($drive in @("C:\", "D:\")) {
        foreach ($vdir in (Get-ChildItem -Path (Join-Path $drive "Xilinx") -Directory -ErrorAction SilentlyContinue)) {
            $bin = Join-Path $vdir.FullName "Vivado\bin"
            if (Test-Path (Join-Path $bin "xvlog.bat")) { return $bin }
        }
    }
    return $null
}
$V = Find-VivadoBin
if (-not $V) {
    Write-Host "ERROR: Vivado not found. Install Vivado (AMD/Xilinx) or use rtl\run_tests.ps1 (Icarus)." -ForegroundColor Red
    exit 2
}
Write-Host "Using Vivado tools in: $V"

New-Item -ItemType Directory -Force -Path work | Out-Null

$SRC = @("lif_neuron.v", "bcm_plasticity.v", "stochastic_neuron.v",
         "first_spike_decoder.v", "fault_monitor.v", "synapse_remap.v", "neuro_core.v")

$failed = 0

# ---- compile once ------------------------------------------------------------
& "$V\xvlog.bat" -sv @SRC "tb\tb_lif_neuron.v" "tb\tb_neuro_core.v" | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "COMPILE FAILED (xvlog)" -ForegroundColor Red
    exit 1
}

function Run-TB([string]$top, [string]$snap, [string]$passTag) {
    & "$V\xelab.bat" -top $top -s $snap | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ELABORATION FAILED: $top" -ForegroundColor Red
        $script:failed++
        return
    }
    $out = & "$V\xsim.bat" $snap -R 2>&1 | Tee-Object -Variable log
    $ok = ($log -join "`n") -match $passTag
    if ($LASTEXITCODE -eq 0 -and $ok) {
        Write-Host "  -> $passTag found" -ForegroundColor Green
    } else {
        $out | Select-Object -Last 15
        Write-Host "  -> $passTag MISSING" -ForegroundColor Red
        $script:failed++
    }
}

if (-not $CoreOnly) { Run-TB "tb_lif_neuron" "lift_snap" "TB_LIF_PASS" }
Run-TB "tb_neuro_core" "core_snap" "TB_CORE_PASS"

if ($failed -eq 0) {
    Write-Host "ALL RTL TESTS PASSED (Vivado xsim)" -ForegroundColor Green
    exit 0
} else {
    Write-Host "$failed RTL TEST(S) FAILED (Vivado xsim)" -ForegroundColor Red
    exit 1
}
