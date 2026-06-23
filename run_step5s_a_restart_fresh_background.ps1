# STEP 5S-A bs2/accum4 — FRESH RESTART (epoch 1) background launcher.
# Previous checkpoints were corrupted by Windows WDDM TDR during torch.save().
# Runner now uses atomic checkpoint saving (write .tmp then os.replace) to prevent future corruption.
# TDR error is now caught explicitly and logged with resume instructions.
#
# Usage: .\run_step5s_a_restart_fresh_background.ps1
# Monitor: Get-Content -Wait -Tail 40 "<RUN_DIR>/logs/step5s_a_restart_fresh.log"

$ErrorActionPreference = "Stop"

$REPO    = "C:/flood_research/repos/physics-informed-flood-segmentation"
$PYTHON  = "E:/flood_research/venvs/terramind-gpu/Scripts/python.exe"
$RUNNER  = "$REPO/scripts/step5s_a_bs2_accum4_train.py"
$CONFIG  = "$REPO/configs/step5s_a_terramind_l_upernet_corrected_indices_dice_bs2_accum4.yaml"
$RUN_DIR = "E:/flood_research/experiments/terramind_baseline/runs/step5s_a_terramind_l_upernet_corrected_indices_dice_bs2_accum4"
$LOG     = "$RUN_DIR/logs/step5s_a_restart_fresh.log"

Write-Host "=== STEP 5S-A bs2/accum4 FRESH RESTART launcher ==="
Write-Host "Repo:    $REPO"
Write-Host "Python:  $PYTHON"
Write-Host "Runner:  $RUNNER"
Write-Host "Config:  $CONFIG"
Write-Host "Log:     $LOG"
Write-Host "Note:    Checkpoints now written atomically (.tmp -> rename) to survive TDR crashes."

foreach ($path in @($REPO, $PYTHON, $RUNNER, $CONFIG)) {
    if (!(Test-Path $path)) { throw "Required path not found: $path" }
}

New-Item -ItemType Directory -Force -Path "$RUN_DIR/logs" | Out-Null

$argsList = @(
    $RUNNER,
    "--log-file", $LOG,
    "--config", $CONFIG
)

$process = Start-Process `
    -FilePath $PYTHON `
    -ArgumentList $argsList `
    -WorkingDirectory $REPO `
    -WindowStyle Hidden `
    -PassThru

Write-Host ""
Write-Host "STEP 5S-A fresh restart launched in background."
Write-Host "Launcher PID (PowerShell): $PID"
Write-Host "Python process PID:        $($process.Id)"
Write-Host "Log: $LOG"
Write-Host ""
Write-Host "To monitor:"
Write-Host "  Get-Content -Wait -Tail 40 '$LOG'"
