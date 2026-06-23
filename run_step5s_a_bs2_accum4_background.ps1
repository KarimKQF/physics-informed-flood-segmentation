# STEP 5S-A bs2/accum4 background launcher.
# TerraMind-L + UPerNet corrected indices [5,11,17,23], Dice, FP32.
# batch_size=2, gradient_accumulation_steps=4, effective_batch=8.
# No physics loss, no DEM input, no DARN, no STURM-Flood.
#
# Usage: .\run_step5s_a_bs2_accum4_background.ps1
# Monitor log: Get-Content -Wait -Tail 40 "E:/flood_research/experiments/terramind_baseline/runs/step5s_a_terramind_l_upernet_corrected_indices_dice_bs2_accum4/logs/step5s_a_bs2_accum4_training.log"

$ErrorActionPreference = "Stop"

$REPO    = "C:/flood_research/repos/physics-informed-flood-segmentation"
$PYTHON  = "E:/flood_research/venvs/terramind-gpu/Scripts/python.exe"
$RUNNER  = "$REPO/scripts/step5s_a_bs2_accum4_train.py"
$CONFIG  = "$REPO/configs/step5s_a_terramind_l_upernet_corrected_indices_dice_bs2_accum4.yaml"
$RUN_DIR = "E:/flood_research/experiments/terramind_baseline/runs/step5s_a_terramind_l_upernet_corrected_indices_dice_bs2_accum4"
$LOG     = "$RUN_DIR/logs/step5s_a_bs2_accum4_training.log"

Write-Host "=== STEP 5S-A bs2/accum4 background launcher ==="
Write-Host "Repo:      $REPO"
Write-Host "Python:    $PYTHON"
Write-Host "Runner:    $RUNNER"
Write-Host "Config:    $CONFIG"
Write-Host "Run dir:   $RUN_DIR"
Write-Host "Log:       $LOG"

foreach ($path in @($REPO, $PYTHON, $RUNNER, $CONFIG)) {
    if (!(Test-Path $path)) { throw "Required path not found: $path" }
}

New-Item -ItemType Directory -Force -Path "$RUN_DIR/logs" | Out-Null

$process = Start-Process `
    -FilePath $PYTHON `
    -ArgumentList @($RUNNER) `
    -WorkingDirectory $REPO `
    -WindowStyle Hidden `
    -PassThru

Write-Host ""
Write-Host "STEP 5S-A bs2/accum4 training launched in background."
Write-Host "Launcher PID (PowerShell parent): $PID"
Write-Host "Python process PID:               $($process.Id)"
Write-Host "Log path: $LOG"
Write-Host ""
Write-Host "To monitor:"
Write-Host "  Get-Content -Wait -Tail 40 '$LOG'"
