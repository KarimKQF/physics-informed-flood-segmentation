# STEP 5S-A bs2/accum4 — RESUME from epoch 15 background launcher.
# Resumes from best_checkpoint.pt saved at epoch 14 (val_mIoU=0.8432).
# Original run crashed at epoch 15 start due to Windows WDDM TDR driver reset.
# TDR delay was NOT changed (no admin rights). Monitor for TDR recurrence.
#
# Usage: .\run_step5s_a_resume_epoch15_background.ps1
# Monitor: Get-Content -Wait -Tail 40 "<RUN_DIR>/logs/step5s_a_resume_from_epoch15.log"

$ErrorActionPreference = "Stop"

$REPO    = "C:/flood_research/repos/physics-informed-flood-segmentation"
$PYTHON  = "E:/flood_research/venvs/terramind-gpu/Scripts/python.exe"
$RUNNER  = "$REPO/scripts/step5s_a_bs2_accum4_train.py"
$CONFIG  = "$REPO/configs/step5s_a_terramind_l_upernet_corrected_indices_dice_bs2_accum4.yaml"
$RUN_DIR = "E:/flood_research/experiments/terramind_baseline/runs/step5s_a_terramind_l_upernet_corrected_indices_dice_bs2_accum4"
$CKPT    = "$RUN_DIR/checkpoints/best_checkpoint.pt"
$LOG     = "$RUN_DIR/logs/step5s_a_resume_from_epoch15.log"

Write-Host "=== STEP 5S-A bs2/accum4 RESUME launcher ==="
Write-Host "Repo:       $REPO"
Write-Host "Python:     $PYTHON"
Write-Host "Runner:     $RUNNER"
Write-Host "Config:     $CONFIG"
Write-Host "Checkpoint: $CKPT"
Write-Host "Run dir:    $RUN_DIR"
Write-Host "Log:        $LOG"

foreach ($path in @($REPO, $PYTHON, $RUNNER, $CONFIG, $CKPT)) {
    if (!(Test-Path $path)) { throw "Required path not found: $path" }
}

New-Item -ItemType Directory -Force -Path "$RUN_DIR/logs" | Out-Null

$argsList = @(
    $RUNNER,
    "--resume", $CKPT,
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
Write-Host "STEP 5S-A bs2/accum4 RESUME launched in background."
Write-Host "Launcher PID (PowerShell parent): $PID"
Write-Host "Python process PID:               $($process.Id)"
Write-Host "Resume checkpoint: $CKPT"
Write-Host "Log path: $LOG"
Write-Host ""
Write-Host "To monitor:"
Write-Host "  Get-Content -Wait -Tail 40 '$LOG'"
