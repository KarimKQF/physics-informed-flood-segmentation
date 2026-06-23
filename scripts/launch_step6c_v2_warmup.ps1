# Launcher: STEP 6C v2 warmup full training (background, detached).
# Recipe = STEP 5S-A baseline + topographic prior with warmup lambda schedule
#   (epochs 1-5 lambda=0; 6-20 ramp 0->0.5; 21+ = 0.5).
# Guardrails: DEM loss-only (never model input), FP32, init from original TerraMind
#   pretrained checkpoint (NOT 5S-A best), no DARN, no STURM, raw data unchanged,
#   never overwrites the failed lambda=0.5 run.
$ErrorActionPreference = "Stop"

$venv   = "E:/flood_research/venvs/terramind-gpu/Scripts/python.exe"
$repo   = "C:/flood_research/repos/physics-informed-flood-segmentation"
$script = "$repo/scripts/step6c_v2_train.py"
$config = "$repo/configs/step6c_v2_terramind_l_upernet_dice_topographic_lambda05_warmup.yaml"
$runDir = "E:/flood_research/experiments/terramind_baseline/runs/step6c_v2_terramind_l_upernet_dice_topographic_lambda05_warmup"
$logDir = "$runDir/logs"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$trainLog   = "$logDir/step6c_v2_lambda05_warmup_training.log"   # runner's own logging handler
$consoleOut = "$logDir/step6c_v2_lambda05_warmup_console.out.log"
$consoleErr = "$logDir/step6c_v2_lambda05_warmup_console.err.log"

$proc = Start-Process -FilePath $venv `
    -ArgumentList @($script, "--config", $config, "--log-file", $trainLog) `
    -WorkingDirectory $repo `
    -RedirectStandardOutput $consoleOut `
    -RedirectStandardError $consoleErr `
    -WindowStyle Hidden `
    -PassThru

$proc.Id | Out-File -FilePath "$logDir/training_pid.txt" -Encoding ascii

Write-Output "PYTHON_PID=$($proc.Id)"
Write-Output "TRAIN_LOG=$trainLog"
Write-Output "CONSOLE_OUT=$consoleOut"
Write-Output "CONSOLE_ERR=$consoleErr"
Write-Output "RUN_DIR=$runDir"
