# E1+E2 Multi-seed N=50 launch script
# DO NOT run all at once — run one condition at a time sequentially.
# Each run takes ~1.2–1.7h. Total: ~20-25h for 15 runs.
#
# Recommended order:
#   1. All 5 Dice-only runs first (E1 baseline — to confirm/deny collapse pattern)
#   2. All 5 Physics real DEM runs (E1 rescue — same seeds as above)
#   3. All 5 Physics shuffled DEM runs (E2 control — whether real DEM matters)
#
# Usage:
#   Launch one run:
#     .\launch_e1_e2_multiseed.ps1 -Seed 0 -Condition dice_only
#   Launch all Dice-only:
#     .\launch_e1_e2_multiseed.ps1 -Batch dice_only
#   Launch all Physics real DEM:
#     .\launch_e1_e2_multiseed.ps1 -Batch physics_real
#   Launch all Physics shuffled DEM:
#     .\launch_e1_e2_multiseed.ps1 -Batch physics_shuffled
#
# NOTE: This script does NOT run in parallel. Runs are sequential per batch to avoid OOM.

param(
    [int]$Seed = -1,
    [string]$Condition = "",
    [string]$Batch = ""    # dice_only | physics_real | physics_shuffled
)

$repo   = "C:/flood_research/repos/physics-informed-flood-segmentation"
$python = "E:/flood_research/venvs/terramind-gpu/Scripts/python.exe"
$script = "$repo/scripts/step6c_v3_train.py"
$seeds  = @(0, 1, 2, 3, 42)

function Launch-Run([int]$s, [string]$cond) {
    $config  = "$repo/configs/multiseed_n50/n50_seed${s}_${cond}.yaml"
    $tag     = "n50_seed${s}_${cond}"
    $logdir  = "E:/flood_research/experiments/terramind_baseline/runs/${tag}/logs"
    New-Item -ItemType Directory -Force -Path $logdir | Out-Null
    $stdout  = "$logdir/${tag}_stdout.log"
    $stderr  = "$logdir/${tag}_stderr.log"

    Write-Host "Launching: seed=$s condition=$cond"
    Write-Host "  config : $config"
    Write-Host "  stdout : $stdout"

    $proc = Start-Process `
        -FilePath $python `
        -ArgumentList "$script --config $config" `
        -WorkingDirectory $repo `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError  $stderr `
        -NoNewWindow `
        -PassThru

    $proc.Id | Out-File "$logdir/${tag}_pid.txt" -NoNewline
    Write-Host "  PID    : $($proc.Id)"

    # Wait for process to complete before starting next (sequential)
    $proc.WaitForExit()
    $exit = $proc.ExitCode
    Write-Host "  ExitCode: $exit"
    if ($exit -ne 0) {
        Write-Warning "Run FAILED (exit=$exit). Check $stderr"
    }
    return $exit
}

if ($Batch -eq "dice_only") {
    foreach ($s in $seeds) { Launch-Run $s "dice_only" }
} elseif ($Batch -eq "physics_real") {
    foreach ($s in $seeds) { Launch-Run $s "physics_real_dem_lambda05" }
} elseif ($Batch -eq "physics_shuffled") {
    foreach ($s in $seeds) { Launch-Run $s "physics_shuffled_dem_lambda05" }
} elseif ($Seed -ge 0 -and $Condition -ne "") {
    Launch-Run $Seed $Condition
} else {
    Write-Host "Usage:"
    Write-Host "  .\launch_e1_e2_multiseed.ps1 -Seed 0 -Condition dice_only"
    Write-Host "  .\launch_e1_e2_multiseed.ps1 -Batch dice_only"
    Write-Host "  .\launch_e1_e2_multiseed.ps1 -Batch physics_real"
    Write-Host "  .\launch_e1_e2_multiseed.ps1 -Batch physics_shuffled"
}
