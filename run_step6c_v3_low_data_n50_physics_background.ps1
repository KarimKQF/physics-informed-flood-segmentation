# STEP 6C/v3 low-data N=50 physics launcher.
# Launches only the paired N=50 physics run. DEM is loss-only and must never be
# used as model input. No N=100, lambda sweep, DARN, STURM, SegFormer, or Mamba.

$ErrorActionPreference = "Stop"

$RepoRoot = "C:/flood_research/repos/physics-informed-flood-segmentation"
$Python = "E:/flood_research/venvs/terramind-gpu/Scripts/python.exe"
$ConfigPath = "configs/low_data/step6c_v3_low_data_n50_seed42_lambda05_warmup.yaml"
$ScriptPath = "scripts/step6c_v3_train.py"
$RunDir = "E:/flood_research/experiments/terramind_baseline/runs/step6c_v3_low_data_n50_seed42_lambda05_warmup"
$RunTag = "step6c_v3_low_data_n50_seed42_lambda05_warmup"
$LogPath = "$RunDir/logs/${RunTag}_training.log"
$StdoutLog = "$RunDir/logs/${RunTag}_stdout.log"
$StderrLog = "$RunDir/logs/${RunTag}_stderr.log"
$LaunchInfo = "$RunDir/metadata/${RunTag}_launch_info.json"

if (!(Test-Path -LiteralPath $RepoRoot)) {
    throw "Repo root not found: $RepoRoot"
}
if (!(Test-Path -LiteralPath $Python)) {
    throw "Python venv not found: $Python"
}
if (!(Test-Path -LiteralPath (Join-Path $RepoRoot $ConfigPath))) {
    throw "Config not found: $ConfigPath"
}
if (!(Test-Path -LiteralPath (Join-Path $RepoRoot $ScriptPath))) {
    throw "Training script not found: $ScriptPath"
}

foreach ($subdir in @("logs", "metadata")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $RunDir $subdir) | Out-Null
}

$existingArtifacts = @(
    "$RunDir/checkpoints/best_checkpoint.pt",
    "$RunDir/checkpoints/last_checkpoint.pt",
    "$RunDir/metrics/training_epoch_metrics.csv",
    "$RunDir/metrics/training_state.json",
    "$RunDir/metrics/pure_dice_parity_metrics.json"
) | Where-Object { Test-Path -LiteralPath $_ }

if ($existingArtifacts.Count -gt 0) {
    throw "Refusing to overwrite existing STEP 6C/v3 low-data N=50 artifacts: $($existingArtifacts -join ', ')"
}

$arguments = @(
    $ScriptPath,
    "--config",
    $ConfigPath
)

$child = Start-Process `
    -FilePath $Python `
    -ArgumentList $arguments `
    -WorkingDirectory $RepoRoot `
    -RedirectStandardOutput $StdoutLog `
    -RedirectStandardError $StderrLog `
    -WindowStyle Hidden `
    -PassThru

$command = 'cd C:/flood_research/repos/physics-informed-flood-segmentation; & "E:/flood_research/venvs/terramind-gpu/Scripts/python.exe" "scripts/step6c_v3_train.py" --config "configs/low_data/step6c_v3_low_data_n50_seed42_lambda05_warmup.yaml"'
$monitorCommand = "Get-Content -Wait -Tail 50 -LiteralPath '$LogPath'"

$info = [ordered]@{
    step = "6C-v3-low-data-n50-seed42-lambda05-warmup"
    launched_at = (Get-Date).ToUniversalTime().ToString("o")
    repo_root = $RepoRoot
    config_path = (Join-Path $RepoRoot $ConfigPath)
    run_dir = $RunDir
    launcher_script = $PSCommandPath
    log_path = $LogPath
    stdout_log = $StdoutLog
    stderr_log = $StderrLog
    parent_pid = $PID
    python_child_pid = $child.Id
    command = $command
    monitor_command = $monitorCommand
    guardrails = @{
        physics_run_started = $true
        n50_physics_only = $true
        n100_started = $false
        lambda_sweep_started = $false
        dem_as_model_input = $false
        dem_in_loss_only = $true
        keep_dem_outside_batch_image = $true
        darn_started = $false
        sturm_training_started = $false
        segformer_started = $false
        mamba_started = $false
        raw_data_modified = $false
        preserve_existing_runs = $true
    }
}

$info | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $LaunchInfo -Encoding UTF8

Write-Output "config_path=$($info.config_path)"
Write-Output "run_dir=$RunDir"
Write-Output "launcher_script=$PSCommandPath"
Write-Output "log_path=$LogPath"
Write-Output "parent_pid=$PID"
Write-Output "python_child_pid=$($child.Id)"
Write-Output "monitor_command=$monitorCommand"
