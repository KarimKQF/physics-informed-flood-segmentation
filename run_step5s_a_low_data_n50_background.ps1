# STEP 5S-A low-data N=50 baseline launcher.
# Launches only the Dice baseline pilot: no physics, no DEM, no DARN, no STURM,
# no SegFormer, no Mamba.

$ErrorActionPreference = "Stop"

$RepoRoot = "C:/flood_research/repos/physics-informed-flood-segmentation"
$Python = "E:/flood_research/venvs/terramind-gpu/Scripts/python.exe"
$ConfigPath = "configs/low_data/step5s_a_low_data_n50_seed42.yaml"
$ScriptPath = "scripts/step5s_a_low_data_train.py"
$RunDir = "E:/flood_research/experiments/terramind_baseline/runs/step5s_a_low_data_n50_seed42_dice"
$LogPath = "$RunDir/logs/step5s_a_bs2_accum4_training.log"
$StdoutLog = "$RunDir/logs/step5s_a_low_data_n50_stdout.log"
$StderrLog = "$RunDir/logs/step5s_a_low_data_n50_stderr.log"
$LaunchInfo = "$RunDir/metadata/step5s_a_low_data_n50_launch_info.json"

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
    throw "Training wrapper not found: $ScriptPath"
}

foreach ($subdir in @("logs", "metadata")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $RunDir $subdir) | Out-Null
}

$existingArtifacts = @(
    "$RunDir/checkpoints/best_checkpoint.pt",
    "$RunDir/checkpoints/last_checkpoint.pt",
    "$RunDir/metrics/training_epoch_metrics.csv",
    "$RunDir/metrics/training_state.json",
    "$RunDir/metrics/step5s_a_low_data_n50_seed42_dice_summary.json"
) | Where-Object { Test-Path -LiteralPath $_ }

if ($existingArtifacts.Count -gt 0) {
    throw "Refusing to overwrite existing STEP 5S-A low-data N=50 artifacts: $($existingArtifacts -join ', ')"
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

$command = 'cd C:/flood_research/repos/physics-informed-flood-segmentation; & "E:/flood_research/venvs/terramind-gpu/Scripts/python.exe" "scripts/step5s_a_low_data_train.py" --config "configs/low_data/step5s_a_low_data_n50_seed42.yaml"'
$monitorCommand = "Get-Content -Wait -Tail 50 -LiteralPath '$LogPath'"

$info = [ordered]@{
    step = "5S-A-low-data-n50-seed42"
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
        physics_run_started = $false
        n100_started = $false
        dem_as_model_input = $false
        dem_loaded = $false
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
