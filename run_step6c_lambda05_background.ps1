$ErrorActionPreference = "Stop"

$RepoRoot = "C:\flood_research\repos\physics-informed-flood-segmentation"
$Python = "E:\flood_research\venvs\terramind-gpu\Scripts\python.exe"
$RunDir = "E:\flood_research\experiments\terramind_baseline\runs\step6c_terramind_l_upernet_dice_topographic_lambda05"
$ConfigPath = Join-Path $RepoRoot "configs\step6c_terramind_l_upernet_dice_topographic_lambda05.yaml"
$TrainScript = Join-Path $RepoRoot "scripts\step6c_lambda05_train.py"
$LogPath = Join-Path $RunDir "logs\step6c_lambda05_training.log"
$StdoutLog = Join-Path $RunDir "logs\step6c_lambda05_stdout.log"
$StderrLog = Join-Path $RunDir "logs\step6c_lambda05_stderr.log"
$LaunchInfo = Join-Path $RunDir "metadata\step6c_lambda05_launch_info.json"

foreach ($subdir in @("logs", "metadata")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $RunDir $subdir) | Out-Null
}

$existingArtifacts = @(
    (Join-Path $RunDir "checkpoints\best_checkpoint.pt"),
    (Join-Path $RunDir "checkpoints\last_checkpoint.pt"),
    (Join-Path $RunDir "metrics\training_state.json"),
    (Join-Path $RunDir "metrics\training_epoch_metrics.csv"),
    (Join-Path $RunDir "metrics\step6c_lambda05_final_metrics.json")
) | Where-Object { Test-Path -LiteralPath $_ }

if ($existingArtifacts.Count -gt 0) {
    throw "Refusing to overwrite existing STEP 6C lambda=0.5 artifacts: $($existingArtifacts -join ', ')"
}

$arguments = @(
    "scripts\step6c_lambda05_train.py",
    "--config",
    $ConfigPath,
    "--log-file",
    $LogPath
)

$child = Start-Process `
    -FilePath $Python `
    -ArgumentList $arguments `
    -WorkingDirectory $RepoRoot `
    -RedirectStandardOutput $StdoutLog `
    -RedirectStandardError $StderrLog `
    -WindowStyle Hidden `
    -PassThru

$info = [ordered]@{
    step = "6C-lambda05"
    launched_at = (Get-Date).ToUniversalTime().ToString("o")
    repo_root = $RepoRoot
    config_path = $ConfigPath
    run_dir = $RunDir
    launcher_script = $PSCommandPath
    log_path = $LogPath
    stdout_log = $StdoutLog
    stderr_log = $StderrLog
    parent_pid = $PID
    python_child_pid = $child.Id
    command = "$Python scripts\step6c_lambda05_train.py --config `"$ConfigPath`" --log-file `"$LogPath`""
    guardrails = @{
        lambda_topo = 0.5
        dem_as_model_input = $false
        dem_in_loss_only = $true
        darn_started = $false
        sturm_training_started = $false
        raw_data_modified = $false
        initialization = "original TerraMind pretrained checkpoint"
    }
}

$info | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $LaunchInfo -Encoding UTF8

Write-Output "config_path=$ConfigPath"
Write-Output "run_dir=$RunDir"
Write-Output "launcher_script=$PSCommandPath"
Write-Output "log_path=$LogPath"
Write-Output "parent_pid=$PID"
Write-Output "python_child_pid=$($child.Id)"
Write-Output "monitor_command=Get-Content -Wait -Tail 50 -LiteralPath '$LogPath'"
