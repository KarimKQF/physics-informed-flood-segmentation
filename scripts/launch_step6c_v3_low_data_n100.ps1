$ErrorActionPreference = "Stop"

$repo   = "C:/flood_research/repos/physics-informed-flood-segmentation"
$python = "E:/flood_research/venvs/terramind-gpu/Scripts/python.exe"
$script = "scripts/step6c_v3_train.py"
$config = "configs/low_data/step6c_v3_low_data_n100_seed42_lambda05_warmup.yaml"
$logdir = "E:/flood_research/experiments/terramind_baseline/runs/step6c_v3_low_data_n100_seed42_lambda05_warmup/logs"
$stdout = "$logdir/step6c_v3_low_data_n100_stdout.log"
$stderr = "$logdir/step6c_v3_low_data_n100_stderr.log"

Set-Location $repo

$proc = Start-Process `
    -FilePath $python `
    -ArgumentList "$script --config $config" `
    -WorkingDirectory $repo `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError  $stderr `
    -NoNewWindow `
    -PassThru

$proc.Id | Out-File "$logdir/step6c_v3_low_data_n100_pid.txt" -NoNewline
Write-Output "Launched PID $($proc.Id)"
Write-Output "stdout: $stdout"
Write-Output "stderr: $stderr"
