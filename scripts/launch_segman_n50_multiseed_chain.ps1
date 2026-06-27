# SegMAN N=50 multi-seed sequential chain launcher
#
# Runs 16 SegMAN loss-ablation variants sequentially:
#   Seeds 1, 2, 3, 42  x  CE / Dice+CE / Dice+CE+Topo / Dice+CE+Topo+Shuffled
#
# Seed 0 is SKIPPED (already complete).
#
# SAFETY RULES
#   - Default: detect any running train_segman.py process, REPORT it, ABORT.
#   - -KillOrphans: explicitly kill detected orphans before proceeding.
#   - Never overwrites an existing done run directory.
#   - Never modifies seed0 runs.
#
# DETACHED LAUNCH (survives terminal/session closure):
#
#   $ROOT = "C:/flood_research/repos/physics-informed-flood-segmentation"
#   $proc = Start-Process powershell.exe `
#       -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$ROOT\scripts\launch_segman_n50_multiseed_chain.ps1`"" `
#       -WindowStyle Hidden -PassThru
#   Write-Host "Chain PID: $($proc.Id)"
#
# RESUME from a specific run tag:
#   -StartFrom segman_dice_ce_seed2
#
# KILL ORPHANS explicitly:
#   -KillOrphans
#
# STATUS CHECK:
#   Get-Content E:/flood_research/experiments/segman/multiseed_n50_chain.log -Tail 40

param(
    [string]$StartFrom   = "",
    [switch]$KillOrphans
)

$ErrorActionPreference = "Stop"

$PY      = "E:/flood_research/venvs/terramind-gpu/Scripts/python.exe"
$ROOT    = "C:/flood_research/repos/physics-informed-flood-segmentation"
$CFGDIR  = "$ROOT/configs/segman/multiseed_n50"
$RUNS    = "E:/flood_research/experiments/segman/runs"
$LOGFILE = "E:/flood_research/experiments/segman/multiseed_n50_chain.log"
$SCRIPT  = "$ROOT/experiments_cvpr/segman/train_segman.py"

$null = New-Item -ItemType Directory -Force -Path (Split-Path $LOGFILE)

function Write-Chain {
    param([string]$msg, [string]$level = "INFO")
    $ts   = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "$ts $level  $msg"
    Add-Content -Path $LOGFILE -Value $line -Encoding UTF8
    Write-Output $line
}

function Find-TrainingProcesses {
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -match "python" -and
            $_.CommandLine -and
            ($_.CommandLine -match "train_segman\.py")
        }
}

function Get-ProcessInfo {
    param($proc)
    $cmd = $proc.CommandLine
    $cfg = if ($cmd -match '--config\s+"?([^\s"]+)"?') { $Matches[1] } else { "(unknown)" }
    $age = if ($proc.CreationDate) {
        "$([math]::Round(((Get-Date)-$proc.CreationDate).TotalMinutes,1)) min"
    } else { "?" }
    return "PID=$($proc.ProcessId) age=$age config=$(Split-Path $cfg -Leaf 2>$null)"
}

function Stop-TrainingProcesses {
    param([object[]]$procs)
    foreach ($p in $procs) {
        try {
            Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop
            Write-Chain "Killed PID=$($p.ProcessId)"
        } catch {
            Write-Chain "Could not kill PID=$($p.ProcessId): $_" "WARN"
        }
    }
    Start-Sleep -Seconds 5
}

$orphans = @(Find-TrainingProcesses)
if ($orphans.Count -gt 0) {
    Write-Chain "WARN: Found $($orphans.Count) train_segman.py process(es) already running:" "WARN"
    foreach ($o in $orphans) { Write-Chain "  $(Get-ProcessInfo $o)" "WARN" }
    if (-not $KillOrphans) {
        Write-Chain "ABORT: A training run is active. Wait for it, or re-run with -KillOrphans." "ERROR"
        exit 2
    }
    Write-Chain "Killing orphaned processes (-KillOrphans was passed)..."
    Stop-TrainingProcesses $orphans
    Write-Chain "Orphans cleared."
}

function Test-RunHealth {
    param([string]$RunDir, [string]$Tag, [string]$LossMode)

    $best_ckpt    = Join-Path $RunDir "checkpoints/best_checkpoint.pt"
    $last_ckpt    = Join-Path $RunDir "checkpoints/last_checkpoint.pt"
    $summary_path = Join-Path $RunDir "metrics/${Tag}_summary.json"
    $csv_path     = Join-Path $RunDir "metrics/training_epoch_metrics.csv"

    foreach ($f in @($best_ckpt, $last_ckpt)) {
        if (-not (Test-Path $f)) {
            Write-Chain "HEALTH FAIL [$Tag]: $(Split-Path $f -Leaf) missing" "ERROR"
            return $false
        }
    }
    if (-not (Test-Path $summary_path)) {
        Write-Chain "HEALTH FAIL [$Tag]: summary JSON missing" "ERROR"
        return $false
    }

    $s = Get-Content $summary_path -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($s.status -ne "done") {
        Write-Chain "HEALTH FAIL [$Tag]: status='$($s.status)' (expected 'done')" "ERROR"
        return $false
    }

    $best_miou = [double]$s.best_validation_miou
    if ($best_miou -lt 0.30) {
        Write-Chain "HEALTH FAIL [$Tag]: best_miou=$($best_miou.ToString('F5')) < 0.30 (degenerate)" "ERROR"
        return $false
    }

    if (Test-Path $csv_path) {
        $last_row = Get-Content $csv_path -Encoding UTF8 | Select-Object -Last 1
        if ($last_row -match "\bNaN\b|\bInf\b") {
            Write-Chain "HEALTH FAIL [$Tag]: NaN or Inf in last CSV row" "ERROR"
            return $false
        }

        if ($LossMode -match "topo") {
            $headers = (Get-Content $csv_path -Encoding UTF8 | Select-Object -First 1) -split ","
            $vals    = $last_row -split ","
            $lt_idx  = [array]::IndexOf($headers, "lambda_topo_loss")
            if ($lt_idx -ge 0) {
                $lt_val = [double]$vals[$lt_idx]
                if ($lt_val -le 0.0) {
                    Write-Chain "HEALTH FAIL [$Tag]: lambda_topo_loss=$lt_val (expected > 0 for topo variant)" "ERROR"
                    return $false
                }
            }
        }

        $log_path = Join-Path $RunDir "logs/${Tag}_training.log"
        if (Test-Path $log_path) {
            if (Select-String -Path $log_path -Pattern "CUDA out of memory|OutOfMemoryError" -Quiet) {
                Write-Chain "HEALTH FAIL [$Tag]: OOM detected in log" "ERROR"
                return $false
            }
        }
    }

    $epochs = if ($s.PSObject.Properties['last_epoch']) { $s.last_epoch } else { "?" }
    Write-Chain "HEALTH OK [$Tag]: status=done | best_miou=$($best_miou.ToString('F5')) | best_epoch=$($s.best_epoch) | total_epochs=$epochs | loss_mode=$LossMode"
    return $true
}

function Test-PartialRun {
    param([string]$RunDir, [string]$Tag)
    $csv     = Join-Path $RunDir "metrics/training_epoch_metrics.csv"
    $summary = Join-Path $RunDir "metrics/${Tag}_summary.json"
    if (-not (Test-Path $csv)) { return $false }
    if (Test-Path $summary) {
        try {
            $s = Get-Content $summary -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($s.status -eq "done") { return $false }
        } catch {
            # JSON parse error; treat as partial
        }
    }
    return $true
}

$variants = @()
foreach ($seed in @(1, 2, 3, 42)) {
    $variants += [pscustomobject]@{
        tag       = "segman_ce_seed${seed}"
        config    = "$CFGDIR/segman_ce_n50_seed${seed}.yaml"
        run_dir   = "$RUNS/segman_ce_seed${seed}"
        loss_mode = "ce"
    }
    $variants += [pscustomobject]@{
        tag       = "segman_dice_ce_seed${seed}"
        config    = "$CFGDIR/segman_dice_ce_n50_seed${seed}.yaml"
        run_dir   = "$RUNS/segman_dice_ce_seed${seed}"
        loss_mode = "dice_ce"
    }
    $variants += [pscustomobject]@{
        tag       = "segman_dice_ce_topo_seed${seed}"
        config    = "$CFGDIR/segman_dice_ce_topo_n50_seed${seed}.yaml"
        run_dir   = "$RUNS/segman_dice_ce_topo_seed${seed}"
        loss_mode = "dice_ce_topo"
    }
    $variants += [pscustomobject]@{
        tag       = "segman_dice_ce_topo_dem_shuffled_seed${seed}"
        config    = "$CFGDIR/segman_dice_ce_topo_dem_shuffled_n50_seed${seed}.yaml"
        run_dir   = "$RUNS/segman_dice_ce_topo_dem_shuffled_seed${seed}"
        loss_mode = "dice_ce_topo_dem_shuffled"
    }
}

$start_idx = 0
if ($StartFrom -ne "") {
    $found = $false
    for ($i = 0; $i -lt $variants.Count; $i++) {
        if ($variants[$i].tag -eq $StartFrom) {
            $start_idx = $i
            $found = $true
            break
        }
    }
    if (-not $found) {
        Write-Chain "ABORT: -StartFrom '$StartFrom' not found in variant table." "ERROR"
        exit 1
    }
}

Write-Chain "===== SegMAN N=50 multi-seed chain START (total=$($variants.Count) runs, start_idx=$start_idx, KillOrphans=$KillOrphans) ====="

for ($i = $start_idx; $i -lt $variants.Count; $i++) {
    $v = $variants[$i]

    Write-Chain "--- [$($i+1)/$($variants.Count)] $($v.tag) ---"

    if ($v.run_dir -match "seed0") {
        Write-Chain "ABORT: run_dir contains 'seed0' -- will not modify seed0 runs." "ERROR"
        exit 1
    }

    $summary_path = Join-Path $v.run_dir "metrics/$($v.tag)_summary.json"
    if (Test-Path $summary_path) {
        try {
            $ex = Get-Content $summary_path -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($ex.status -eq "done") {
                if (Test-RunHealth -RunDir $v.run_dir -Tag $v.tag -LossMode $v.loss_mode) {
                    Write-Chain "SKIP [$($v.tag)]: already done (best_miou=$($ex.best_validation_miou) best_epoch=$($ex.best_epoch))"
                    continue
                } else {
                    Write-Chain "ABORT [$($v.tag)]: summary says done but health check failed -- run may be incomplete." "ERROR"
                    exit 1
                }
            }
        } catch {
            Write-Chain "WARN [$($v.tag)]: could not parse summary JSON: $_" "WARN"
        }
    }

    if (Test-PartialRun -RunDir $v.run_dir -Tag $v.tag) {
        Write-Chain "ABORT [$($v.tag)]: Partial run detected in $($v.run_dir)" "ERROR"
        Write-Chain "  CSV exists but summary status is not 'done'." "ERROR"
        Write-Chain "  Remove the run directory manually to restart, or use -StartFrom to skip." "ERROR"
        exit 1
    }

    if (-not (Test-Path $v.config)) {
        Write-Chain "ABORT [$($v.tag)]: Config file missing: $($v.config)" "ERROR"
        exit 1
    }

    $py_args = @($SCRIPT, "--config", $v.config)
    Write-Chain "CMD: $PY $($py_args -join ' ')"
    $t_start = Get-Date

    & $PY @py_args
    $exit_code = $LASTEXITCODE

    $elapsed = [math]::Round(((Get-Date) - $t_start).TotalMinutes, 1)
    Write-Chain "$($v.tag) exited (code=$exit_code, elapsed=${elapsed}min)"

    $pid_file = Join-Path $v.run_dir "run.pid"
    if (Test-Path $pid_file) {
        $train_pid = (Get-Content $pid_file -Raw -Encoding UTF8).Trim()
        Write-Chain "$($v.tag) training PID was $train_pid"
    }

    if ($exit_code -ne 0) {
        Write-Chain "CHAIN STOPPED: $($v.tag) exited with code $exit_code" "ERROR"
        exit 1
    }

    if (-not (Test-RunHealth -RunDir $v.run_dir -Tag $v.tag -LossMode $v.loss_mode)) {
        Write-Chain "CHAIN STOPPED: health check failed after $($v.tag)" "ERROR"
        exit 1
    }

    Write-Chain "--- $($v.tag) COMPLETE (run $($i+1)/$($variants.Count)) ---"
}

Write-Chain "===== SegMAN N=50 multi-seed chain COMPLETE (all $($variants.Count) runs done) ====="
exit 0
