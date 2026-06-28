# SegMAN N=100 Diagnostic — Sequential chain launcher
#
# 9 runs: 3 conditions x 3 seeds
#   seed0: Dice+CE  |  Topo real  |  Topo shuffled
#   seed1: Dice+CE  |  Topo real  |  Topo shuffled
#   seed2: Dice+CE  |  Topo real  |  Topo shuffled
#
# SAFETY RULES:
#   - Detects any running train_segman.py and ABORTs (or kills with -KillOrphans).
#   - Never overwrites an existing completed run.
#   - Never runs CE, N=200, or lambda sweeps.
#   - Never runs parallel training.
#   - DEM is not a model input (enforced by train_segman.py itself).
#
# DETACHED LAUNCH (survives terminal closure):
#
#   $ROOT = "C:/flood_research/repos/physics-informed-flood-segmentation"
#   $proc = Start-Process powershell.exe `
#       -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$ROOT\scripts\launch_segman_n100_diagnostic_chain.ps1`"" `
#       -WindowStyle Hidden -PassThru
#   Write-Host "Chain PID: $($proc.Id)"
#
# STATUS CHECK:
#   Get-Content E:/flood_research/logs/segman_n100_diagnostic_master_stdout.log -Tail 40
#
# RESUME from a specific run tag:
#   -StartFrom segman_n100_dice_ce_topo_seed1
#
# KILL ORPHANS explicitly:
#   -KillOrphans

param(
    [string]$StartFrom   = "",
    [switch]$KillOrphans
)

$ErrorActionPreference = "Continue"

$PY      = "E:/flood_research/venvs/terramind-gpu/Scripts/python.exe"
$ROOT    = "C:/flood_research/repos/physics-informed-flood-segmentation"
$CFGDIR  = "$ROOT/configs/segman/multiseed_n100"
$RUNS    = "E:/flood_research/experiments/segman/runs"
$LOGFILE = "E:/flood_research/logs/segman_n100_diagnostic_master_stdout.log"
$SCRIPT  = "$ROOT/experiments_cvpr/segman/train_segman.py"

$null = New-Item -ItemType Directory -Force -Path (Split-Path $LOGFILE)

# ── Logging ───────────────────────────────────────────────────────────────────
function Write-Chain {
    param([string]$msg, [string]$level = "INFO")
    $ts   = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "$ts $level  $msg"
    Add-Content -Path $LOGFILE -Value $line -Encoding UTF8
    Write-Output $line
}

# ── Orphan detection ──────────────────────────────────────────────────────────
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

# ── Health check after each run ───────────────────────────────────────────────
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
            Write-Chain "HEALTH FAIL [$Tag]: NaN or Inf detected in last CSV row" "ERROR"
            return $false
        }

        if ($LossMode -match "topo") {
            $headers = (Get-Content $csv_path -Encoding UTF8 | Select-Object -First 1) -split ","
            $vals    = $last_row -split ","
            $lt_idx  = [array]::IndexOf($headers, "lambda_topo_loss")
            if ($lt_idx -ge 0) {
                $lt_val = [double]$vals[$lt_idx]
                if ($lt_val -le 0.0) {
                    Write-Chain "HEALTH FAIL [$Tag]: lambda_topo_loss=$lt_val (expected > 0 for topo)" "ERROR"
                    return $false
                }
            }
        }

        $log_path = Join-Path $RunDir "logs/${Tag}_training.log"
        if (Test-Path $log_path) {
            if (Select-String -Path $log_path -Pattern "CUDA out of memory|OutOfMemoryError" -Quiet) {
                Write-Chain "HEALTH FAIL [$Tag]: OOM detected in training log" "ERROR"
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
        } catch { }
    }
    return $true
}

# ── Variant table (order: seed0 first, by condition) ─────────────────────────
$variants = @(
    [pscustomobject]@{
        tag       = "segman_n100_dice_ce_seed0"
        config    = "$CFGDIR/n100_seed0_dice_ce.yaml"
        run_dir   = "$RUNS/segman_n100_dice_ce_seed0"
        loss_mode = "dice_ce"
    }
    [pscustomobject]@{
        tag       = "segman_n100_dice_ce_topo_seed0"
        config    = "$CFGDIR/n100_seed0_dice_ce_topo.yaml"
        run_dir   = "$RUNS/segman_n100_dice_ce_topo_seed0"
        loss_mode = "dice_ce_topo"
    }
    [pscustomobject]@{
        tag       = "segman_n100_dice_ce_topo_dem_shuffled_seed0"
        config    = "$CFGDIR/n100_seed0_dice_ce_topo_dem_shuffled.yaml"
        run_dir   = "$RUNS/segman_n100_dice_ce_topo_dem_shuffled_seed0"
        loss_mode = "dice_ce_topo_dem_shuffled"
    }
    [pscustomobject]@{
        tag       = "segman_n100_dice_ce_seed1"
        config    = "$CFGDIR/n100_seed1_dice_ce.yaml"
        run_dir   = "$RUNS/segman_n100_dice_ce_seed1"
        loss_mode = "dice_ce"
    }
    [pscustomobject]@{
        tag       = "segman_n100_dice_ce_topo_seed1"
        config    = "$CFGDIR/n100_seed1_dice_ce_topo.yaml"
        run_dir   = "$RUNS/segman_n100_dice_ce_topo_seed1"
        loss_mode = "dice_ce_topo"
    }
    [pscustomobject]@{
        tag       = "segman_n100_dice_ce_topo_dem_shuffled_seed1"
        config    = "$CFGDIR/n100_seed1_dice_ce_topo_dem_shuffled.yaml"
        run_dir   = "$RUNS/segman_n100_dice_ce_topo_dem_shuffled_seed1"
        loss_mode = "dice_ce_topo_dem_shuffled"
    }
    [pscustomobject]@{
        tag       = "segman_n100_dice_ce_seed2"
        config    = "$CFGDIR/n100_seed2_dice_ce.yaml"
        run_dir   = "$RUNS/segman_n100_dice_ce_seed2"
        loss_mode = "dice_ce"
    }
    [pscustomobject]@{
        tag       = "segman_n100_dice_ce_topo_seed2"
        config    = "$CFGDIR/n100_seed2_dice_ce_topo.yaml"
        run_dir   = "$RUNS/segman_n100_dice_ce_topo_seed2"
        loss_mode = "dice_ce_topo"
    }
    [pscustomobject]@{
        tag       = "segman_n100_dice_ce_topo_dem_shuffled_seed2"
        config    = "$CFGDIR/n100_seed2_dice_ce_topo_dem_shuffled.yaml"
        run_dir   = "$RUNS/segman_n100_dice_ce_topo_dem_shuffled_seed2"
        loss_mode = "dice_ce_topo_dem_shuffled"
    }
)

# ── Orphan check ──────────────────────────────────────────────────────────────
$orphans = @(Find-TrainingProcesses)
if ($orphans.Count -gt 0) {
    Write-Chain "WARN: Found $($orphans.Count) train_segman.py process(es) already running:" "WARN"
    foreach ($o in $orphans) { Write-Chain "  $(Get-ProcessInfo $o)" "WARN" }
    if (-not $KillOrphans) {
        Write-Chain "ABORT: training already running. Wait or re-run with -KillOrphans." "ERROR"
        exit 2
    }
    Write-Chain "Killing orphaned processes (-KillOrphans was passed)..."
    Stop-TrainingProcesses $orphans
    Write-Chain "Orphans cleared."
}

# ── StartFrom index ───────────────────────────────────────────────────────────
$start_idx = 0
if ($StartFrom -ne "") {
    $found = $false
    for ($i = 0; $i -lt $variants.Count; $i++) {
        if ($variants[$i].tag -eq $StartFrom) {
            $start_idx = $i; $found = $true; break
        }
    }
    if (-not $found) {
        Write-Chain "ABORT: -StartFrom '$StartFrom' not found in variant table." "ERROR"
        exit 1
    }
}

Write-Chain "===== SegMAN N=100 Diagnostic Chain START (total=$($variants.Count) runs, start=$start_idx, KillOrphans=$KillOrphans) ====="
Write-Chain "Conditions: Dice+CE | Topo real | Topo shuffled  x  Seeds 0, 1, 2"
Write-Chain "CE NOT launched. N=200 NOT launched. DEM is NOT a model input."

# ── Main loop ─────────────────────────────────────────────────────────────────
for ($i = $start_idx; $i -lt $variants.Count; $i++) {
    $v = $variants[$i]
    Write-Chain "--- [$($i+1)/$($variants.Count)] $($v.tag) ---"

    # Safety: refuse any CE config (should never happen, but belts-and-suspenders)
    if ($v.config -match "_ce\.yaml$" -and $v.config -notmatch "dice_ce") {
        Write-Chain "ABORT: Refusing to launch CE-only config: $($v.config)" "ERROR"
        exit 1
    }

    # Skip already-completed runs
    $summary_path = Join-Path $v.run_dir "metrics/$($v.tag)_summary.json"
    if (Test-Path $summary_path) {
        try {
            $ex = Get-Content $summary_path -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($ex.status -eq "done") {
                if (Test-RunHealth -RunDir $v.run_dir -Tag $v.tag -LossMode $v.loss_mode) {
                    Write-Chain "SKIP [$($v.tag)]: already done (best_miou=$($ex.best_validation_miou) best_epoch=$($ex.best_epoch))"
                    continue
                } else {
                    Write-Chain "ABORT [$($v.tag)]: summary says done but health check failed." "ERROR"
                    exit 1
                }
            }
        } catch {
            Write-Chain "WARN [$($v.tag)]: could not parse summary JSON: $_" "WARN"
        }
    }

    # Abort on partial runs (CSV exists but no done status)
    if (Test-PartialRun -RunDir $v.run_dir -Tag $v.tag) {
        Write-Chain "ABORT [$($v.tag)]: Partial run detected in $($v.run_dir). Remove manually to restart." "ERROR"
        exit 1
    }

    # Config must exist
    if (-not (Test-Path $v.config)) {
        Write-Chain "ABORT [$($v.tag)]: Config missing: $($v.config)" "ERROR"
        exit 1
    }

    # Per-run logs in run directory
    $run_stdout = Join-Path $v.run_dir "logs/$($v.tag)_chain_stdout.log"
    $run_stderr = Join-Path $v.run_dir "logs/$($v.tag)_chain_stderr.log"
    $null = New-Item -ItemType Directory -Force -Path (Split-Path $run_stdout)

    $py_args_str = "`"$SCRIPT`" --config `"$($v.config)`""
    Write-Chain "CMD: $PY $py_args_str"
    $t_start = Get-Date

    # Use Start-Process for OS-level redirection — avoids PowerShell 5.1 ErrorRecord
    # wrapping that kills the chain when Python writes to stderr (warnings, tqdm, etc.)
    $proc = Start-Process -FilePath $PY `
        -ArgumentList @($SCRIPT, "--config", $v.config) `
        -Wait -PassThru -NoNewWindow `
        -RedirectStandardOutput $run_stdout `
        -RedirectStandardError  $run_stderr
    $exit_code = $proc.ExitCode

    # Append stdout+stderr to master log
    foreach ($lf in @($run_stdout, $run_stderr)) {
        if (Test-Path $lf) {
            Get-Content $lf -Encoding UTF8 -ErrorAction SilentlyContinue |
                ForEach-Object { Add-Content -Path $LOGFILE -Value $_ -Encoding UTF8 }
        }
    }

    $elapsed = [math]::Round(((Get-Date) - $t_start).TotalMinutes, 1)
    Write-Chain "$($v.tag) exited (code=$exit_code, elapsed=${elapsed}min)"

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

Write-Chain "===== SegMAN N=100 Diagnostic Chain COMPLETE (all $($variants.Count) runs done) ====="
exit 0
