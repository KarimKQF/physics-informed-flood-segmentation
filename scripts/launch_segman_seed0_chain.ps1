# SegMAN seed-0 sequential chain launcher  (v2)
#
# Runs four SegMAN loss variants sequentially with orphan detection, PID
# tracking, and per-run health checks.  Safe to re-run: skips variants whose
# summary.json already has status=done.
#
# SAFETY RULES
#   - Default: detect any running train_segman.py process, REPORT it, and
#     ABORT. Never silently kill a training process.
#   - -KillOrphans: explicitly kill detected orphans before proceeding.
#
# DETACHED LAUNCH (survives terminal/session closure):
#
#   $ROOT = "C:/flood_research/repos/physics-informed-flood-segmentation"
#   $proc = Start-Process powershell.exe `
#       -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$ROOT\scripts\launch_segman_seed0_chain.ps1`"" `
#       -WindowStyle Hidden -PassThru
#   Write-Host "Chain PID: $($proc.Id)"
#
# RESUME from a specific variant:
#   ... -File `"...\launch_segman_seed0_chain.ps1`" -StartFrom segman_dice_ce_seed0
#
# KILL ORPHANS explicitly:
#   ... -File `"...\launch_segman_seed0_chain.ps1`" -KillOrphans
#
# LOGS
#   Chain orchestration : E:/flood_research/experiments/segman/chain_seed0.log
#   Per-run training    : E:/flood_research/experiments/segman/runs/<tag>/logs/<tag>_training.log
#   Per-run PID file    : E:/flood_research/experiments/segman/runs/<tag>/run.pid
#     (PID file written by train_segman.py at startup; persists after exit for
#      orphan detection in future chain invocations.)

param(
    [string]$StartFrom   = "segman_ce_seed0_clean",
    [switch]$KillOrphans
)

$ErrorActionPreference = "Stop"

$PY      = "E:/flood_research/venvs/terramind-gpu/Scripts/python.exe"
$ROOT    = "C:/flood_research/repos/physics-informed-flood-segmentation"
$RUNS    = "E:/flood_research/experiments/segman/runs"
$LOGFILE = "E:/flood_research/experiments/segman/chain_seed0.log"
$SCRIPT  = "$ROOT/experiments_cvpr/segman/train_segman.py"

$null = New-Item -ItemType Directory -Force -Path (Split-Path $LOGFILE)

function Write-Chain {
    param([string]$msg, [string]$level = "INFO")
    $ts   = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "$ts $level  $msg"
    Add-Content -Path $LOGFILE -Value $line -Encoding UTF8
    Write-Output $line
}

# ---------------------------------------------------------------------------
# Orphan detection
# ---------------------------------------------------------------------------
function Find-TrainingProcesses {
    <#
    Returns all Win32_Process objects that are Python processes running train_segman.py.
    Filters to python.exe only to avoid false positives from PowerShell sessions
    whose inline command text happens to contain the string "train_segman.py".
    #>
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
    $cfg = if ($cmd -match '--config\s+"?([^\s"]+)"?')  { $Matches[1] } else { "(unknown)" }
    $tag = if ($cmd -match '--run-tag\s+"?([^\s"]+)"?') { $Matches[1] } else { "(from config)" }
    $age = if ($proc.CreationDate) {
        $s = ((Get-Date) - $proc.CreationDate).TotalSeconds
        "$([math]::Round($s/60, 1)) min"
    } else { "?" }
    "PID=$($proc.ProcessId) age=$age config=$(Split-Path $cfg -Leaf) tag=$tag"
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
    Start-Sleep -Seconds 5   # let CUDA context release
}

# --- Startup orphan check ---------------------------------------------------
$orphans = @(Find-TrainingProcesses)
if ($orphans.Count -gt 0) {
    Write-Chain "WARN: Found $($orphans.Count) train_segman.py process(es) already running:" "WARN"
    foreach ($o in $orphans) {
        Write-Chain "  $(Get-ProcessInfo $o)" "WARN"
    }
    if (-not $KillOrphans) {
        Write-Chain "ABORT: A training run is active. Wait for it to finish, or re-run" "ERROR"
        Write-Chain "       with -KillOrphans to force-terminate it before proceeding." "ERROR"
        exit 2
    }
    Write-Chain "Killing orphaned processes (-KillOrphans was passed)..."
    Stop-TrainingProcesses $orphans
    Write-Chain "Orphans cleared."
}

# ---------------------------------------------------------------------------
# Health check (called after each run exits)
# ---------------------------------------------------------------------------
function Test-RunHealth {
    param([string]$RunDir, [string]$Tag)

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
        Write-Chain "HEALTH FAIL [$Tag]: summary status='$($s.status)' (expected 'done')" "ERROR"
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
    }

    $epochs = if ($s.PSObject.Properties['last_epoch']) { $s.last_epoch } else { "?" }
    Write-Chain "HEALTH OK [$Tag]: status=done | best_miou=$($best_miou.ToString('F5')) | best_epoch=$($s.best_epoch) | total_epochs=$epochs"
    return $true
}

# ---------------------------------------------------------------------------
# Detect partial run (artifacts exist but status != done)
# ---------------------------------------------------------------------------
function Test-PartialRun {
    param([string]$RunDir, [string]$Tag)
    $csv     = Join-Path $RunDir "metrics/training_epoch_metrics.csv"
    $summary = Join-Path $RunDir "metrics/${Tag}_summary.json"
    if (-not (Test-Path $csv)) { return $false }
    if (Test-Path $summary) {
        try {
            $s = Get-Content $summary -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($s.status -eq "done") { return $false }   # complete, not partial
        } catch {}
    }
    return $true  # CSV exists but not done → partial / interrupted
}

# ---------------------------------------------------------------------------
# Variant table
# ---------------------------------------------------------------------------
# CE uses CLI overrides (--run-dir / --run-tag) so the config file is left
# unmodified. The other three configs already embed the correct run_dir.
$variants = @(
    [pscustomobject]@{
        tag        = "segman_ce_seed0_clean"
        config     = "segman_ce.yaml"
        run_dir    = "$RUNS/segman_ce_seed0_clean"
        run_tag    = "segman_ce_seed0_clean"
        cli_extras = @("--run-dir", "$RUNS/segman_ce_seed0_clean",
                       "--run-tag", "segman_ce_seed0_clean")
    },
    [pscustomobject]@{
        tag        = "segman_dice_ce_seed0"
        config     = "segman_dice_ce.yaml"
        run_dir    = "$RUNS/segman_dice_ce_seed0"
        run_tag    = "segman_dice_ce_seed0"
        cli_extras = @()
    },
    [pscustomobject]@{
        tag        = "segman_dice_ce_topo_seed0"
        config     = "segman_dice_ce_topo.yaml"
        run_dir    = "$RUNS/segman_dice_ce_topo_seed0"
        run_tag    = "segman_dice_ce_topo_seed0"
        cli_extras = @()
    },
    [pscustomobject]@{
        tag        = "segman_dice_ce_topo_dem_shuffled_seed0"
        config     = "segman_dice_ce_topo_dem_shuffled.yaml"
        run_dir    = "$RUNS/segman_dice_ce_topo_dem_shuffled_seed0"
        run_tag    = "segman_dice_ce_topo_dem_shuffled_seed0"
        cli_extras = @()
    }
)

# Locate start index
$start_idx = 0
for ($i = 0; $i -lt $variants.Count; $i++) {
    if ($variants[$i].tag -eq $StartFrom) { $start_idx = $i; break }
}

Write-Chain "===== SegMAN seed-0 chain START (from=$StartFrom KillOrphans=$KillOrphans) ====="

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
for ($i = $start_idx; $i -lt $variants.Count; $i++) {
    $v = $variants[$i]

    # -- Skip if already fully done ------------------------------------------
    $summary_path = Join-Path $v.run_dir "metrics/$($v.run_tag)_summary.json"
    if (Test-Path $summary_path) {
        try {
            $ex = Get-Content $summary_path -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($ex.status -eq "done") {
                Write-Chain "SKIP [$($v.tag)]: already done (best_miou=$($ex.best_validation_miou) best_epoch=$($ex.best_epoch))"
                continue
            }
        } catch {}
    }

    # -- Abort if partial run artifacts exist --------------------------------
    if (Test-PartialRun -RunDir $v.run_dir -Tag $v.run_tag) {
        Write-Chain "ABORT [$($v.tag)]: Partial run detected in $($v.run_dir)" "ERROR"
        Write-Chain "  (CSV exists but summary status is not 'done'.)" "ERROR"
        Write-Chain "  Remove the run directory manually and re-run the chain, or" "ERROR"
        Write-Chain "  choose a different run name via cli_extras." "ERROR"
        exit 1
    }

    # -- Launch training ------------------------------------------------------
    Write-Chain "--- Launching $($v.tag) ---"
    $py_args = @($SCRIPT, "--config", "$ROOT/configs/segman/$($v.config)") + $v.cli_extras
    Write-Chain "CMD: $PY $($py_args -join ' ')"

    # Run synchronously; Python writes its own PID to {run_dir}/run.pid
    & $PY @py_args
    $exit_code = $LASTEXITCODE

    # -- Read back the training PID (for audit / future orphan detection) ----
    $pid_file = Join-Path $v.run_dir "run.pid"
    if (Test-Path $pid_file) {
        $train_pid = (Get-Content $pid_file -Raw -Encoding UTF8).Trim()
        Write-Chain "$($v.tag) training PID was $train_pid (from run.pid)"
    }

    Write-Chain "$($v.tag) launcher exited (code=$exit_code)"

    if ($exit_code -ne 0) {
        Write-Chain "CHAIN STOPPED: $($v.tag) exited with code $exit_code" "ERROR"
        exit 1
    }

    # -- Health check --------------------------------------------------------
    if (-not (Test-RunHealth -RunDir $v.run_dir -Tag $v.run_tag)) {
        Write-Chain "CHAIN STOPPED: health check failed after $($v.tag)" "ERROR"
        exit 1
    }

    Write-Chain "--- $($v.tag) COMPLETE ---"
}

Write-Chain "===== SegMAN seed-0 chain COMPLETE (all variants done) ====="
exit 0
