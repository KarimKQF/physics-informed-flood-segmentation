# SegMAN N=100 D8 downstream loss experiment -- Sequential chain launcher (seed0 only)
#
# 2 runs: D8 real DEM | D8 shuffled DEM
#
# SAFETY RULES:
#   - Detects any running train_segman.py and ABORTs (or kills with -KillOrphans).
#   - Never overwrites an existing completed run.
#   - Never runs CE, Dice+CE baseline, N=200, seeds 1/2, V1-topo, or parallel training.
#   - DEM is not a model input (enforced by train_segman.py itself).
#
# PREREQUISITE: Run validate_and_smoke_d8_n100.py first and confirm all PASS.
#
# DETACHED LAUNCH (survives terminal closure):
#
#   $ROOT = "C:/flood_research/repos/physics-informed-flood-segmentation"
#   $proc = Start-Process powershell.exe `
#       -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$ROOT\scripts\launch_segman_n100_d8_seed0_chain.ps1`"" `
#       -WindowStyle Hidden -PassThru
#   Write-Host "Chain PID: $($proc.Id)"
#
# STATUS CHECK:
#   Get-Content E:/flood_research/logs/segman_n100_d8_seed0_master_stdout.log -Tail 30
#
# RESUME from a specific run tag:
#   -StartFrom segman_n100_d8_dem_shuffled_lambda1p0_seed0
#
# KILL ORPHANS explicitly:
#   -KillOrphans

param(
    [string]$StartFrom = "",
    [switch]$KillOrphans
)

$ErrorActionPreference = "Continue"

$PY      = "E:/flood_research/venvs/terramind-gpu/Scripts/python.exe"
$ROOT    = "C:/flood_research/repos/physics-informed-flood-segmentation"
$CFGDIR  = "$ROOT/configs/segman/d8_n100_seed0"
$RUNS    = "E:/flood_research/experiments/segman/runs"
$LOGFILE = "E:/flood_research/logs/segman_n100_d8_seed0_master_stdout.log"
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
        Where-Object { $_.Name -match "python" -and $_.CommandLine -and $_.CommandLine -match "train_segman\.py" }
}

function Stop-TrainingProcesses {
    param([object[]]$procs)
    foreach ($p in $procs) {
        try { Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop; Write-Chain "Killed PID=$($p.ProcessId)" }
        catch { Write-Chain "Could not kill PID=$($p.ProcessId): $_" "WARN" }
    }
    Start-Sleep -Seconds 5
}

function Test-RunHealth {
    param([string]$RunDir, [string]$Tag)

    $best_ckpt    = Join-Path $RunDir "checkpoints/best_checkpoint.pt"
    $last_ckpt    = Join-Path $RunDir "checkpoints/last_checkpoint.pt"
    $summary_path = Join-Path $RunDir "metrics/${Tag}_summary.json"
    $csv_path     = Join-Path $RunDir "metrics/training_epoch_metrics.csv"
    $stdout_log   = Join-Path $RunDir "logs/${Tag}_chain_stdout.log"
    $stderr_log   = Join-Path $RunDir "logs/${Tag}_chain_stderr.log"

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
    if ($best_miou -lt 0.20) {
        Write-Chain "HEALTH FAIL [$Tag]: best_miou=$($best_miou.ToString('F5')) < 0.20 (degenerate)" "ERROR"
        return $false
    }

    if (Test-Path $csv_path) {
        $last_row = Get-Content $csv_path -Encoding UTF8 | Select-Object -Last 1
        if ($last_row -match "\bNaN\b|\bInf\b") {
            Write-Chain "HEALTH FAIL [$Tag]: NaN or Inf in last CSV row" "ERROR"
            return $false
        }
    }

    foreach ($lf in @($stdout_log, $stderr_log)) {
        if (Test-Path $lf) {
            if (Select-String -Path $lf -Pattern "CUDA out of memory|OutOfMemoryError|Traceback" -Quiet) {
                Write-Chain "HEALTH FAIL [$Tag]: OOM or traceback detected in logs" "ERROR"
                return $false
            }
        }
    }

    $le = if ($s.PSObject.Properties["last_epoch"]) { $s.last_epoch } else { "?" }
    Write-Chain "HEALTH OK [$Tag]: status=done | best_miou=$($best_miou.ToString('F5')) | best_epoch=$($s.best_epoch) | last_epoch=$le"
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

# ── Variant table (real first, then shuffled) ────────────────────────────────
$variants = @(
    [pscustomobject]@{
        tag      = "segman_n100_d8_lambda1p0_seed0"
        config   = "$CFGDIR/n100_seed0_dice_ce_d8_lambda1p0.yaml"
        run_dir  = "$RUNS/segman_n100_d8_lambda1p0_seed0"
    }
    [pscustomobject]@{
        tag      = "segman_n100_d8_dem_shuffled_lambda1p0_seed0"
        config   = "$CFGDIR/n100_seed0_dice_ce_d8_dem_shuffled_lambda1p0.yaml"
        run_dir  = "$RUNS/segman_n100_d8_dem_shuffled_lambda1p0_seed0"
    }
)

# ── Orphan check ──────────────────────────────────────────────────────────────
$orphans = @(Find-TrainingProcesses)
if ($orphans.Count -gt 0) {
    Write-Chain "WARN: $($orphans.Count) train_segman.py process(es) already running:" "WARN"
    foreach ($o in $orphans) {
        Write-Chain "  PID=$($o.ProcessId)  cmd=$($o.CommandLine.Substring(0,[math]::Min(120,$o.CommandLine.Length)))" "WARN"
    }
    if (-not $KillOrphans) {
        Write-Chain "ABORT: training already running. Wait or re-run with -KillOrphans." "ERROR"
        exit 2
    }
    Write-Chain "Killing orphaned processes (-KillOrphans passed)..."
    Stop-TrainingProcesses $orphans
    Write-Chain "Orphans cleared."
}

# ── StartFrom index ───────────────────────────────────────────────────────────
$start_idx = 0
if ($StartFrom -ne "") {
    $found = $false
    for ($i = 0; $i -lt $variants.Count; $i++) {
        if ($variants[$i].tag -eq $StartFrom) { $start_idx = $i; $found = $true; break }
    }
    if (-not $found) {
        Write-Chain "ABORT: -StartFrom '$StartFrom' not in variant table." "ERROR"
        exit 1
    }
}

Write-Chain "===== SegMAN N=100 D8 Downstream Loss (seed0) START -- total=$($variants.Count) runs, start=$start_idx ====="
Write-Chain "Conditions: D8 real DEM | D8 shuffled DEM (negative control)"
Write-Chain "NO CE, NO Dice+CE baseline, NO V1-topo, NO N=200, NO seeds 1/2, NO parallel training."
Write-Chain "DEM is NOT a model input. Prerequisite: validate_and_smoke_d8_n100.py PASSED."

# ── Main loop ─────────────────────────────────────────────────────────────────
for ($i = $start_idx; $i -lt $variants.Count; $i++) {
    $v = $variants[$i]
    Write-Chain "--- [$($i+1)/$($variants.Count)] $($v.tag) ---"

    # Safety: refuse any CE-only or baseline-only configs.
    if ($v.config -notmatch "dice_ce_d8") {
        Write-Chain "ABORT: Refusing non-D8 config: $($v.config)" "ERROR"
        exit 1
    }

    # Skip already-completed runs.
    $summary_path = Join-Path $v.run_dir "metrics/$($v.tag)_summary.json"
    if (Test-Path $summary_path) {
        try {
            $ex = Get-Content $summary_path -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($ex.status -eq "done") {
                if (Test-RunHealth -RunDir $v.run_dir -Tag $v.tag) {
                    Write-Chain "SKIP [$($v.tag)]: already done (best_miou=$($ex.best_validation_miou) best_epoch=$($ex.best_epoch))"
                    continue
                } else {
                    Write-Chain "ABORT [$($v.tag)]: summary says done but health check failed." "ERROR"
                    exit 1
                }
            }
        } catch {
            Write-Chain "WARN [$($v.tag)]: could not parse summary: $_" "WARN"
        }
    }

    if (Test-PartialRun -RunDir $v.run_dir -Tag $v.tag) {
        Write-Chain "ABORT [$($v.tag)]: Partial run in $($v.run_dir). Remove manually to restart." "ERROR"
        exit 1
    }

    if (-not (Test-Path $v.config)) {
        Write-Chain "ABORT [$($v.tag)]: Config missing: $($v.config)" "ERROR"
        exit 1
    }

    $run_stdout = Join-Path $v.run_dir "logs/$($v.tag)_chain_stdout.log"
    $run_stderr = Join-Path $v.run_dir "logs/$($v.tag)_chain_stderr.log"
    $null = New-Item -ItemType Directory -Force -Path (Split-Path $run_stdout)

    Write-Chain "CMD: $PY `"$SCRIPT`" --config `"$($v.config)`""
    $t_start = Get-Date

    $proc = Start-Process -FilePath $PY `
        -ArgumentList @($SCRIPT, "--config", $v.config) `
        -Wait -PassThru -NoNewWindow `
        -RedirectStandardOutput $run_stdout `
        -RedirectStandardError  $run_stderr
    $exit_code = $proc.ExitCode

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

    if (-not (Test-RunHealth -RunDir $v.run_dir -Tag $v.tag)) {
        Write-Chain "CHAIN STOPPED: health check failed after $($v.tag)" "ERROR"
        exit 1
    }

    Write-Chain "--- $($v.tag) COMPLETE (run $($i+1)/$($variants.Count)) ---"
}

Write-Chain "===== SegMAN N=100 D8 (seed0) COMPLETE (all $($variants.Count) runs done) ====="
Write-Chain "Next step: python experiments_cvpr/segman/aggregate_segman_n100_d8_seed0.py"
exit 0
