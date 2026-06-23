param(
    [switch]$Background,
    [switch]$Overwrite,
    [switch]$SaveNpz
)

# ============================================================
# STEP 6B4 - Full Copernicus DEM alignment for Sen1Floods11
# ============================================================

$ErrorActionPreference = "Stop"

$REPO = "C:/flood_research/repos/physics-informed-flood-segmentation"
$PYTHON = "E:/flood_research/venvs/terramind-gpu/Scripts/python.exe"
$RUN_DIR = "E:/flood_research/experiments/terramind_baseline/runs/step6b4_full_dem_alignment"
$LOG_DIR = "$RUN_DIR/logs"
$SCRIPT_DIR = "$RUN_DIR/scripts"
$ALIGN_SCRIPT = "$REPO/scripts/physics/step6b4_full_dem_alignment.py"
$SMOKE_SCRIPT = "$REPO/scripts/physics/step6b4_loss_compatibility_smoke.py"
$LOG = "$LOG_DIR/step6b4_full_dem_alignment.log"
$ERR_LOG = "$LOG_DIR/step6b4_full_dem_alignment_stderr.log"
$PID_FILE = "$LOG_DIR/STEP_6B4_full_alignment_process.pid"
$CHILD_PID_FILE = "$LOG_DIR/STEP_6B4_full_alignment_python_child.pid"
$BG_SCRIPT = "$SCRIPT_DIR/step6b4_full_alignment_background.ps1"

New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $SCRIPT_DIR | Out-Null

if (!(Test-Path $REPO)) {
    throw "Repo not found: $REPO"
}
if (!(Test-Path $PYTHON)) {
    throw "Python not found: $PYTHON"
}
if (!(Test-Path $ALIGN_SCRIPT)) {
    throw "Alignment script not found: $ALIGN_SCRIPT"
}
if (!(Test-Path $SMOKE_SCRIPT)) {
    throw "Smoke script not found: $SMOKE_SCRIPT"
}

Copy-Item -LiteralPath $ALIGN_SCRIPT -Destination "$SCRIPT_DIR/step6b4_full_dem_alignment.py" -Force
Copy-Item -LiteralPath $SMOKE_SCRIPT -Destination "$SCRIPT_DIR/step6b4_loss_compatibility_smoke.py" -Force
Copy-Item -LiteralPath "$REPO/run_step6b4_full_dem_alignment.ps1" -Destination "$SCRIPT_DIR/run_step6b4_full_dem_alignment.ps1" -Force

$alignArgs = @(
    $ALIGN_SCRIPT,
    "--run-dir", $RUN_DIR
)
if ($Overwrite) {
    $alignArgs += "--overwrite"
}
if ($SaveNpz) {
    $alignArgs += "--save-npz"
}

$smokeArgs = @(
    $SMOKE_SCRIPT,
    "--run-dir", $RUN_DIR
)

Write-Host "STEP 6B4 full DEM alignment"
Write-Host "Run dir: $RUN_DIR"
Write-Host "Log: $LOG"
Write-Host "Output DEM folder: E:/flood_research/data/derived/sen1floods11_topography/dem_aligned"

if ($Background) {
    $alignArgsLiteral = ($alignArgs | ForEach-Object { "'" + ($_ -replace "'", "''") + "'" }) -join ", "
    $smokeArgsLiteral = ($smokeArgs | ForEach-Object { "'" + ($_ -replace "'", "''") + "'" }) -join ", "
    $bgContent = @"
`$ErrorActionPreference = "Stop"
Set-Location '$REPO'
`$python = '$PYTHON'
`$alignArgs = @($alignArgsLiteral)
`$smokeArgs = @($smokeArgsLiteral)
`$log = '$LOG'
`$errLog = '$ERR_LOG'
"STEP 6B4 background workflow started: $(Get-Date -Format o)" | Out-File -FilePath `$log -Encoding utf8
try {
    & `$python @alignArgs 2>&1 | Tee-Object -FilePath `$log -Append
    `$alignExit = `$LASTEXITCODE
    "STEP 6B4 alignment exit code: `$alignExit" | Tee-Object -FilePath `$log -Append
    if (`$alignExit -ne 0) { exit `$alignExit }
    & `$python @smokeArgs 2>&1 | Tee-Object -FilePath `$log -Append
    `$smokeExit = `$LASTEXITCODE
    "STEP 6B4 smoke exit code: `$smokeExit" | Tee-Object -FilePath `$log -Append
    exit `$smokeExit
} catch {
    `$_.Exception.ToString() | Tee-Object -FilePath `$errLog -Append
    exit 1
}
"@
    Set-Content -LiteralPath $BG_SCRIPT -Value $bgContent -Encoding UTF8
    $process = Start-Process -FilePath "powershell.exe" -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $BG_SCRIPT
    ) -WindowStyle Hidden -PassThru
    $process.Id | Set-Content -LiteralPath $PID_FILE -Encoding ASCII
    Start-Sleep -Seconds 2
    $child = Get-CimInstance Win32_Process |
        Where-Object { $_.ParentProcessId -eq $process.Id -and $_.Name -match "python" } |
        Select-Object -First 1
    if ($child) {
        $child.ProcessId | Set-Content -LiteralPath $CHILD_PID_FILE -Encoding ASCII
        Write-Host "Parent PowerShell PID: $($process.Id)"
        Write-Host "Python child PID: $($child.ProcessId)"
    } else {
        Write-Host "Parent PowerShell PID: $($process.Id)"
        Write-Host "Python child PID: not detected yet; check $CHILD_PID_FILE or process tree."
    }
    Write-Host "PID file: $PID_FILE"
    Write-Host "Log path: $LOG"
    exit 0
}

Set-Location $REPO
& $PYTHON @alignArgs 2>&1 | Tee-Object -FilePath $LOG
$alignExit = $LASTEXITCODE
if ($alignExit -ne 0) {
    exit $alignExit
}
& $PYTHON @smokeArgs 2>&1 | Tee-Object -FilePath $LOG -Append
exit $LASTEXITCODE
