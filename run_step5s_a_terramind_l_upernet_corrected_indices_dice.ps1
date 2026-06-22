# STEP 5S-A corrected TerraMind-L UPerNet indices launcher.
# Classical segmentation only: no physics loss, no topographic loss, no DEM input.

$ErrorActionPreference = "Stop"

$REPO = "C:/Users/Karim/Desktop/flood-segmentation-training/physics-informed-flood-segmentation"
$PYTHON = "E:/flood_research/venvs/terramind-gpu/Scripts/python.exe"
$RUNNER = "$REPO/scripts/step5s_a_corrected_indices_runner.py"
$CONFIG = "$REPO/configs/step5s_a_terramind_l_upernet_corrected_indices_dice.yaml"
$RUN_DIR = "E:/flood_research/experiments/terramind_baseline/runs/step5s_a_terramind_l_upernet_corrected_indices_dice"
$LOG = "$RUN_DIR/logs/step5s_a_training.log"

Write-Host "=== STEP 5S-A corrected-indices launcher ==="
Write-Host "Repo: $REPO"
Write-Host "Python: $PYTHON"
Write-Host "Runner: $RUNNER"
Write-Host "Config: $CONFIG"
Write-Host "Run directory: $RUN_DIR"
Write-Host "Log: $LOG"

if (!(Test-Path $REPO)) {
    throw "Repository not found: $REPO"
}
if (!(Test-Path $PYTHON)) {
    throw "Python venv not found: $PYTHON"
}
if (!(Test-Path $RUNNER)) {
    throw "Runner not found: $RUNNER"
}
if (!(Test-Path $CONFIG)) {
    throw "Config not found: $CONFIG"
}

New-Item -ItemType Directory -Force -Path "$RUN_DIR/logs" | Out-Null

$argsList = @(
    $RUNNER,
    "--mode",
    "train",
    "--config",
    $CONFIG
)

$process = Start-Process -FilePath $PYTHON -ArgumentList $argsList -WorkingDirectory $REPO -WindowStyle Hidden -PassThru
Start-Sleep -Seconds 5

$pythonChildPid = $process.Id
$children = Get-CimInstance Win32_Process -Filter "ParentProcessId=$($process.Id)" -ErrorAction SilentlyContinue
if ($children) {
    $pythonChild = $children | Where-Object { $_.CommandLine -match "step5s_a_corrected_indices_runner.py" } | Select-Object -First 1
    if ($pythonChild) {
        $pythonChildPid = [int]$pythonChild.ProcessId
    }
}

& $PYTHON $RUNNER --mode mark-running --config $CONFIG --parent-pid $PID --python-child-pid $pythonChildPid

Write-Host "STEP 5S-A training launched in background."
Write-Host "Parent PID: $PID"
Write-Host "Launcher Python PID: $($process.Id)"
Write-Host "Python child PID: $pythonChildPid"
Write-Host "Log path: $LOG"
