param(
    [ValidateSet("dry-run", "download")]
    [string]$Mode = "dry-run",
    [int]$MaxWorkers = 4
)

# ============================================================
# STEP 6B2b - Copernicus DEM GLO-30 AWS public download
# ============================================================

$ErrorActionPreference = "Stop"

$REPO = "C:/Users/Karim/Desktop/flood-segmentation-training/physics-informed-flood-segmentation"
$PYTHON = "E:/flood_research/venvs/terramind-gpu/Scripts/python.exe"
$RUN_DIR = "E:/flood_research/experiments/terramind_baseline/runs/step6b2b_copernicus_dem_aws_download"
$REQUIRED_CELLS = "E:/flood_research/experiments/terramind_baseline/runs/step6b2_dem_source_acquisition/manifests/required_dem_cells.csv"
$OUTPUT_ROOT = "E:/flood_research/data/raw/dem/copernicus_glo30"
$SCRIPT = "$REPO/scripts/physics/step6b2b_download_copernicus_dem_aws.py"
$LOG_DIR = "$RUN_DIR/logs"

New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null

if (!(Test-Path $REPO)) {
    throw "Repo not found: $REPO"
}
if (!(Test-Path $PYTHON)) {
    throw "Python not found: $PYTHON"
}
if (!(Test-Path $REQUIRED_CELLS)) {
    throw "Required cells CSV not found: $REQUIRED_CELLS"
}
if (!(Test-Path $SCRIPT)) {
    throw "Download script not found: $SCRIPT"
}

Set-Location $REPO

$argsList = @(
    $SCRIPT,
    "--required-cells", $REQUIRED_CELLS,
    "--output-root", $OUTPUT_ROOT,
    "--run-dir", $RUN_DIR,
    "--source", "copernicus_glo30",
    "--aws-no-sign-request",
    "--max-workers", "$MaxWorkers"
)

if ($Mode -eq "download") {
    $argsList += "--download"
    $LOG = "$LOG_DIR/step6b2b_copernicus_dem_download.log"
} else {
    $argsList += "--dry-run"
    $LOG = "$LOG_DIR/step6b2b_copernicus_dem_dry_run.log"
}

Write-Host "STEP 6B2b mode: $Mode"
Write-Host "Run dir: $RUN_DIR"
Write-Host "Output root: $OUTPUT_ROOT"
Write-Host "Log: $LOG"

& $PYTHON @argsList 2>&1 | Tee-Object -FilePath $LOG
