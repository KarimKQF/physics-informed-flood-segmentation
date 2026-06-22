# ============================================================
# STEP 5P â€” Big/production classical training
# TerraMind-L pretrained + UPerNet
# ============================================================

$ErrorActionPreference = "Stop"

# ----------------------------
# Paths
# ----------------------------
$REPO = "C:/Users/Karim/Desktop/flood-segmentation-training/physics-informed-flood-segmentation"
$VENV = "E:/flood_research/venvs/terramind-gpu"
$EXP_ROOT = "E:/flood_research/experiments"
$BASE_RUN = "E:/flood_research/experiments/terramind_baseline/runs/step5o_terramind_l_upernet_long_classical_training"
$NEW_RUN = "E:/flood_research/experiments/terramind_baseline/runs/step5p_terramind_l_upernet_big_classical_training"

$OLD_CONFIG = "$BASE_RUN/configs/terramind_l_upernet_long_classical_train.yaml"
$NEW_CONFIG_DIR = "$NEW_RUN/configs"
$NEW_CONFIG = "$NEW_CONFIG_DIR/terramind_l_upernet_big_classical_train.yaml"

$PYTHON = "$VENV/Scripts/python.exe"

# ----------------------------
# Training budget
# ----------------------------
$MAX_EPOCHS = 80
$PATIENCE = 15
$MIN_EPOCHS = 30

# ----------------------------
# Safety checks
# ----------------------------
Write-Host "=== STEP 5P launcher ==="

if (!(Test-Path $REPO)) {
    throw "Repo not found: $REPO"
}

if (!(Test-Path $PYTHON)) {
    throw "Python venv not found: $PYTHON"
}

if (!(Test-Path $BASE_RUN)) {
    throw "STEP 5O run not found: $BASE_RUN"
}

if (!(Test-Path $OLD_CONFIG)) {
    throw "STEP 5O config not found: $OLD_CONFIG"
}

Set-Location $REPO

Write-Host "Repo: $REPO"
Write-Host "Python: $PYTHON"
Write-Host "Base config: $OLD_CONFIG"
Write-Host "New run: $NEW_RUN"

# ----------------------------
# Create folders
# ----------------------------
$folders = @(
    "$NEW_RUN",
    "$NEW_RUN/configs",
    "$NEW_RUN/checkpoints",
    "$NEW_RUN/logs",
    "$NEW_RUN/predictions",
    "$NEW_RUN/predictions/valid",
    "$NEW_RUN/predictions/test",
    "$NEW_RUN/predictions/bolivia",
    "$NEW_RUN/metrics",
    "$NEW_RUN/reports",
    "$NEW_RUN/reports/figures"
)

foreach ($f in $folders) {
    if (!(Test-Path $f)) {
        New-Item -ItemType Directory -Path $f | Out-Null
    }
}

# ----------------------------
# Copy and patch config
# ----------------------------
Copy-Item $OLD_CONFIG $NEW_CONFIG -Force

$patchScript = @"
from pathlib import Path
import re

config_path = Path(r"$NEW_CONFIG")
text = config_path.read_text(encoding="utf-8")

old_run = r"$BASE_RUN".replace("\\", "/")
new_run = r"$NEW_RUN".replace("\\", "/")

text = text.replace(old_run, new_run)
text = text.replace(r"$BASE_RUN".replace("/", "\\"), r"$NEW_RUN".replace("/", "\\"))

def replace_or_append(text, key, value):
    pattern = rf"(^\s*{re.escape(key)}\s*:\s*).*$"
    if re.search(pattern, text, flags=re.MULTILINE):
        return re.sub(pattern, rf"\g<1>{value}", text, flags=re.MULTILINE)
    return text + f"\n{key}: {value}\n"

for key in ["max_epochs", "epochs", "num_epochs"]:
    text = replace_or_append(text, key, $MAX_EPOCHS)

for key in ["patience", "early_stopping_patience"]:
    text = replace_or_append(text, key, $PATIENCE)

text = replace_or_append(text, "precision", 32)
text = replace_or_append(text, "batch_size", 1)
text = replace_or_append(text, "ignore_index", -1)

text += """
# ------------------------------------------------------------
# STEP 5P metadata
# ------------------------------------------------------------
step_name: step5p_terramind_l_upernet_big_classical_training
source_step: step5o_terramind_l_upernet_long_classical_training
training_budget: big_classical_training
max_epochs_requested: $MAX_EPOCHS
early_stopping_patience_requested: $PATIENCE
min_epochs_expected: $MIN_EPOCHS
physics_loss_started: false
darn_started: false
sturm_training_started: false
raw_data_modified: false
"""

config_path.write_text(text, encoding="utf-8")
print(f"Patched config written to: {config_path}")
"@

$patchFile = "$NEW_RUN/patch_step5p_config.py"
$patchScript | Out-File -FilePath $patchFile -Encoding utf8
& $PYTHON $patchFile

# ----------------------------
# Locate STEP 5O runner
# ----------------------------
Write-Host "Searching for existing STEP 5O training runner..."

$candidates = Get-ChildItem -Path $REPO -Recurse -Include *.py |
    Where-Object {
        $_.FullName -notmatch "\\.venv|\\venv|__pycache__" -and
        (
            (Select-String -Path $_.FullName -Pattern "step5o|terramind_l_upernet_long|TerraMind-L|TerraMind_v1_large|UPerNet" -Quiet)
        )
    } |
    Select-Object -ExpandProperty FullName

if ($candidates.Count -eq 0) {
    Write-Host "Could not automatically find the STEP 5O training runner."
    Write-Host "The config has been created here:"
    Write-Host $NEW_CONFIG
    Write-Host ""
    Write-Host "Search manually in the repo for the script used in STEP 5O, then run it with this config."
    throw "No training runner found."
}

Write-Host "Candidate runners:"
$candidates | ForEach-Object { Write-Host " - $_" }

$runner = $candidates | Where-Object { $_ -match "5o|step5o" } | Select-Object -First 1

if (!$runner) {
    $runner = $candidates | Select-Object -First 1
}

Write-Host "Selected runner:"
Write-Host $runner

# ----------------------------
# Launch training
# ----------------------------
$LOG = "$NEW_RUN/logs/step5p_big_training_console.log"

Write-Host "Launching STEP 5P training..."
Write-Host "Log: $LOG"

& $PYTHON $runner --config $NEW_CONFIG 2>&1 | Tee-Object -FilePath $LOG

Write-Host "Training command completed."

# ----------------------------
# Post-run reminder
# ----------------------------
Write-Host ""
Write-Host "STEP 5P run directory:"
Write-Host $NEW_RUN
Write-Host ""
Write-Host "Now verify that the run created:"
Write-Host "- checkpoints/best_checkpoint.pt"
Write-Host "- checkpoints/last_checkpoint.pt"
Write-Host "- metrics/valid_summary.json"
Write-Host "- metrics/test_summary.json"
Write-Host "- metrics/bolivia_summary.json"
Write-Host "- reports/STEP_5P_terramind_l_upernet_big_classical_training_report.md"
Write-Host ""
Write-Host "If the previous runner did not automatically evaluate valid/test/Bolivia, run the existing inference/evaluation script manually using the same STEP 5P run directory."
