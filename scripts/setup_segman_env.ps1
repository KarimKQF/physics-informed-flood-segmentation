# Setup for the SegMAN (CVPR 2025) subproject.
#
# The pure-PyTorch path reuses the existing terramind-gpu venv with NO new or
# conflicting packages (no mmcv / mmsegmentation / natten / mamba-ssm / triton).
# This script only (1) verifies the venv and (2) vendors the official SegMAN
# source into external/SegMAN (git-ignored). See docs/segman_setup.md.

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Venv = "E:/flood_research/venvs/terramind-gpu/Scripts/python.exe"
$External = Join-Path $RepoRoot "external/SegMAN"
$SegmanCommit = "9ced66ab27f23b0dbc1b0a2fe0ba42e133c55ac8"

Write-Host "== Verifying venv ==" -ForegroundColor Cyan
if (-not (Test-Path $Venv)) { throw "venv python not found: $Venv" }
& $Venv -c "import torch, terratorch, segmentation_models_pytorch, timm, einops; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

Write-Host "== Vendoring SegMAN @ $SegmanCommit ==" -ForegroundColor Cyan
if (Test-Path (Join-Path $External ".git")) {
    Write-Host "external/SegMAN already present; skipping clone."
} else {
    git clone https://github.com/yunxiangfu2001/SegMAN.git $External
    Push-Location $External
    git checkout $SegmanCommit
    Pop-Location
}

Write-Host "== Verifying SegMAN loads with pure-torch shims ==" -ForegroundColor Cyan
& $Venv -c "import sys; sys.path.insert(0, r'$RepoRoot/experiments_cvpr/segman'); from segman_kernels.compat import load_segman_encoder_module, load_segman_decoder_module; load_segman_encoder_module(); load_segman_decoder_module(); print('SegMAN encoder + decoder import OK (pure-torch kernels)')"

Write-Host "`nSetup complete. Run smoke tests:" -ForegroundColor Green
Write-Host "  & '$Venv' experiments_cvpr/segman/smoke_tests.py --config configs/segman/segman_dice_ce_topo_dem_shuffled.yaml"
