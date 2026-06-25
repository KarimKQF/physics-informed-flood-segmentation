# SegMAN Integration — Audit & Migration Note

Date: 2026-06-25 · Branch: `experiments/segman-cvpr2025`

## New objective

The project's main experimental objective is now **evaluating physics-informed
loss functions on recent CVPR semantic-segmentation architectures**, with the
architecture held fixed and only the loss changed. Model line: **SegMAN (CVPR
2025)** first, then EoMT (CVPR 2025 Highlight), then ViT-P (CVPR 2026). The prior
TerraMind/SegFormer work is **preserved**, not replaced.

## Repository audit (summary)

- **Framework**: TerraTorch 1.2.8 on PyTorch Lightning, but training is driven by
  **hand-written plain-PyTorch loops** in `scripts/` (not the Lightning Trainer).
  Models are TerraTorch `SemanticSegmentationTask` + `EncoderDecoderFactory`. It
  is **not** MMSegmentation-based.
- **Data**: Sen1Floods11, multimodal **S2L1C (13 band) + S1GRD (2 band)** = 15
  channels, 512×512, mask `LabelHand` with `ignore_index=-1`, water=class 1.
- **DEM**: Copernicus GLO-30 aligned rasters on `E:/`, used **loss-only** via
  `batch["topography"]` (never a model input).
- **Losses** (`src/losses/combined_loss.py`): `CombinedDiceCELoss`,
  `CombinedDicePhysicsLoss`, topo term in `physics_topographic_loss.py`. No
  Dice+CE+Topo existed → added in the SegMAN subproject.
- **DEM shuffle**: reproducible derangement maps already exist
  (`manifests/.../dem_shuffle_map_n50_seed*.json`).
- **Metrics**: confusion-matrix based, inline in `step6c_lambda05_train.py`.
- **Existing `configs/multiseed_n50/`** already encodes the four-variant design
  (dice_only / dice_ce / physics_real_dem / physics_shuffled_dem) — the SegMAN
  job slots into the same N=50 seed-0 protocol.

## Environment constraint (the deciding factor)

Host has the GPU/driver (RTX 5000 Ada, 32 GB) **but no CUDA toolkit, no MSVC, no
WSL**. SegMAN's official stack (mmcv-full 1.x + mmseg 0.30 + NATTEN + Mamba SS2D
CUDA kernels) **cannot be compiled here** and would also conflict with the
TerraTorch environment.

## Chosen strategy — **Option C: model wrapper + pure-torch kernels**

Vendor only SegMAN's `nn.Module`s (encoder + MMSCopE decoder) and drive them with
the **existing** custom training loop, losses, metrics, DEM pipeline, and
DEM-shuffle machinery. SegMAN's compiled dependencies are replaced by
`sys.modules` shims with pure-PyTorch equivalents (validated):

- selective scan (Mamba SS2D): exact, unit-tested chunked SSD scan.
- cross-scan / NATTEN / mmcv / mmseg decode-head: pure-torch drop-ins.

Rejected: Option B (SegMAN's MMSegmentation) — legacy mmcv/mmseg, conflicts +
unbuildable on Windows. Option A (drop into TerraTorch's `EncoderDecoderFactory`)
— SegMAN is not a registered TerraTorch backbone.

## Decisions taken (confirmed with user)

- **Input = 15 channels** (S2+S1) via an inflated stem.
- **Kernels = pure-PyTorch fallback** (run today on Windows, slower).

## Safety / non-destructive guarantees

- New branch off current state; **no existing files modified** beyond appending
  `external/` to `.gitignore` and adding new files under `experiments_cvpr/`,
  `configs/segman/`, `docs/`.
- Pre-existing uncommitted changes from the prior experiment line left untouched.
- No existing runs, configs, checkpoints, or reports altered or deleted.
- New code isolated under `experiments_cvpr/segman/` + `external/SegMAN/`.
- Reuses (does not duplicate) the dataset, DEM loader, DEM-shuffle maps, topo
  loss, Dice loss, and metric helpers.

## What was added

```
external/SegMAN/                         # vendored official repo (git-ignored)
experiments_cvpr/segman/
  segman_kernels/                        # pure-torch scan, cross-scan, NATTEN, mmcv/mmseg shims
  model/segman_model.py                  # SegMANSegmentor (15-ch stem, MMSCopE decoder, upsample)
  segman_losses/segman_loss.py           # 4-mode config-driven loss selector (+Dice+CE+Topo)
  train_segman.py                        # training runner (reuses TopographyDataModule + metrics)
  debug_one_batch.py, smoke_tests.py     # smoke A/B/C/D
configs/segman/segman_{ce,dice_ce,dice_ce_topo,dice_ce_topo_dem_shuffled}.yaml
docs/segman_setup.md, docs/SEGMAN_MIGRATION_NOTE.md
```

See [segman_setup.md](segman_setup.md) for environment and run commands.
