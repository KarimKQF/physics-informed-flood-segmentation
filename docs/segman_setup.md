# SegMAN (CVPR 2025) Setup & Usage

This document covers the SegMAN experimental subproject under
[`experiments_cvpr/segman/`](../experiments_cvpr/segman/): how the environment is
configured, how SegMAN runs without its compiled CUDA dependencies, and how to
run the four physics-informed-loss experiments.

> New main objective: **evaluate physics-informed loss functions on recent CVPR
> semantic-segmentation architectures**, starting with SegMAN. The architecture
> is held fixed; only the loss changes across the four variants.

## 1. Environment

SegMAN runs in the **existing** project venv — no new/conflicting packages are
installed (the pure-PyTorch path removes the need for `mmcv`, `mmsegmentation`,
`natten`, `mamba-ssm`, `causal-conv1d`, `triton`).

| Component | Version |
|---|---|
| venv | `E:/flood_research/venvs/terramind-gpu` |
| Python | 3.11.9 |
| PyTorch | 2.5.1+cu121 |
| GPU | NVIDIA RTX 5000 Ada (32 GB), driver 581.42 |
| terratorch / smp / timm / einops | 1.2.8 / 0.5.0 / 1.0.27 / 0.8.2 |

This host has **no CUDA toolkit (`nvcc`), no MSVC, no WSL**, so SegMAN's official
custom kernels cannot be compiled. We therefore run SegMAN's *unmodified* source
with pure-PyTorch drop-in kernels (see §3).

```powershell
# Activate the existing env (no installs needed for the pure-torch path)
E:/flood_research/venvs/terramind-gpu/Scripts/Activate.ps1
# One-time: clone the official SegMAN source (vendored, not committed)
.\scripts\setup_segman_env.ps1
```

## 2. Vendored SegMAN

The official repo is cloned to `external/SegMAN/` (pinned commit
`9ced66a`, CVPR 2025) and is git-ignored. We use the **unmodified** model files:

- `external/SegMAN/models/segman_encoder.py` — the SegMAN-S encoder (LASS blocks:
  NATTEN local attention + SS2D state-space token mixer).
- `external/SegMAN/segmentation/.../decode_heads/segman_decoder.py` — the MMSCopE
  decoder.

## 3. Pure-PyTorch kernels (how it runs without CUDA ops)

[`experiments_cvpr/segman/segman_kernels/`](../experiments_cvpr/segman/segman_kernels/)
registers drop-in `sys.modules` shims **before** importing SegMAN's files:

| Official dependency | Pure-torch replacement | Fidelity |
|---|---|---|
| `selective_scan_cuda_oflex` (Mamba SS2D) | `selective_scan.py` (chunked SSD scan) | **Exact** — unit-tested vs a sequential reference (max err ~1e-5) |
| `csm_triton` (Triton cross-scan) | `cross_scan.py` | Exact (differentiable VMamba cross-scan/merge) |
| `natten` (neighborhood attention) | `compat.na2d_qk/na2d_av` (unfold + valid-mask + central RPB) | Exact on interior pixels; small approximation at image borders only |
| `mmcv.cnn.ConvModule`, `mmseg` decode-head infra | `compat.py` shims (SyncBN→BN) | Functionally equivalent on 1 GPU |

`VSSM._selective_scan` is monkeypatched to the pure-torch scan; everything else
is SegMAN's original code. Model: SegMAN-S, **33.4 M params**, validated at
512×512 / batch 2: forward+backward **3.2 s**, **15.9 GB** with gradient
checkpointing.

## 4. Input & architecture decisions

- **Input: 15 channels** = Sentinel-2 L1C (13 bands) + Sentinel-1 GRD (2 bands).
  SegMAN's stem conv is inflated from 3→15 inputs. (Choose RGB-only later by
  setting `model.in_chans` + a 3-band assembler if desired.)
- **`image_size: 512`** must equal the tile size — SegMAN's stage-4 *global*
  attention uses a learned position bias tied to a fixed resolution. Sen1Floods11
  tiles are 512×512.
- **DEM is never a model input.** It is loaded into `batch["topography"]` and used
  only inside the topographic loss.
- **Pretrained weights**: not bundled. Provide `model.pretrained_encoder: <path>`
  to load an official SegMAN ImageNet encoder; the 3-ch stem is auto-inflated to
  15 ch. Default is train-from-scratch.

## 5. The four loss variants (architecture fixed)

Selected by config `loss.mode` (never hardcoded):

```
L_DiceCE = L_Dice + alpha * L_CE
L_total  = L_DiceCE + lambda_topo * L_topo
```

| Config | `loss.mode` | Loss |
|---|---|---|
| `configs/segman/segman_ce.yaml` | `ce` | CE only |
| `configs/segman/segman_dice_ce.yaml` | `dice_ce` | Dice + α·CE |
| `configs/segman/segman_dice_ce_topo.yaml` | `dice_ce_topo` | Dice + α·CE + λ·Topo (real DEM) |
| `configs/segman/segman_dice_ce_topo_dem_shuffled.yaml` | `dice_ce_topo_dem_shuffled` | same loss, DEM spatially deranged (control) |

Components reused unchanged: Dice = `smp.losses.DiceLoss` (parity with the
TerraMind baselines), Topo = `src/losses/physics_topographic_loss.py`, DEM +
reproducible shuffle map = `scripts/step6c_v3_train.TopographyDataModule`.
`ignore_index = -1`, `water_class = 1`, `alpha` and `lambda_topo` configurable;
`lambda_topo` supports a `warmup_linear` schedule.

## 6. Protocol

N=50 seed-0 low-data subset (`flood_train_low_data_n50_seed0.txt`), standard
filtered valid/test/Bolivia splits — **directly comparable to the existing
TerraMind N=50 physics experiments**. Optimizer AdamW lr=6e-5 wd=0.01, batch 2 ×
grad-accum 4 (eff. 8), fp32, gradient checkpointing, ReduceLROnPlateau, early
stopping (patience 15, min 30 epochs), max 80 epochs.

## 7. Commands

```powershell
$PY = "E:/flood_research/venvs/terramind-gpu/Scripts/python.exe"

# Debug one batch (shapes / stats / DEM / ignored pixels)
& $PY experiments_cvpr/segman/debug_one_batch.py --config configs/segman/segman_dice_ce_topo.yaml

# Smoke tests B/C/DEM-shuffle
& $PY experiments_cvpr/segman/smoke_tests.py --config configs/segman/segman_dice_ce_topo_dem_shuffled.yaml

# Train the four variants (seed 0)
& $PY experiments_cvpr/segman/train_segman.py --config configs/segman/segman_ce.yaml
& $PY experiments_cvpr/segman/train_segman.py --config configs/segman/segman_dice_ce.yaml
& $PY experiments_cvpr/segman/train_segman.py --config configs/segman/segman_dice_ce_topo.yaml
& $PY experiments_cvpr/segman/train_segman.py --config configs/segman/segman_dice_ce_topo_dem_shuffled.yaml
```

Each run writes to `E:/flood_research/experiments/segman/runs/<run_tag>/`:
`configs/`, `logs/`, `metrics/training_epoch_metrics.csv`, `metrics/<tag>_summary.json`,
`checkpoints/{best,last}_checkpoint.pt`, `predictions/`.

## 8. Multi-seed (after seed 0 is confirmed clean)

Duplicate the four configs, changing `seed`/`seed_everything`, `run_dir`,
`run_tag`, and (for the shuffled control) `dem.dem_tile_id_map_file` to the
matching `dem_shuffle_map_n50_seed{0,2,42}.json`. Do **not** launch multi-seed
until seed 0 is validated.

## 9. Known limitations

- Pure-torch SS2D + unfold-NATTEN are slower than the official CUDA kernels
  (acceptable; chosen deliberately for this Windows host).
- NATTEN border pixels are a slight approximation (interior is exact).
- BatchNorm runs with batch 2 (noisy stats) — identical across all four variants,
  so the loss comparison remains controlled. SegMAN ImageNet pretrained weights
  (when added) and/or larger effective batch will improve absolute accuracy.
