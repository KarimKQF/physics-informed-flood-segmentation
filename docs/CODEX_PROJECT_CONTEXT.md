# Codex Project Context

Date: 2026-06-26
Branch: `experiments/segman-cvpr2025`
Repository: `C:/flood_research/repos/physics-informed-flood-segmentation`
Experiment root: `E:/flood_research/experiments/segman/`

This project studies binary semantic segmentation of water/flood pixels from
satellite imagery, with physics-informed topographic losses evaluated against
standard segmentation losses. The current main research line has shifted from
Earth-observation-specific TerraMind/TerraTorch baselines toward recent
CVPR/AAAI semantic segmentation architectures, starting with SegMAN.

## Objective

Evaluate whether a topographic physics loss improves flood/water segmentation
when the architecture is held fixed and only the loss changes.

Current scientific question:

- Does real aligned DEM information improve generalization beyond Dice+CE?
- Or does the topo term mostly act as a generic/noisy regularizer?

The DEM-shuffled control is central to this question.

## Dataset And Modalities

Main dataset: filtered Sen1Floods11.

Task: binary semantic segmentation.

Classes:

- `0`: background / non-water
- `1`: water / flood
- `-1`: `ignore_index` / invalid pixels

Model input:

- 15 channels total
- 13 Sentinel-2 L1C bands
- 2 Sentinel-1/SAR GRD channels

Critical invariant:

- DEM/topography is not a model input.
- DEM is used only inside the topographic loss and topographic metrics.

This invariant is guarded in the SegMAN training code and configs.

## Model Hierarchy

Current main model:

- SegMAN-S, CVPR 2025
- Integrated under `experiments_cvpr/segman/`
- Dense semantic segmentation logits: `[B, 2, H, W]`
- Best fit now for CE, Dice+CE, and Dice+CE+Topo comparisons

Future candidates, not active now:

- EoMT, CVPR 2025 Highlight: likely harder because of query/mask-style outputs
- ViT-P, CVPR 2026: very recent/SOTA reference, risky to integrate
- VFMNet/VFMSeg, AAAI 2025: useful for domain generalization/OOD questions
- TerraMind, ICCV 2025 / Earth-observation baseline: existing experiments must
  be preserved as a secondary baseline

## SegMAN Implementation

Important implementation files:

- `experiments_cvpr/segman/train_segman.py`
- `experiments_cvpr/segman/model/segman_model.py`
- `experiments_cvpr/segman/segman_losses/segman_loss.py`
- `experiments_cvpr/segman/smoke_tests.py`
- `experiments_cvpr/segman/debug_one_batch.py`
- `experiments_cvpr/segman/aggregate_multiseed_results.py`
- `scripts/launch_segman_seed0_chain.ps1`
- `scripts/launch_segman_n50_multiseed_chain.ps1`
- `configs/segman/`
- `configs/segman/multiseed_n50/`

SegMAN source is vendored under `external/SegMAN/` and ignored by Git. The host
does not have a full CUDA toolkit/MSVC/WSL setup, so official SegMAN CUDA
kernels were not installed. Instead, pure-PyTorch compatibility shims are used
under `experiments_cvpr/segman/segman_kernels/`.

This is acceptable for pipeline validation and controlled loss comparison on
this machine. Publication-grade speed/fidelity may later require WSL/Linux plus
the official CUDA kernels.

Current SegMAN-S setup:

- 15-channel inflated input stem
- `image_size: 512`
- `num_classes: 2`
- train-from-scratch unless `pretrained_encoder` is provided
- gradient checkpointing enabled in configs
- outputs logits upsampled to full mask resolution

## Loss Variants

The controlled SegMAN ablation uses four variants:

1. CE
   - `L = L_CE`
2. Dice+CE
   - `L_DiceCE = L_Dice + alpha * L_CE`
3. Dice+CE+Topo with real DEM
   - `L_total = L_DiceCE + lambda_topo * L_topo(DEM_real)`
4. Dice+CE+Topo with shuffled DEM
   - `L_total = L_DiceCE + lambda_topo * L_topo(DEM_shuffled)`

Current topo coefficient:

- `lambda_topo = 0.5`
- warmup schedule: epochs 1-5 lambda 0, then linear ramp to 0.5 by epoch 20

Do not launch a lambda sweep now. Candidate values for later are `0.1`, `0.5`,
`1.0`, and `2.0`, but only after multi-seed results justify it.

## Seed0 Results

Completed SegMAN-S N=50 seed0 runs:

- `segman_ce_seed0_clean`
- `segman_dice_ce_seed0`
- `segman_dice_ce_topo_seed0`
- `segman_dice_ce_topo_dem_shuffled_seed0`

Summary:

| Loss | Best epoch | Total epochs | Val mIoU | Test mIoU | Bolivia mIoU |
|---|---:|---:|---:|---:|---:|
| CE | 43 | 58 | 0.8388 | 0.8420 | 0.8110 |
| Dice+CE | 44 | 59 | 0.8382 | 0.8389 | 0.8363 |
| Dice+CE+Topo real DEM | 44 | 59 | 0.8343 | 0.8496 | 0.8332 |
| Dice+CE+Topo DEM-shuffled | 44 | 59 | 0.8415 | 0.8637 | 0.8368 |

Water IoU:

| Loss | Val | Test | Bolivia |
|---|---:|---:|---:|
| CE | 0.7158 | 0.7268 | 0.6840 |
| Dice+CE | 0.7167 | 0.7234 | 0.7300 |
| Dice+CE+Topo real DEM | 0.7097 | 0.7414 | 0.7251 |
| Dice+CE+Topo DEM-shuffled | 0.7219 | 0.7646 | 0.7307 |

Water F1:

| Loss | Val | Test | Bolivia |
|---|---:|---:|---:|
| CE | 0.8343 | 0.8418 | 0.8123 |
| Dice+CE | 0.8350 | 0.8395 | 0.8440 |
| Dice+CE+Topo real DEM | 0.8302 | 0.8515 | 0.8406 |
| Dice+CE+Topo DEM-shuffled | 0.8385 | 0.8666 | 0.8444 |

Interpretation:

- SegMAN-S works well in the N=50 setting.
- All variants converged without collapse, NaN, Inf, or OOM in seed0.
- Dice+CE improves Bolivia/OOD versus CE.
- Real DEM topo improves test versus Dice+CE, but not validation or Bolivia.
- DEM-shuffled outperforms real DEM on seed0.
- Current seed0 evidence favors a regularization hypothesis over a clearly
  demonstrated physical DEM-driven effect.
- No strong physics-informed claim should be made until multi-seed analysis is
  complete.

## DEM-Shuffled Audit

The DEM-shuffled ablation is valid and scoped to training.

Verified behavior:

- Shuffle maps contain exactly the 50 N=50 training tile IDs.
- Dataset lookup uses `dem_tile_id = dem_map.get(tile_id, tile_id)`.
- Validation/test/Bolivia IDs are not in the training shuffle map, so they use
  their original real DEM.
- DEM path construction includes `split=self.split_name`, so accidental
  cross-split DEM swapping would fail loudly by missing file path.
- Eval-time topo metrics use real aligned DEM.

Primary audit doc:

- `docs/segman_dem_shuffle_scope_check.md`

Earlier audit doc:

- `docs/segman_seed0_topo_audit.md`

Note: the earlier audit initially flagged shuffle scope as needing confirmation;
the later scope-check resolved that question as safe.

## Current Multi-Seed Chain

Per the handoff, the SegMAN-S N=50 multi-seed chain has been launched and should
not be disturbed.

Seeds to run:

- `1`
- `2`
- `3`
- `42`

Variants per seed:

- CE
- Dice+CE
- Dice+CE+Topo real DEM
- Dice+CE+Topo DEM-shuffled

Expected new runs:

- 4 seeds x 4 variants = 16 runs

Seed0 is already complete and must not be rerun.

The chain is designed to run sequentially: one GPU run at a time, with the next
run starting only after a health check passes.

Launcher:

- `scripts/launch_segman_n50_multiseed_chain.ps1`

Log path for light status checks only when asked:

- `E:/flood_research/experiments/segman/multiseed_n50_chain.log`

Run root:

- `E:/flood_research/experiments/segman/runs/`

## Metrics Reporting Policy

Reports should include all available metrics, not only mIoU and water IoU.

For validation, test, and Bolivia/OOD, report whenever available:

- pixel accuracy
- mIoU
- IoU background
- IoU water
- F1 background
- F1 water
- macro F1
- precision background
- precision water
- recall background
- recall water
- confusion matrix raw counts
- TP water
- FP water
- TN background
- FN water
- support / pixels per class
- ignored pixels count
- predicted water pixels
- predicted water ratio
- ground-truth water pixels
- ground-truth water ratio
- topo violation fraction
- topo violation count if available

Training reports should include whenever available:

- best epoch
- total epochs
- early stopping reason
- loss_total
- loss_ce
- loss_dice
- loss_dice_ce
- loss_topo
- lambda_topo
- effective topo contribution
- `lambda_topo * loss_topo / loss_dice_ce`
- gradient norm
- learning rate
- NaN / Inf status
- OOM status
- checkpoint used for final evaluation

If a metric is not logged, do not invent it. Mark it as `not logged` or `not
available`. If it can be recomputed safely from existing predictions and masks
using CPU only, mention that explicitly.

## Next Actions

Immediate next safe action:

- Let the multi-seed chain finish.

After all 16 new runs are complete:

1. Aggregate seed0/1/2/3/42 results.
2. Compute mean +/- std per variant and split.
3. Determine whether real DEM beats shuffled on average, shuffled beats real DEM
   on average, or the effects are seed-unstable.
4. Only then decide whether a lambda sweep is justified.

Aggregation script:

- `experiments_cvpr/segman/aggregate_multiseed_results.py`

Expected aggregate outputs:

- `reports/segman_n50_multiseed_results.csv`
- `reports/segman_n50_multiseed_results.json`
- `docs/segman_n50_multiseed_summary.md`

The aggregator is intended to be CPU-only and should not load checkpoints or use
the GPU.

## Do Not Do Now

Do not:

- launch training
- use the GPU
- stop, kill, or modify any running process
- delete, move, or rename experiment directories
- push to GitHub
- merge the open SegMAN/CVPR PR
- run full tests
- rerun seed0
- launch TerraMind
- launch EoMT
- launch VFMNet/VFMSeg
- launch ViT-P
- launch full-set experiments
- launch lambda sweep
- start WSL/Linux/CUDA official SegMAN migration while the current chain is
  running
- run GPU notebooks
- repeatedly monitor logs

Manual chain status checks should be done only when explicitly requested.

## Important Files

Docs:

- `docs/segman_setup.md`
- `docs/SEGMAN_MIGRATION_NOTE.md`
- `docs/segman_seed0_topo_audit.md`
- `docs/segman_dem_shuffle_scope_check.md`
- `reports/segman_seed0_loss_comparison_summary.json`

SegMAN code:

- `experiments_cvpr/segman/train_segman.py`
- `experiments_cvpr/segman/model/segman_model.py`
- `experiments_cvpr/segman/segman_losses/segman_loss.py`
- `experiments_cvpr/segman/segman_kernels/`
- `experiments_cvpr/segman/smoke_tests.py`
- `experiments_cvpr/segman/debug_one_batch.py`
- `experiments_cvpr/segman/aggregate_multiseed_results.py`

Configs:

- `configs/segman/segman_ce.yaml`
- `configs/segman/segman_dice_ce.yaml`
- `configs/segman/segman_dice_ce_topo.yaml`
- `configs/segman/segman_dice_ce_topo_dem_shuffled.yaml`
- `configs/segman/multiseed_n50/`

Launchers:

- `scripts/launch_segman_seed0_chain.ps1`
- `scripts/launch_segman_n50_multiseed_chain.ps1`

Manifests and DEM shuffle maps:

- `manifests/terramind_baseline/low_data_multiseed/`
- `manifests/terramind_baseline/low_data_multiseed/dem_shuffle_map_n50_seed*.json`

External experiment storage:

- `E:/flood_research/experiments/segman/runs/`
- `E:/flood_research/experiments/segman/multiseed_n50_chain.log`

