# STEP 6C v3 Pure-Dice Parity With STEP 5S-A Dataloader

Generated: 2026-06-22

## Scope

This task built a STEP 6C/v3 runner that preserves STEP 5S-A dataloader-side Albumentations D4 behavior while adding aligned DEM as a loss-only batch tensor.

No full STEP 6C training was launched. No DARN or STURM-Flood training was started. Raw data was not modified. Previous runs, logs, reports, metrics, and checkpoints were preserved.

## Files

- Runner: `scripts/step6c_v3_train.py`
- Pure-Dice config: `configs/step6c_v3_pure_dice_parity_5sa_dataloader.yaml`
- Pure-Dice run directory: `E:/flood_research/experiments/terramind_baseline/runs/step6c_v3_pure_dice_parity_5sa_dataloader/`
- Metrics JSON: `E:/flood_research/experiments/terramind_baseline/runs/step6c_v3_pure_dice_parity_5sa_dataloader/metrics/pure_dice_parity_metrics.json`
- Prepared physics config: `configs/step6c_v3_terramind_l_upernet_dice_topographic_lambda05_warmup_5sa_dataloader.yaml`

## Implementation

The v3 runner starts from the successful STEP 5S-A data path:

- D4 is applied by Albumentations in the dataset/dataloader path.
- D4 is not applied inside the training loop.
- Aligned DEM is loaded at sample level before transform.
- DEM is included as `topography` in the Albumentations transform targets.
- DEM is returned as `batch["topography"]`.
- DEM is never inserted into `batch["image"]` and is never passed to the model.
- `CombinedDicePhysicsLoss(lambda_topo=0)` uses exact `smp.losses.DiceLoss` parity with TerraTorch/STEP 5S-A.

An initial attempt failed during validation-only physical metric computation because the violation counter expected `[B,H,W]` DEM while v3 returned `[B,1,H,W]`. The failed attempt log/JSON/script/config were preserved with the suffix `attempt1_failed_metric_shape_20260622T2309`; no checkpoint had been written. The metric code was fixed by squeezing DEM only for the violation counter, then the required run was repeated.

## Mini-Run Setup

- Epochs: 3
- Initialization: original TerraMind pretrained checkpoint
- Backbone: `terramind_v1_large`
- Decoder: `UperNetDecoder`
- Feature indices: `[5, 11, 17, 23]`
- Loss: Dice
- `lambda_topo`: 0
- Batch size: 2
- Gradient accumulation steps: 4
- Effective batch size: 8
- Precision: FP32
- Train split: same filtered Sen1Floods11 train split as STEP 5S-A
- Valid split: same filtered Sen1Floods11 valid split as STEP 5S-A

## Results

| Epoch | train Dice loss | val mIoU | val IoU water | val F1 water | val water predicted pixels | max grad norm |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.389159 | 0.753068 | 0.560843 | 0.718641 | 1,450,761 | 25.865109 |
| 2 | 0.299600 | 0.799171 | 0.651150 | 0.788723 | 2,379,860 | 14.277511 |
| 3 | 0.260938 | 0.794508 | 0.645360 | 0.784461 | 2,574,206 | 6.781690 |

Best validation mIoU: `0.799171` at epoch 2.

## Acceptance Criteria

| Criterion | Result |
|---|---|
| No all-background collapse | PASS |
| Validation predicted water pixels > 0 by epoch 1 or 2 | PASS |
| `val_iou_water` does not remain exactly 0 | PASS |
| Gradients remain nonzero | PASS |
| Behavior resembles STEP 5S-A early training | PASS |

## Decision

Pure-Dice parity passed. The collapse is no longer reproduced when STEP 6C/v3 uses STEP 5S-A-style dataloader-side Albumentations D4 and returns DEM only for loss computation.

Because the mini-run passed, the corrected physics warmup config was prepared:

`configs/step6c_v3_terramind_l_upernet_dice_topographic_lambda05_warmup_5sa_dataloader.yaml`

Full STEP 6C is technically safe to launch from the prepared v3 warmup config, but only after explicit human instruction. Human validation is required before launching.
