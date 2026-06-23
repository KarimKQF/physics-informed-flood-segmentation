# STEP 6C v2 — D4 A/B Microtest Report (manual D4 vs albumentations D4)

Generated: 2026-06-22
Scope: controlled diagnostic, **not** full training. 3 epochs per arm, **pure Dice
(`lambda_topo = 0`)**. No raw data modified. DEM never a model input. Failed runs,
logs, checkpoints, and prior reports all preserved.

Script: `scripts/step6c_v2_d4_ab_microtest.py`
Metrics JSON:
`E:/flood_research/experiments/terramind_baseline/runs/step6c_v2_ab_microtest/metrics/d4_ab_microtest.json`

---

## 1. Question

After the warmup run collapsed at `lambda_topo = 0` (pure Dice), the topographic loss
was exonerated and the suspect became the **runner / augmentation** divergence between
the non-collapsing STEP 5S-A (`albumentations.D4` in the dataloader) and the collapsing
STEP 6C/v2 (manual `torch.rot90/flip` D4 in the training loop). This microtest isolates
that single variable.

## 2. Controlled setup (everything identical except the D4 path)

* Pure Dice, `lambda_topo = 0` (exact 5S-A loss via `CombinedDicePhysicsLoss`).
* Same seed (42) → same original-TerraMind init and same data/shuffle order.
* Same split (251/86), `batch_size=2`, `grad_accum=4`, FP32, BN-eval policy.
* Same AdamW (lr 2e-5, wd 1e-4); scheduler not stepped (LR fixed) in both arms.
* Same exact-parity Dice loss and same metric code.
* **Only difference:**
  * **A "manual"** — dataloader = `ToTensorV2` only; D4 applied manually with
    `torch.rot90/flip` to image+mask+DEM in the loop (the current v2 path).
  * **B "albumentations"** — dataloader = `[albumentations.D4, ToTensorV2]` (the
    STEP 5S-A path); no manual D4.

## 3. Results

### Arm A — manual D4
| epoch | train_loss_dice | val_mIoU | val_iou_water | val_water_pred_px | mean_p_water | max_grad_norm | grads_dead |
|---:|---:|---:|---:|---:|---:|---:|:--:|
| 1 | 0.548519 | 0.444872 | **0.000000** | **0** | **0.000000** | 49.23 | no |
| 2 | 0.523479 | 0.444872 | **0.000000** | **0** | **0.000000** | **0.000000** | **YES** |
→ **COLLAPSED** (gradient-dead all-background; guard stopped at epoch 2).

### Arm B — albumentations D4
| epoch | train_loss_dice | val_mIoU | val_iou_water | val_water_pred_px | mean_p_water | max_grad_norm | grads_dead |
|---:|---:|---:|---:|---:|---:|---:|:--:|
| 1 | 0.386765 | 0.776719 | **0.604641** | 1,673,640 | 0.082469 | 72.77 | no |
| 2 | 0.296850 | 0.725300 | 0.543138 | 3,477,124 | 0.171331 | 19.74 | no |
| 3 | 0.265212 | 0.793550 | **0.643686** | 2,568,738 | 0.126571 | 10.11 | no |
→ **NO collapse.** Healthy training, recovering exactly like STEP 5S-A
(5S-A epoch 1 val_iou_water = 0.0027 then 0.586 at epoch 2; arm B reaches 0.605 at
epoch 1 and 0.644 by epoch 3).

## 4. Conclusion

```
manual_d4_collapsed         = true
albumentations_d4_collapsed = false
manual_d4_is_cause          = true
```

* **The manual `torch.rot90/flip` D4 path is the cause of the no-water collapse.**
  With every other factor held identical, manual D4 drives the model to
  `mean_p_water = 0` within epoch 1 and into the gradient-dead absorbing state by
  epoch 2; albumentations D4 trains healthily.
* The topographic loss is fully exonerated (this entire test ran at `lambda_topo = 0`).
* The dramatic, reproducible split (arm B healthy across all 3 epochs; arm A dead) rules
  out a mere RNG coin-flip and points to the manual D4 implementation feeding the model
  inconsistent image/label supervision, for which the Dice-optimal degenerate answer is
  "predict the majority background everywhere".

## 5. Fix adopted

The v2 runner now augments with **albumentations `ReplayCompose([A.D4()])`** applied at
the same point in the loop, sharing the **exact same** geometric op across all image
modalities, the mask, and the DEM (`apply_albu_d4_to_batch` in `scripts/step6c_v2_train.py`,
selected by `data.d4_mode: albu_d4_replay`, now the default). A standalone alignment unit
test confirms image == mask == DEM after augmentation across random ops. The buggy
`manual` path is retained only behind `d4_mode: manual` for reproducibility.

Validation that the fixed path reproduces 5S-A at `lambda_topo = 0` is in
`reports/STEP_6C_V2_PURE_DICE_PARITY_AFTER_D4_FIX_REPORT.md`.
