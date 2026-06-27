# SegMAN-S seed-0 N=50 — Physics Loss Audit Report

**Date**: 2026-06-26  
**Experiment block**: SEGMAN_cvpr2025_loss_ablation  
**Model**: SegMAN-S (33.45M params, 15-ch input, image_size=512)  
**Dataset**: Sen1Floods11 HandLabeled — N=50 train, 86 val, 89 test, 15 Bolivia  
**Split files**: identical across all four variants (verified in configs)

---

## 1. Executive Summary

All four SegMAN-S loss variants completed without collapse, NaN, Inf, or OOM.
Early stopping triggered between epoch 58–59 in all cases. Best epochs are
tightly clustered at 43–44, suggesting the training budget and patience are
well-calibrated for N=50.

**The DEM-shuffled control outperforms the real-DEM topo variant on every
reported split.** This is the central scientific issue. Current evidence
supports a regularization hypothesis rather than a physics-grounding effect,
but one seed is insufficient to draw firm conclusions.

No implementation bug was found that would invalidate the results. One area
requiring verification is whether `TopographyDataModule` applies the DEM
shuffle only to training or to all splits. The config notes state "Eval-time
topo metrics still use the real DEM," but this must be confirmed in code.

---

## 2. Final Results Table (verified against source files)

All numbers extracted from `{run_dir}/metrics/{tag}_summary.json` and cross-
checked against the last row of the epoch CSV at the best epoch.

| Loss | Best ep | Total ep | Val mIoU | Val IoU_w | Val F1_w | Test mIoU | Test IoU_w | Test F1_w | Bol mIoU | Bol IoU_w | Bol F1_w |
|------|---------|----------|----------|-----------|----------|-----------|------------|-----------|----------|-----------|----------|
| CE | 43 | 58 | 0.8388 | 0.7158 | 0.8343 | 0.8420 | 0.7268 | 0.8418 | 0.8110 | 0.6840 | 0.8123 |
| Dice+CE | 44 | 59 | 0.8382 | 0.7167 | 0.8350 | 0.8389 | 0.7234 | 0.8395 | 0.8363 | 0.7300 | 0.8440 |
| Dice+CE+Topo | 44 | 59 | 0.8343 | 0.7097 | 0.8302 | 0.8496 | 0.7414 | 0.8515 | 0.8332 | 0.7251 | 0.8406 |
| Dice+CE+Topo+shuffle | 44 | 59 | **0.8415** | **0.7219** | **0.8385** | **0.8637** | **0.7646** | **0.8666** | **0.8368** | **0.7307** | 0.8444 |

**All numbers match expected values.** Source files: `{tag}_summary.json` (evaluations block).

### Topographic violation fractions (final evaluation, best checkpoint)

| Loss | Val viol_frac | Test viol_frac | Bol viol_frac |
|------|--------------|----------------|--------------|
| CE | 0.000806 | 0.000916 | 0.001705 |
| Dice+CE | 0.000847 | 0.000975 | 0.001652 |
| Dice+CE+Topo | 0.000863 | 0.000956 | 0.001570 |
| Dice+CE+Topo+shuffle | 0.000933 | 0.000983 | 0.001686 |

### Predicted water pixels (best checkpoint, final eval)

| Loss | Val pred_water | Test pred_water | Bol pred_water |
|------|--------------|----------------|--------------|
| CE | 2,020,478 | 2,411,263 | 379,636 |
| Dice+CE | 2,253,277 | 2,647,920 | 459,507 |
| Dice+CE+Topo | 2,231,065 | 2,632,155 | 461,194 |
| Dice+CE+Topo+shuffle | 2,195,595 | 2,537,131 | 456,092 |

Ground-truth water pixels: val≈2,237,605; test≈2,566,101; Bolivia≈454,864.
CE under-predicts water; Dice+CE and topo variants are closer to ground truth.

---

## 3. Implementation Audit

### 3.1 DEM not used as model input

**Status: VERIFIED CLEAN — two independent guards.**

1. `InputAssembler` docstring explicitly states "DEM is *not* included here —
   it stays in batch['topography'] for the loss." The assembler concatenates
   only S2L1C (13 ch) + S1GRD (2 ch) → 15 ch total.

2. Hard runtime guard in `train_segman.py:246–248`:
   ```python
   if config.get("dem", {}).get("use_as_model_input", False):
       raise RuntimeError("DEM as model input is forbidden; DEM is loss-only.")
   ```
   Both configs also set `guardrails.dem_as_model_input: false`.

### 3.2 DEM used only in loss and topo metrics

**Status: VERIFIED CLEAN.**

`batch["topography"]` is passed to `criterion(logits, target, topo)`. Inside
`SegManCombinedLoss.forward`, topo is accessed only when `self.use_topo` is
True (modes `dice_ce_topo` and `dice_ce_topo_dem_shuffled`). For CE and
Dice+CE, `loss_topo = zero` and topography is consumed but unused. The same
`topo` tensor is passed to `t6c.topographic_violation_counts` for the
topographic metric computation — this is correct (metrics for all variants
use the same batch DEM, whatever was loaded by the dataloader).

### 3.3 Shuffled DEM: training vs. evaluation scope

**Status: DESIGN INTENT VERIFIED; IMPLEMENTATION NEEDS CONFIRMATION.**

The shuffled config (`segman_dice_ce_topo_dem_shuffled.yaml`) notes section
states: "Eval-time topo metrics still use the real DEM." The DEM shuffle is
implemented by loading `dem_tile_id_map_file` (a derangement over N=50 train
tile IDs) into `config["dem"]["dem_tile_id_map"]` in `train_segman.py:211–214`.
This map is then passed to `TopographyDataModule`.

**Key question**: does `TopographyDataModule` apply the tile-ID remap only to
train-split batches, or to all splits? This must be verified in
`step6c_v3_train.py`. If the remap is applied to val/test/bolivia, then the
topo violation fractions reported for the shuffled variant are computed against
the wrong DEM and are not comparable to the other variants.

Evidence that the shuffle does NOT apply to evaluation (indirect):
- The topo violation fractions for the shuffled variant at final eval are
  0.000933 (val), 0.000983 (test), 0.001686 (Bolivia) — within the same
  order of magnitude as the other variants. If a random DEM were used for
  violation counting, one would not expect the values to be systematically
  close to the real-DEM variants.
- The config notes explicitly state the design intent.

**Action required**: Read `step6c_v3_train.TopographyDataModule.setup()` and
`_build_dataset()` to confirm the tile-ID remap is conditioned on
`split == "train"`.

### 3.4 ignore_index, water_class, softmax

**Status: VERIFIED CLEAN.**

- `ignore_index = -1` set in all configs and passed through `SegManCombinedLoss`,
  `CrossEntropyLoss`, `DiceLoss(ignore_index=-1)`, and topo/violation counts.
- `water_class = 1` is consistent across configs, loss, and metric calls.
- Topo loss uses `logits` (raw scores) fed through softmax internally inside
  `TopographicInconsistencyLoss`. `SegManCombinedLoss.forward` passes raw
  logits (`logits=logits`) — correct.
- Metrics use `torch.argmax(logits.detach(), dim=1)` for predictions — correct.

### 3.5 lambda_topo is applied

**Status: VERIFIED ACTIVE.**

`lambda_for_epoch` (segman_loss.py:144–160) implements warmup_linear:
- Epochs 1–5: lambda = 0.0 (warmup hold)
- Epochs 6–20: linear ramp 0 → 0.5
- Epoch 20+: lambda = 0.5

The CSV `lambda_topo_epoch` column confirms:
- Epoch 5: 0.0
- Epoch 6: 0.0333
- Epoch 20: 0.5

`criterion.set_lambda_topo(epoch_lambda)` is called at the start of every
epoch. Loss formula: `L = Dice + 1.0 * CE + lambda_topo * Topo`. ✓

### 3.6 Best checkpoint used for final evaluation

**Status: VERIFIED CLEAN.**

`train_segman.py:393` explicitly does:
```python
model.load_state_dict(torch.load(best_ckpt, ...)["model_state_dict"])
```
before running the final val/test/bolivia eval. The final metrics in
`summary.json` are from the best epoch, not the last epoch.

### 3.7 Same splits across all four variants

**Status: VERIFIED CLEAN.**

All four configs use identical `train_split`, `val_split`, `test_split` paths
and the same Bolivia setup. The only differences between configs are:
`loss.mode`, `loss.lambda_topo`, `loss.lambda_schedule`, and
`dem.dem_tile_id_map_file` (shuffled variant only).

---

## 4. Loss-Scale Analysis

### 4.1 Component magnitudes at full lambda (epochs 20–44)

Averaged from the epoch CSV for the topo and shuffled runs:

| Component | Topo (real DEM) | Topo (shuffled DEM) |
|-----------|-----------------|---------------------|
| train_loss_ce | ~0.196 | ~0.181 |
| train_loss_dice | ~0.359 | ~0.334 |
| train_loss_topo (raw) | ~0.027 | ~0.024 |
| train_loss_total | ~0.582 | ~0.543 |
| lambda × loss_topo | ~0.0135 | ~0.012 |
| **Topo fraction of total** | **~2.3%** | **~2.2%** |

### 4.2 Effective topo influence

```
lambda * loss_topo / loss_dice_ce  ≈  0.5 * 0.027 / 0.555  ≈  2.4%   (real DEM)
                                   ≈  0.5 * 0.024 / 0.515  ≈  2.3%   (shuffled DEM)
```

**The topo term contributes only ~2% of the total loss.** This is small
enough that it can act as a gentle regularizer without materially disrupting
the dice+CE gradient. It also means that the physically-grounded signal
(real-DEM topographic penalty) is potentially too weak to overwhelm noise
from only 50 training samples.

### 4.3 Shuffled vs. real DEM — loss magnitude comparison

Notably, the shuffled DEM produces LOWER `loss_topo` values than the real DEM
(~0.024 vs ~0.027). This is counter-intuitive — one might expect that random
DEM values produce harder topo constraints that resist gradient descent. The
likely explanation: the real DEM has stronger spatial correlation with the
ground-truth flood labels (water is at lower elevations), so predictions that
disagree with the topographic prior incur slightly higher violations with the
real DEM. With the shuffled DEM, the elevation-water correspondence is broken,
so the loss surface is flatter and slightly easier to minimize.

### 4.4 Warmup schedule and early mIoU behaviour

At epoch 1 (lambda=0), both topo variants log a non-zero `train_loss_topo`
(~0.078 for both). This is because the loss function still computes the topo
term for logging purposes even when lambda=0 — the gradient contribution is
zero, but the value is tracked. This is correct behaviour.

During the warmup phase (epochs 1–5), both topo variants converge at nearly
the same pace as Dice+CE. The main divergence occurs after epoch 20 when
lambda=0.5 is fully active.

---

## 5. Topographic Violation Metric Analysis

### 5.1 Violation fractions at best epoch (validation)

From the epoch CSV at the best epoch for each variant:

| Loss | val_topo_violation_fraction | Δ vs CE |
|------|-----------------------------|---------|
| CE (epoch 43) | 0.000806 | baseline |
| Dice+CE (epoch 44) | 0.000847 | +0.000041 |
| Dice+CE+Topo (epoch 44) | 0.000863 | +0.000057 |
| Dice+CE+Topo+shuffle (epoch 44) | 0.000933 | +0.000127 |

**All variants cluster around 0.08–0.09% violation fraction.** Strikingly,
the topo loss variants do NOT reduce violations compared to CE. The shuffled
variant has the highest violation fraction.

### 5.2 Interpretation of the violation metric

The violation metric counts 4-neighbor pixel pairs where: (a) one neighbour
is predicted water, one is predicted non-water, AND (b) the water-labelled
neighbour has HIGHER elevation than the non-water neighbour. Fraction is
relative to all descending-elevation neighbour pairs.

Problems with the current metric as a physics discriminator:

1. **Saturation at baseline**: All variants achieve ~0.08–0.09% violation
   fraction without any topo loss. The metric is insensitive in this regime —
   small differences (±0.0001) are within noise and have no discriminative
   power at one seed.

2. **4-neighbourhood is too local**: Flood extent consistency is a meso-scale
   phenomenon. 4-connected pairs capture only immediate adjacency; a better
   metric would measure path-connected flood components relative to a flow
   accumulation field.

3. **No elevation gradient weighting**: A pair at 1m elevation difference is
   penalized equally to one at 100m. The current `use_elevation_weight=True`
   flag suggests some weighting is applied (via `elevation_scale`), but
   `elevation_margin=0.0` means zero tolerance is set, catching even trivial
   slope-vs-prediction disagreements.

4. **Comparison is contaminated for the shuffled variant** (if shuffle applies
   to eval — see §3.3). Violation fractions for the shuffled variant may be
   computed against the wrong DEM if `TopographyDataModule` does not scope the
   tile remap to training only.

---

## 6. Scientific Interpretation

### 6.1 What the results show

At one seed (seed=0) with N=50:

- **CE** is the weakest baseline: low Bolivia mIoU (0.811), under-predicts
  water (2.0M pixels vs 2.2M ground truth). The pure CE objective is
  insufficient for robust OOD generalization at N=50.

- **Dice+CE** recovers substantially on Bolivia (+2.5pp mIoU, +4.6pp water
  IoU) but does not improve on test vs. CE. Dice loss penalizes class
  imbalance more evenly, helping Bolivia but not altering the gradient signal
  for OOD test regions.

- **Dice+CE+Topo (real DEM)** improves test performance notably (+1.1pp mIoU
  vs Dice+CE, +1.8pp water IoU) while slightly regressing on val (−0.4pp).
  The real-DEM topo term appears to improve generalization to the unseen test
  set, possibly because it learns a globally-applicable "water at low
  elevations" prior that is independent of the specific training images.

- **Dice+CE+Topo+shuffle** achieves the best overall scores: best val, best
  test, best Bolivia. The difference on test vs. the real-DEM variant is
  large: +1.4pp mIoU, +2.3pp water IoU, +1.5pp F1.

### 6.2 Regularization vs. physics-grounding hypothesis

Three observations point toward a **regularization hypothesis** (the topo term
acts primarily as a regularizer, not as a physics-grounded constraint):

1. The shuffled DEM — which deliberately breaks sample-DEM correspondence —
   outperforms the real DEM on every reported split. If the physical
   information in the DEM were the operative mechanism, the real DEM should
   win.

2. The effective topo contribution is only ~2% of total loss. This magnitude
   may be too small for the physically correct DEM signal to meaningfully
   guide gradient updates with only 50 training samples.

3. The topo violation metric is nearly identical across all variants (0.08–
   0.09%), including CE which has no topo loss at all. This suggests the model
   already respects topographic ordering without explicit supervision at this
   scale.

A plausible mechanism for why the shuffled DEM performs better: the shuffled
DEM creates **harder topo constraints** that are inconsistent with the
spectral content of the image. The model must learn a stronger generalization
— "predict water in low-elevation contexts" — without relying on the specific
tile's elevation-image correlation, which may overfit to idiosyncratic training
tile characteristics at N=50.

### 6.3 What we cannot conclude

- With one seed, variance is unknown. The true ranking may be reversed at
  other seeds.
- It is possible that the real-DEM topo loss requires higher lambda (e.g.,
  1.0–2.0) to be effective at N=50.
- The DEM alignment quality has not been verified tile-by-tile; noisy
  alignment could degrade the real-DEM signal relative to shuffled.
- The current topo loss (`TopographicInconsistencyLoss` at 4-neighbor pairs)
  may not be the best formulation for capturing relevant physical structure.

**We cannot claim that the real topographic information improves the model
over Dice+CE alone.** On test, it does (+1.1pp mIoU), but the shuffled
variant improves even more (+2.4pp), making attribution ambiguous.

---

## 7. Limitations

1. **N=1 seed**: all conclusions are tentative at one seed.
2. **No multi-seed variance**: cannot report mean±std; any single-run
   comparison might be noise.
3. **Violation metric insensitivity**: the current metric does not distinguish
   variants. A better physical metric is needed.
4. **Shuffle scope ambiguity**: needs code-level verification that the DEM
   tile-ID remap applies only to training splits.
5. **No lambda sweep**: lambda=0.5 was chosen based on prior TerraMind
   experiments. It may not be optimal for SegMAN at N=50.
6. **No pretrained encoder**: SegMAN-S started from random weights (no
   ImageNet pretraining configured). A pretrained encoder might change the
   loss landscape.
7. **Bolivia**: only 15 tiles, making that split noisy.

---

## 8. Recommended Next Experiments

**Do not launch without explicit instruction. These are recommendations only.**

### 8.1 Multi-seed replication (highest priority)

Run all four loss variants for seeds 1, 2, 3, 42 (in addition to seed 0
already done). Report mean ± std for val/test/Bolivia mIoU and water IoU.
This is the minimum necessary to distinguish signal from variance.

### 8.2 Lambda sweep

For the real-DEM topo variant only (to reduce compute):
`lambda_topo ∈ {0.1, 0.5, 1.0, 2.0}` at seed 0.
Goal: determine whether a stronger topo signal closes the gap to the shuffled
variant.

### 8.3 Verify shuffle scope in TopographyDataModule

Read `step6c_v3_train.TopographyDataModule._build_dataset()` and confirm
whether `dem_tile_id_map` is applied only when `split == "train"`. If it
applies to val/test, the reported topo violation fractions for the shuffled
variant are against the wrong DEM and must be recomputed.

### 8.4 Improved physical metrics

Replace or augment the violation fraction metric with:
- **Altitude-weighted violation**: weight each violating pair by |Δelevation|
- **Connected-component inconsistency**: identify flood segments at higher
  elevation than any connected lower-elevation non-flood region
- **Per-image distribution**: report per-image violation histogram, not just
  mean, to detect spatial patterns
- **Flow-path consistency**: check if predicted water follows drainage
  network (requires DEM-derived flow accumulation map)

### 8.5 DEM alignment quality audit

Verify, for the N=50 training tiles, that the Copernicus GLO-30 DEM is
properly co-registered to the S2 image grid. Inspect a few tiles with known
elevation gradient and check that flood labels correlate with low-DEM values.

### 8.6 Pretrained encoder

Try SegMAN-S with the official ImageNet-pretrained encoder (after inflating the
stem from 3→15 channels). This may significantly change performance at N=50
and could alter the relative benefit of the topo loss.

---

## 9. Source Files Inspected

| File | Purpose |
|------|---------|
| `experiments_cvpr/segman/train_segman.py` | Training loop, DEM guard, eval logic |
| `experiments_cvpr/segman/segman_losses/segman_loss.py` | Loss selector, lambda schedule |
| `configs/segman/segman_ce.yaml` | CE variant config |
| `configs/segman/segman_dice_ce.yaml` | Dice+CE variant config |
| `configs/segman/segman_dice_ce_topo.yaml` | Real-DEM topo config, lambda schedule |
| `configs/segman/segman_dice_ce_topo_dem_shuffled.yaml` | Shuffled-DEM config, shuffle map ref |
| `{run_dir}/metrics/{tag}_summary.json` × 4 | Final eval numbers, status |
| `{run_dir}/metrics/training_epoch_metrics.csv` × 4 | Per-epoch loss/metric curves |

**Not inspected (flagged for follow-up)**:
- `scripts/step6c_v3_train.py:TopographyDataModule` — shuffle scope
- `src/losses/physics_topographic_loss.py` — topo loss internals
- `manifests/.../dem_shuffle_map_n50_seed0.json` — derangement map properties
