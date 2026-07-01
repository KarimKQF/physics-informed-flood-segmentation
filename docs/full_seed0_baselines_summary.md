# Full Sen1Floods11 SegMAN-S Seed0 Baselines — Summary
## Status: COMPLETE

**Started:** 2026-06-30 22:48:34  
**Completed:** 2026-07-01 12:39:33  
**Total wall-clock:** ~13h51m  
**Runs:** CE → Dice → Dice+CE (sequential, one GPU — RTX 5000 Ada 32 GB)  
**Master log:** `E:\flood_research\logs\full_sen1floods11_baselines_master_stdout.log`  
**Status JSON:** `E:\flood_research\experiments\segman\full_baselines_chain_status.json`

---

## Configuration

| Parameter | Value |
|---|---|
| Architecture | SegMAN-S |
| Parameters | 33,447,272 (33.45M) |
| Input channels | 15 (13 S2L1C + 2 S1GRD) |
| DEM as model input | NEVER |
| Dataset | Sen1Floods11 full (N=251 train / 86 val / 89 test / 15 Bolivia OOD) |
| Seed | 0 |
| Batch size | 2 (eff. 8 with grad accum ×4) |
| Optimizer | AdamW lr=6e-5 wd=0.01 |
| Scheduler | ReduceLROnPlateau factor=0.5 patience=5 |
| Max epochs | 100 |
| Early stop | patience=20 (min 30 epochs) |
| Monitor | val mIoU |

---

## Run Status

| Run | Loss | Best Epoch | Last Epoch | Best val mIoU | Wall-clock |
|---|---|---|---|---|---|
| full_seed0_ce | CE | 24 | 44 | 0.8651 | 4.20h |
| full_seed0_dice | Dice (ce_alpha=0) | 20 | 40 | 0.8565 | 3.68h |
| **full_seed0_dice_ce** | **Dice+CE** | **43** | **63** | **0.8666** | **5.89h** |

---

## Validation Metrics (best checkpoint)

| Run | mIoU | Water IoU | F1-water | Precision | Recall | Topo-VF |
|---|---|---|---|---|---|---|
| CE | 0.8651 | 0.7618 | 0.8648 | 0.8977 | 0.8342 | 0.000962 |
| Dice | 0.8565 | 0.7489 | 0.8564 | 0.8352 | 0.8788 | 0.001259 |
| **Dice+CE** | **0.8666** | **0.7654** | **0.8671** | **0.8721** | **0.8622** | 0.001161 |

## Test Metrics (best checkpoint)

| Run | mIoU | Water IoU | F1-water | Precision | Recall | Topo-VF |
|---|---|---|---|---|---|---|
| CE | 0.8678 | 0.7710 | 0.8707 | 0.8928 | 0.8496 | 0.000971 |
| Dice | 0.8615 | 0.7618 | 0.8648 | 0.8470 | 0.8834 | 0.001280 |
| **Dice+CE** | **0.8797** | **0.7920** | **0.8839** | **0.8835** | **0.8844** | 0.001185 |

## Bolivia OOD Metrics (best checkpoint)

| Run | mIoU | Water IoU | F1-water | Precision | Recall | Topo-VF |
|---|---|---|---|---|---|---|
| CE | 0.7963 | 0.6574 | 0.7933 | 0.9205 | 0.6969 | 0.001616 |
| **Dice** | **0.8551** | **0.7604** | **0.8639** | **0.8624** | **0.8653** | 0.001650 |
| Dice+CE | 0.8243 | 0.7050 | 0.8270 | 0.9229 | 0.7491 | 0.001850 |

---

## Key Observations

### 1. Dice+CE is the primary baseline
Best val mIoU (0.8666) and clearly best test mIoU (0.8797, +0.012 vs Dice).
Best precision/recall balance on test (P=0.884, R=0.884 — nearly equal).
**Recommended primary baseline for all physics-loss comparisons.**

### 2. CE over-suppresses on Bolivia (OOD)
CE achieves very high precision (0.921) but poor recall (0.697) on Bolivia →
the model becomes overly conservative at inference on out-of-distribution scenes.
Bolivia mIoU 0.796 is the worst of the three. This mirrors the N=100 pattern.

### 3. Dice is the best OOD generalizer
Dice-only gives the best Bolivia mIoU (0.855) with balanced P/R (0.862/0.865).
This suggests that CE regularization may overfit to the label distribution and
hurt generalization on geographically distinct OOD flood events (Bolivia).

### 4. Full-dataset vs N=100 comparison
N=100 Dice+CE baseline (seed0): val mIoU=0.855, test mIoU=0.862, Bolivia=0.843.
Full-dataset Dice+CE (seed0):   val mIoU=0.867, test mIoU=0.880, Bolivia=0.824.
Full-dataset gains +0.012 val, +0.018 test — but loses −0.019 Bolivia mIoU.
The full-dataset model is more precise but less conservative on OOD scenes.

### 5. Topographic violation fractions (no physics loss)
All three baselines show VF ≪ 0.002 at best checkpoint:
CE: 0.000962–0.001616, Dice: 0.001259–0.001650, Dice+CE: 0.001161–0.001850.
This confirms the N=100 headroom finding: full-dataset models also already
satisfy topographic-order constraints at least as well as reference labels.

---

## GPU and Timing

| Run | Wall-clock | Best epoch | Last epoch | Notes |
|---|---|---|---|---|
| CE | 4.20h | 24 | 44 | Early stop patience=20, min_ep=30 |
| Dice | 3.68h | 20 | 40 | Fastest convergence |
| Dice+CE | 5.89h | 43 | 63 | Slowest, best final performance |

---

## Issues Encountered

- **Dice-only mode:** No standalone `dice` mode in segman_loss.py.
  Used `dice_ce` with `ce_alpha=0.0` → L = Dice + 0·CE = Dice only. No code change.
- **Launcher path bug (fixed before first run):** Initial path used `experiments_cvpr/train_segman.py`;
  corrected to `experiments_cvpr/segman/train_segman.py`.

---

## Recommendation

**Primary baseline for physics-loss study: `full_seed0_dice_ce`**
- Best val mIoU: 0.8666 (epoch 43)
- Best test mIoU: 0.8797
- Best precision/recall balance on test
- Checkpoint: `E:\flood_research\experiments\segman\runs\full_seed0_dice_ce\checkpoints\best_checkpoint.pt`

**Secondary baseline for OOD analysis: `full_seed0_dice`**
- Best Bolivia mIoU: 0.8551 (vs 0.8243 for Dice+CE)
- Consider for OOD generalization experiments

---

## Next Steps

1. Apply AUC / VF headroom / conditional redundancy screen to `full_seed0_dice_ce`
   checkpoint (mirrors N=100 diagnostic, but now on full dataset).
2. Multi-seed confirmation (seeds 1, 2, 3, 42) for Dice+CE baseline.
3. D8 physics-loss variants on full dataset (real vs shuffled DEM, λ sweep).
4. EoMT architecture integration and same diagnostic-first screen.
