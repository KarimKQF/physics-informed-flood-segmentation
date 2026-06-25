# E1+E2 Multi-seed N=50 — Final Report

**Generated:** 2026-06-24  
**Model:** TerraMind-L + UPerNet (backbone `terramind_v1_large`, feature indices [5,11,17,23])  
**Dataset:** Sen1Floods11 HandLabeled — train N=50, val 86, test 89, Bolivia OOD 15  
**Loss:** `smp.DiceLoss + λ(epoch) * TopographicInconsistencyLoss`  
**Lambda schedule (E1/E2 main block):** warmup_linear — λ=0 epochs 1–5, linear ramp epochs 6–19, λ=0.5 epoch 20+  
**DEM:** Copernicus GLO-30 DSM — never model input, loaded as `batch["topography"]` only

---

## 1. Collapse Rate Summary

| Condition | Collapsed | Rescued | Collapse rate |
|---|---|---|---|
| Dice-only | 5 / 5 | 0 / 5 | **100%** |
| Physics real DEM (λ=0.5, warmup) | 2 / 5 | 3 / 5 | **40%** |
| Physics shuffled DEM (λ=0.5, warmup) | 3 / 5 | 2 / 5 | **60%** |

**Collapse definition:** val_water_iou < 0.01 OR val_pred_water_px < 5000 OR pred_water_frac > 0.90 OR val_miou < 0.10

---

## 2. Results by Seed

| Seed | Dice-only | Physics real DEM | Physics shuffled DEM | Note |
|---|---|---|---|---|
| **0** | collapsed (all-bg) | **rescued** — water_iou=0.700 | collapsed (all-water) — miou=0.055, frac=0.9999 | Only seed where real vs shuffled differs |
| **1** | collapsed (all-bg) | collapsed (all-bg) | collapsed (all-bg) | Collapses all conditions |
| **2** | collapsed (all-water) | **rescued** — water_iou=0.720 | **rescued** — water_iou=0.717 | Rescued by both real and shuffled |
| **3** | collapsed (all-bg) | collapsed (all-bg) | collapsed (all-bg) | Collapses all conditions |
| **42** | collapsed (all-bg) | **rescued** — water_iou=0.691 | **rescued** — water_iou=0.685 | Rescued by both real and shuffled |

### Full metrics table

| Seed | Condition | best_ep | stop_ep | val_mIoU | val_water_IoU | val_water_F1 | val_pred_px | pred_frac | topo_viol | collapsed |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | dice_only | 1 | 30 | 0.4449 | 0.0000 | — | 0 | 0.000 | 0.0 | YES (all-bg) |
| 1 | dice_only | 1 | 30 | 0.4449 | 0.0000 | — | 0 | 0.000 | 0.0 | YES (all-bg) |
| 2 | dice_only | 1 | 30 | 0.0551 | 0.1103 | 0.199 | 20 294 725 | 0.9999 | 0.0 | YES (all-water) |
| 3 | dice_only | 1 | 30 | 0.4449 | 0.0000 | — | 0 | 0.000 | 0.0 | YES (all-bg) |
| 42 | dice_only | 1 | 30 | 0.4449 | 0.0000 | 0.000 | 212 | 0.000 | 0.0 | YES (all-bg) |
| 0 | physics_real | 38 | — | 0.8276 | 0.6999 | 0.823 | 2 403 824 | — | 0.00111 | NO |
| 1 | physics_real | 1 | 30 | 0.4449 | 0.0000 | — | 0 | 0.000 | 0.0 | YES (all-bg) |
| 2 | physics_real | 52 | — | 0.8399 | 0.7197 | 0.837 | 2 272 237 | — | 0.000992 | NO |
| 3 | physics_real | 1 | 30 | 0.4449 | 0.0000 | — | 0 | 0.000 | 0.0 | YES (all-bg) |
| 42 | physics_real | 55 | 80 | 0.8243 | 0.6914 | 0.818 | 2 116 688 | — | 0.000836 | NO |
| 0 | shuffled | 36 | 51 | 0.0551 | 0.1103 | 0.199 | 20 294 283 | 0.9999 | 4.5e-6 | YES (all-water) |
| 1 | shuffled | 1 | 30 | 0.4449 | 0.0000 | — | 0 | 0.000 | 0.0 | YES (all-bg) |
| 2 | shuffled | 62 | 77 | 0.8374 | 0.7166 | 0.835 | 2 407 591 | 0.119 | 0.00107 | NO |
| 3 | shuffled | 1 | 30 | 0.4449 | 0.0000 | — | 0 | 0.000 | 0.0 | YES (all-bg) |
| 42 | shuffled | 41 | 56 | 0.8206 | 0.6853 | 0.813 | 2 131 469 | 0.105 | 0.000766 | NO |

---

## 3. Conclusion

**Dice-only N=50 is systematically unstable.** 5/5 seeds collapse, 4 to all-background and 1 to all-water, confirming that Dice loss alone cannot reliably learn flood segmentation at N=50.

**Physics real DEM reduces collapse from 5/5 to 2/5.** Seeds 0, 2, and 42 are rescued with val_water_IoU in the range 0.69–0.72, demonstrating that the topographic consistency loss substantially improves training stability in the low-data regime.

**Shuffled DEM also reduces collapse, from 5/5 to 3/5.** Seeds 2 and 42 are rescued by shuffled DEM with val_water_IoU comparable to real DEM (seed2: 0.717 vs 0.720; seed42: 0.685 vs 0.691). This shows that a significant component of the rescue effect comes from the regularization mechanism and schedule structure rather than from real topographic content per se.

**The specific role of true topographic correspondence is supported by seed 0 only.** Seed 0 is rescued by real DEM (water_IoU=0.700) but collapses all-water with shuffled DEM (miou=0.055). This is the single discriminating case; the evidence is not statistically robust with N=1 discriminating seed.

**Seeds 1 and 3 collapse under all conditions.** Their instability is independent of the physics loss variant and appears to be driven by the random initialization. This motivates the no-warmup ablation: applying λ=0.5 from epoch 1 may provide stronger gradient signal early enough to prevent these seeds from locking into a degenerate attractor.

---

## 4. Pending Ablation

**Physics real DEM, no-warmup (λ=0.5 from epoch 1):**
- `n50_seed1_physics_real_dem_lambda05_no_warmup`
- `n50_seed3_physics_real_dem_lambda05_no_warmup`

Scientific question: does removing the warmup phase (which zeros the physics gradient for epochs 1–5) allow the topographic constraint to act earlier and prevent the all-background collapse observed in seeds 1 and 3?
