# Conditional DEM Redundancy Test

**Inference-only. SegMAN weights frozen. DEM never a model input — used only as a
post-hoc feature after inference.**

## Question
Does the DEM add predictive information about the water label `y` once the trained
model probability `p_water` is known?  `I(y ; z_DEM | p_water) ?= 0`.

## Method
- Checkpoint: `E:\flood_research\experiments\segman\runs\segman_n100_dice_ce_seed0\checkpoints\best_checkpoint.pt` (best_ep=34, best_miou=0.8546)
- Predictor: **HistGradientBoostingClassifier(max_iter=300)**, fit on `train` (1981892 px, prevalence 0.090), eval on held-out splits.
- DEM features: rel_elev, slope, tpi5, tpi15, tpi31, curvature (computed without label leakage).
- Sampling: random (natural prevalence), cap_train=20000, cap_eval=20000/tile.
- Models:  A: `y~p_water`  ·  B: `y~p_water+DEM`  ·  C: `y~DEM`  ·  D: `y~p_water+shuffle(DEM)`.
- Bootstrap: by tile, n=500, 5000 px/tile, 95% CI.

## Results

| split | model | AUC | AP | Brier | LogLoss | n_pix | prev |
|-------|-------|----:|---:|------:|--------:|------:|-----:|
| val | A | 0.9840 | 0.9275 | 0.0244 | 0.0878 | 1720000 | 0.114 |
| val | B | 0.9861 | 0.9322 | 0.0242 | 0.0854 | 1720000 | 0.114 |
| val | C | 0.8366 | 0.5223 | 0.0704 | 0.2558 | 1720000 | 0.114 |
| val | D | 0.9840 | 0.9302 | 0.0244 | 0.0879 | 1720000 | 0.114 |
| test | A | 0.9848 | 0.9162 | 0.0297 | 0.1042 | 1780000 | 0.135 |
| test | B | 0.9840 | 0.9128 | 0.0319 | 0.1121 | 1780000 | 0.135 |
| test | C | 0.8336 | 0.5789 | 0.0771 | 0.2849 | 1780000 | 0.135 |
| test | D | 0.9848 | 0.9173 | 0.0297 | 0.1044 | 1780000 | 0.135 |
| bolivia | A | 0.9817 | 0.9254 | 0.0370 | 0.1268 | 300000 | 0.170 |
| bolivia | B | 0.9822 | 0.9260 | 0.0373 | 0.1260 | 300000 | 0.170 |
| bolivia | C | 0.7660 | 0.3646 | 0.1305 | 0.4116 | 300000 | 0.170 |
| bolivia | D | 0.9817 | 0.9287 | 0.0370 | 0.1269 | 300000 | 0.170 |

### Main comparison  B − A  (model+DEM vs model)

| split | dAUC | 95% CI | dAP | 95% CI | dBrier | dLogLoss |
|-------|-----:|--------|----:|--------|-------:|---------:|
| val | 0.0021 | [0.0008, 0.0043] | 0.0047 | [0.0002, 0.0122] | -0.0001 | -0.0025 |
| test | -0.0009 | [-0.0038, 0.0016] | -0.0034 | [-0.0201, 0.0060] | 0.0022 | 0.0079 |
| bolivia | 0.0004 | [-0.0014, 0.0029] | 0.0005 | [-0.0066, 0.0121] | 0.0003 | -0.0008 |

### Anti-ceiling check — B vs A on model-uncertain pixels only
Restricted to pixels where the model is unsure (lo < p_water < hi): if DEM helps anywhere it
is here. dAUC ≈ 0 here rules out 'redundancy is just a ceiling artifact'.

| split | n_pix | prev | A AUC | B AUC | dAUC | C(DEM-only) AUC |
|-------|------:|-----:|------:|------:|-----:|----------------:|
| val | 341849 | 0.221 | 0.8799 | 0.8912 | 0.0113 | 0.6884 |
| test | 347037 | 0.295 | 0.8820 | 0.8614 | -0.0205 | 0.6521 |
| bolivia | 86476 | 0.254 | 0.8839 | 0.8884 | 0.0045 | 0.6231 |

### Positive control (weak spectral predictor, NDWI)
*weak spectral predictor NDWI=(green-NIR)/(green+NIR) in place of p_water; DEM should add conditional info (dAUC>0), showing the test detects headroom and that the trained model has absorbed this DEM-complementary signal*. dAUC should be **positive** if the test can detect headroom.

| split | NDWI AUC (A_pos) | NDWI+DEM AUC (B_pos) | dAUC (pos) | dAP (pos) |
|-------|-----------------:|---------------------:|-----------:|----------:|
| val | 0.9796 | 0.9823 | 0.0027 | -0.0017 |
| test | 0.9807 | 0.9829 | 0.0023 | 0.0012 |
| bolivia | 0.9833 | 0.9852 | 0.0019 | 0.0027 |

### Permutation feature importance — model B (Δ ROC-AUC)

| feature | importance |
|---------|-----------:|
| p_water | 0.3881 |
| rel_elev | 0.0020 |
| tpi15 | 0.0014 |
| tpi31 | 0.0010 |
| slope | 0.0009 |
| curvature | 0.0002 |
| tpi5 | 0.0001 |

## Interpretation
- If **dAUC ≈ 0** and **dAP ≈ 0** (CIs spanning 0) while **C (DEM-only)** is non-trivial,
  the DEM is *conditionally redundant* given the trained model: it carries marginal water
  information, but none beyond `p_water`. No DEM loss — regardless of formulation or
  complexity — has residual label-exploitable signal once the image-only model has converged.
- The D8 null is therefore not a formulation failure but an informational property of the
  model–data pair.

## Limitation (must be stated)
This test is **relative to the available reference labels**. It does NOT prove the DEM
carries no information about the *true* physical flood state. If labels are noisy or
topographically inconsistent, the DEM could still help relative to unobserved truth.
The claim is only about improving agreement with the labels used for evaluation.

## Positive-control status
- Synthetic power check: **run** (weak spectral predictor NDWI=(green-NIR)/(green+NIR) in place of p_water; DEM should add conditional info (dAUC>0), showing the test detects headroom and that the trained model has absorbed this DEM-complementary signal).
- Real positive control: **MISSING**.
- Checkpoints present: best_checkpoint.pt, last_checkpoint.pt.
- TODO: Definitive positive control NOT run: no early/undertrained or S1-only checkpoint exists. Next: train an S1-only baseline (drop S2) OR save an early-epoch checkpoint, then re-run this test on it; expect dAUC>0 if conditional DEM headroom exists.

---
*Generated by `experiments_cvpr/segman/tools/conditional_dem_redundancy_test.py`.*
