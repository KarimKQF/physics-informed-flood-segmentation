# SegMAN-S N=100 D8 Seed0 — Post-hoc Diagnostic Report

**Date**: 2026-06-29  |  **Branch**: `experiments/segman-cvpr2025`  |  **Commit**: `8209243`
**Runs**: Dice+CE baseline, D8 real DEM, D8 shuffled DEM (all seed0, N=100)
**No training was launched. All data from existing epoch CSV and summary JSON files.**

## Part C — Matched-Epoch Analysis (Early-Stopping Confound)

Val mIoU at same epoch for all three runs:

| Epoch | Baseline | D8 real | D8 shuffled | Real−Shuffled | Real−Baseline | Shuf−Baseline |
|-------|----------|---------|-------------|---------------|---------------|---------------|
| ep22 ← shuf best | 0.76612 | 0.82679 | 0.84013 | -0.01334 | +0.06067 | +0.07400 |
| ep31 ← real best | 0.84792 | 0.85860 | 0.83317 | +0.02543 | +0.01068 | -0.01475 |
| ep34 ← base best | 0.85456 | 0.85431 | 0.82913 | +0.02518 | -0.00025 | -0.02543 |

Val precision / recall / predicted-water-ratio at matched epochs:

| Epoch | Run | Precision | Recall | Pred water % | Topo VF |
|-------|-----|-----------|--------|--------------|---------|
| ep22 | Baseline | 0.66131 | 0.87113 | 14.52% | 0.001181 |
| ep22 | D8 real | 0.81562 | 0.82764 | 11.19% | 0.001065 |
| ep22 | D8 shuf | 0.88692 | 0.79007 | 9.82% | 0.000648 |
| ep31 | Baseline | 0.87993 | 0.81313 | 10.19% | 0.000874 |
| ep31 | D8 real | 0.90596 | 0.81335 | 9.90% | 0.000775 |
| ep31 | D8 shuf | 0.79978 | 0.86258 | 11.89% | 0.000913 |
| ep34 | Baseline | 0.85043 | 0.85716 | 11.11% | 0.001122 |
| ep34 | D8 real | 0.87605 | 0.83082 | 10.46% | 0.000891 |
| ep34 | D8 shuf | 0.77886 | 0.87909 | 12.44% | 0.001021 |

## Part D — D8 Activation Diagnostic

| Metric | D8 real | D8 shuffled |
|--------|---------|-------------|
| Max eff. D8 contribution | 0.01% | 0.01% |
| Epoch of max eff. D8 | 37 | 21 |
| Mean eff. D8 (all epochs) | 0.00% | 0.01% |
| Eff. D8 at best epoch | 0.01% | 0.01% |
| Ever above 0.1% | False | False |
| Ever above 1.0% | False | False |
| Ever above 3.0% | False | False |
| Max raw D8 loss | 0.00004155 | 0.00006604 |
| Max lambda applied | 1.00000 | 1.00000 |

## Part E — Full Evaluation Metrics by Split

### Valid

| Metric | Baseline | D8 real | D8 shuffled | Real−Base | Shuf−Base | Real−Shuf |
|--------|----------|---------|-------------|-----------|-----------|-----------|
| mIoU | 0.85456 | 0.85860 | 0.84013 | +0.00404 | -0.01443 | +0.01847 |
| IoU water | 0.74486 | 0.75003 | 0.71777 | +0.00517 | -0.02709 | +0.03226 |
| IoU background | 0.96425 | 0.96717 | 0.96248 | +0.00292 | -0.00177 | +0.00469 |
| Macro F1 | 0.91779 | 0.92024 | 0.90829 | +0.00245 | -0.00950 | +0.01194 |
| F1 water | 0.85378 | 0.85716 | 0.83570 | +0.00338 | -0.01808 | +0.02146 |
| F1 background | 0.98180 | 0.98331 | 0.98088 | +0.00151 | -0.00092 | +0.00243 |
| Pixel accuracy | 0.96763 | 0.97011 | 0.96575 | +0.00248 | -0.00188 | +0.00436 |
| Balanced accuracy | 0.91924 | 0.90145 | 0.88880 | -0.01779 | -0.03044 | +0.01265 |
| Precision water | 0.85043 | 0.90596 | 0.88692 | +0.05553 | +0.03650 | +0.01903 |
| Recall water | 0.85716 | 0.81335 | 0.79007 | -0.04380 | -0.06709 | +0.02328 |
| Specificity (TNR) | 0.98132 | 0.98954 | 0.98752 | +0.00822 | +0.00620 | +0.00202 |
| FPR | 0.01868 | 0.01046 | 0.01248 | -0.00822 | -0.00620 | -0.00202 |
| FNR | 0.14284 | 0.18665 | 0.20993 | +0.04380 | +0.06709 | -0.02328 |
| GT water ratio | 0.1103 | 0.1103 | 0.1103 | +0.0000 | +0.0000 | +0.0000 |
| Pred water ratio | 0.1111 | 0.0990 | 0.0982 | -0.0121 | -0.0129 | +0.0008 |
| Water ratio error | 0.0009 | 0.0113 | 0.0120 | +0.0104 | +0.0112 | -0.0008 |
| Pred/GT ratio | 1.0079 | 0.8978 | 0.8908 | -0.1101 | -0.1171 | +0.0070 |
| Topo violation frac | 0.001122 | 0.000775 | 0.000648 | -0.000346 | -0.000474 | +0.000127 |

**Confusion matrix:**

| | Baseline | D8 real | D8 shuffled |
|---|---|---|---|
| TP | 1,917,981 | 1,819,966 | 1,767,870 |
| FP | 337,335 | 188,921 | 225,388 |
| FN | 319,624 | 417,639 | 469,735 |
| TN | 17,719,785 | 17,868,199 | 17,831,732 |
| GT water px | 2,237,605 | 2,237,605 | 2,237,605 |
| Pred water px | 2,255,316 | 2,008,887 | 1,993,258 |

### Test

| Metric | Baseline | D8 real | D8 shuffled | Real−Base | Shuf−Base | Real−Shuf |
|--------|----------|---------|-------------|-----------|-----------|-----------|
| mIoU | 0.86147 | 0.86001 | 0.84603 | -0.00146 | -0.01544 | +0.01398 |
| IoU water | 0.76084 | 0.75693 | 0.73279 | -0.00391 | -0.02805 | +0.02414 |
| IoU background | 0.96209 | 0.96309 | 0.95926 | +0.00100 | -0.00284 | +0.00383 |
| Macro F1 | 0.92243 | 0.92142 | 0.91250 | -0.00101 | -0.00993 | +0.00892 |
| F1 water | 0.86418 | 0.86165 | 0.84579 | -0.00253 | -0.01839 | +0.01586 |
| F1 background | 0.98068 | 0.98120 | 0.97921 | +0.00052 | -0.00148 | +0.00199 |
| Pixel accuracy | 0.96617 | 0.96690 | 0.96335 | +0.00072 | -0.00282 | +0.00354 |
| Balanced accuracy | 0.92086 | 0.90577 | 0.89488 | -0.01509 | -0.02598 | +0.01089 |
| Precision water | 0.86796 | 0.90259 | 0.89270 | +0.03463 | +0.02474 | +0.00989 |
| Recall water | 0.86043 | 0.82426 | 0.80357 | -0.03617 | -0.05686 | +0.02069 |
| Specificity (TNR) | 0.98129 | 0.98728 | 0.98619 | +0.00600 | +0.00490 | +0.00109 |
| FPR | 0.01871 | 0.01272 | 0.01381 | -0.00600 | -0.00490 | -0.00109 |
| FNR | 0.13957 | 0.17574 | 0.19643 | +0.03617 | +0.05686 | -0.02069 |
| GT water ratio | 0.1251 | 0.1251 | 0.1251 | +0.0000 | +0.0000 | +0.0000 |
| Pred water ratio | 0.1240 | 0.1142 | 0.1126 | -0.0098 | -0.0114 | +0.0016 |
| Water ratio error | 0.0011 | 0.0109 | 0.0125 | +0.0098 | +0.0114 | -0.0016 |
| Pred/GT ratio | 0.9913 | 0.9132 | 0.9002 | -0.0781 | -0.0912 | +0.0131 |
| Topo violation frac | 0.001099 | 0.000911 | 0.000794 | -0.000188 | -0.000306 | +0.000117 |

**Confusion matrix:**

| | Baseline | D8 real | D8 shuffled |
|---|---|---|---|
| TP | 2,207,957 | 2,115,141 | 2,062,052 |
| FP | 335,890 | 228,267 | 247,858 |
| FN | 358,144 | 450,960 | 504,049 |
| TN | 17,615,376 | 17,722,999 | 17,703,408 |
| GT water px | 2,566,101 | 2,566,101 | 2,566,101 |
| Pred water px | 2,543,847 | 2,343,408 | 2,309,910 |

### Bolivia

| Metric | Baseline | D8 real | D8 shuffled | Real−Base | Shuf−Base | Real−Shuf |
|--------|----------|---------|-------------|-----------|-----------|-----------|
| mIoU | 0.84338 | 0.84194 | 0.83788 | -0.00143 | -0.00549 | +0.00406 |
| IoU water | 0.74230 | 0.73631 | 0.73065 | -0.00599 | -0.01165 | +0.00566 |
| IoU background | 0.94445 | 0.94757 | 0.94512 | +0.00312 | +0.00067 | +0.00245 |
| Macro F1 | 0.91176 | 0.91061 | 0.90807 | -0.00116 | -0.00369 | +0.00253 |
| F1 water | 0.85209 | 0.84813 | 0.84436 | -0.00396 | -0.00773 | +0.00377 |
| F1 background | 0.97143 | 0.97308 | 0.97179 | +0.00165 | +0.00035 | +0.00130 |
| Pixel accuracy | 0.95212 | 0.95427 | 0.95223 | +0.00215 | +0.00012 | +0.00204 |
| Balanced accuracy | 0.91864 | 0.89375 | 0.89735 | -0.02489 | -0.02129 | -0.00360 |
| Precision water | 0.83527 | 0.89601 | 0.87366 | +0.06074 | +0.03840 | +0.02235 |
| Recall water | 0.86960 | 0.80511 | 0.81697 | -0.06449 | -0.05264 | -0.01186 |
| Specificity (TNR) | 0.96767 | 0.98239 | 0.97773 | +0.01472 | +0.01006 | +0.00466 |
| FPR | 0.03233 | 0.01761 | 0.02227 | -0.01472 | -0.01006 | -0.00466 |
| FNR | 0.13040 | 0.19489 | 0.18303 | +0.06449 | +0.05264 | +0.01186 |
| GT water ratio | 0.1586 | 0.1586 | 0.1586 | +0.0000 | +0.0000 | +0.0000 |
| Pred water ratio | 0.1651 | 0.1425 | 0.1483 | -0.0226 | -0.0168 | -0.0058 |
| Water ratio error | 0.0065 | 0.0161 | 0.0103 | +0.0096 | +0.0038 | +0.0058 |
| Pred/GT ratio | 1.0411 | 0.8985 | 0.9351 | -0.1426 | -0.1060 | -0.0366 |
| Topo violation frac | 0.001887 | 0.001552 | 0.001284 | -0.000335 | -0.000603 | +0.000268 |

**Confusion matrix:**

| | Baseline | D8 real | D8 shuffled |
|---|---|---|---|
| TP | 395,551 | 366,215 | 371,608 |
| FP | 78,010 | 42,501 | 53,736 |
| FN | 59,313 | 88,649 | 83,256 |
| TN | 2,334,941 | 2,370,450 | 2,359,215 |
| GT water px | 454,864 | 454,864 | 454,864 |
| Pred water px | 473,561 | 408,716 | 425,344 |

## Part F — DEM Geometry / D8 Weight Distribution

### Val (86 tiles)

| Metric | Value |
|--------|-------|
| Tiles (OK/total) | 86/86 |
| Total valid pixels | 22,544,384 |
| Fraction w=0 (flat — zero activation) | 5.62% |
| Fraction 0<w≤0.1 (very low activation) | 22.37% |
| Fraction w>0.1 (moderate activation) | 72.01% |
| Fraction w>0.5 (strong activation) | 28.10% |
| Fraction w=1 (fully saturated) | 13.21% |
| Mean D8 drop per tile (m) | 0.5288 |
| Mean D8 weight per tile | 0.3596 |

s0=1.0m means full weight only when slope ≥ 1 m per 10m pixel (≥10% grade). GLO-30 DEM is originally 30m and resampled to 512×512; if resampled with bilinear interpolation, effective relief per 10m pixel may be much smaller than 1m.

### Train N=100 seed0 (100 tiles)

| Metric | Value |
|--------|-------|
| Tiles (OK/total) | 100/100 |
| Total valid pixels | 26,214,400 |
| Fraction w=0 (flat — zero activation) | 3.31% |
| Fraction 0<w≤0.1 (very low activation) | 21.94% |
| Fraction w>0.1 (moderate activation) | 74.75% |
| Fraction w>0.5 (strong activation) | 28.94% |
| Fraction w=1 (fully saturated) | 14.15% |
| Mean D8 drop per tile (m) | 0.5639 |
| Mean D8 weight per tile | 0.3731 |

s0=1.0m means full weight only when slope ≥ 1 m per 10m pixel (≥10% grade). GLO-30 DEM is originally 30m and resampled to 512×512; if resampled with bilinear interpolation, effective relief per 10m pixel may be much smaller than 1m.

## Part G — Interpretation

### G1. Real > Shuffled: consistent trajectory or early-stopping artifact?

Real–Shuffled gap at ep22 (shuf's best): -0.01334
Real–Shuffled gap at ep31 (real's best): +0.02543

**Conclusion**: Gap grows from ep22 to ep31 (+-0.01334 → +0.02543). NOT purely early-stopping: the real-DEM model continues to improve relative to shuffled after shuffled's early-stop point. A genuine trajectory difference exists.

### G2. D8 underpowered or satisfied at convergence?

Max effective D8 contribution: 0.01% at epoch 37
Ever >0.1%: False  |  Ever >1%: False

**UNDERPOWERED.** D8 contribution never exceeded 0.1%. This is structural: the loss magnitude is negligible relative to Dice+CE throughout training. To reach 1% effective contribution, lambda would need to be ~101× larger. A 'satisfied' loss would show high contribution mid-training decaying to zero; here the contribution is near-zero from the start.

### G3. Does D8-real improve physical consistency or mainly reduce predictions?

valid: topo_vf baseline=0.001122  real=0.000775 (Δ=-0.0003464)  |  pred_water baseline=11.11%  real=9.90%
test: topo_vf baseline=0.001099  real=0.000911 (Δ=-0.0001883)  |  pred_water baseline=12.40%  real=11.42%
bolivia: topo_vf baseline=0.001887  real=0.001552 (Δ=-0.0003348)  |  pred_water baseline=16.51%  real=14.25%

Topo violation fraction can fall simply because fewer pixels are predicted as water (fewer descending pairs to violate). The predicted water ratio drop is the confound. See G4 for the shuffled comparison.

### G4. Does shuffled have lower topo violations due to conservatism?

valid: topo_vf: base=0.001122  real=0.000775  shuf=0.000648  |  pred_water: base=11.11%  real=9.90%  shuf=9.82%
  → shuffled topo_vf < real: conservatism (under-prediction) is driving lower violation, not DEM-correct routing.
test: topo_vf: base=0.001099  real=0.000911  shuf=0.000794  |  pred_water: base=12.40%  real=11.42%  shuf=11.26%
  → shuffled topo_vf < real: conservatism (under-prediction) is driving lower violation, not DEM-correct routing.
bolivia: topo_vf: base=0.001887  real=0.001552  shuf=0.001284  |  pred_water: base=16.51%  real=14.25%  shuf=14.83%
  → shuffled topo_vf < real: conservatism (under-prediction) is driving lower violation, not DEM-correct routing.

### G5. Is the recall drop caused by underprediction of water?

valid: pred/GT ratio: baseline=1.00792  real=0.89778  |  FNR: baseline=14.28%  real=18.66%
test: pred/GT ratio: baseline=0.99133  real=0.91322  |  FNR: baseline=13.96%  real=17.57%
bolivia: pred/GT ratio: baseline=1.04110  real=0.89855  |  FNR: baseline=13.04%  real=19.49%

pred/GT < 1.0 confirms under-prediction. Higher FNR confirms more missed water. The D8 hinge loss penalises p_upstream > p_downstream, which trains the model to lower water probability for pixels that are upstream of dry pixels — systematically suppressing predictions in valid flood zones if they are topographically upstream.

### G6. Is lambda=1.0 too weak?

Max effective contribution at lambda=1.0: 0.01%
To reach 1% target: need lambda ≈ 102  (i.e. ~102× current).
**Yes, lambda=1.0 is too weak by a factor of ~102.**

### G7. Recommended next step (summary)

**Option B: Rescale lambda before multi-seed.** D8 is structurally underpowered. Seeds 1/2 would replicate a null result. Fix the scale first.

## Part H — Recommended Next Experiment

**Recommendation: B — Rescale lambda before multi-seed.**

Peak effective D8 contribution = 0.01% (target: >1%). Lambda=1.0 is too weak by factor ~101.

Proposed sequence:
1. Create new config: `lambda_topo: 101`   (single knob only; tau=0.05 and s0=1.0 unchanged).
2. Rerun D8-real + D8-shuffled at seed0 only (2 runs).
3. Verify eff. contribution reaches 1–5% before proceeding.
4. If confirmed: lock config -> seeds 1/2 for both D8 and baseline.

**Also run baseline seeds 1/2** before any final comparison — you currently have no baseline variance to calibrate deltas against.

**Command to run next (DO NOT run now — for reference only):**
```powershell
# After updating lambda in config:
# .\scripts\launch_segman_n100_d8_seed0_chain.ps1
```
## Figures

- [val_miou_vs_epoch.png](../reports/figures/segman_n100_d8_seed0/val_miou_vs_epoch.png)
- [val_precision_vs_epoch.png](../reports/figures/segman_n100_d8_seed0/val_precision_vs_epoch.png)
- [val_recall_vs_epoch.png](../reports/figures/segman_n100_d8_seed0/val_recall_vs_epoch.png)
- [train_d8_loss_vs_epoch.png](../reports/figures/segman_n100_d8_seed0/train_d8_loss_vs_epoch.png)
- [eff_d8_contribution_vs_epoch.png](../reports/figures/segman_n100_d8_seed0/eff_d8_contribution_vs_epoch.png)
- [pred_water_ratio_vs_epoch.png](../reports/figures/segman_n100_d8_seed0/pred_water_ratio_vs_epoch.png)
- [val_topo_violation_vs_epoch.png](../reports/figures/segman_n100_d8_seed0/val_topo_violation_vs_epoch.png)

---

*Generated by `experiments_cvpr/segman/diagnose_d8_seed0_posthoc.py`.*
*No training launched. No model or loss code modified.*
