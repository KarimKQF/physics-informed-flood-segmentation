# SegMAN-S N=100 D8 Downstream Loss -- seed0 Results

**N_train**: 100  |  **Seed**: 0  |  **Loss**: D8DownstreamLoss (lambda=1.0)
**Baseline** (`segman_n100_dice_ce_seed0`): val=0.85456  test=0.86147  bolivia=0.84338

## Scientific Question

> Does the D8 slope-weighted downstream consistency loss show a DEM-specific
> effect (real > shuffled) that was absent from the V1 4-neighbour loss?

## mIoU Summary

| Condition | val mIoU | test mIoU | bolivia mIoU |
|-----------|----------|-----------|--------------|
| Baseline (Dice+CE) | 0.85456 | 0.86147 | 0.84338 |
| D8 real DEM        | 0.85860 | 0.86001 | 0.84194 |
| D8 shuffled DEM    | 0.84013 | 0.84603 | 0.83788 |
| **Real - Shuffled**  | **+0.01847** | **+0.01398** | **+0.00406** |

## Detailed Metrics by Split

### Val

| Metric | Baseline | D8 real | D8 shuffled | Real-Baseline | Shuf-Baseline | Real-Shuffled |
|--------|----------|---------|-------------|---------------|---------------|---------------|
| mean_iou | 0.85456 | 0.85860 | 0.84013 | +0.00404 | -0.01443 | +0.01847 |
| iou_water | 0.74486 | 0.75003 | 0.71777 | +0.00517 | -0.02709 | +0.03226 |
| f1_water | 0.85378 | 0.85716 | 0.83570 | +0.00338 | -0.01808 | +0.02146 |
| precision_water | 0.85043 | 0.90596 | 0.88692 | +0.05553 | +0.03650 | +0.01903 |
| recall_water | 0.85716 | 0.81335 | 0.79007 | -0.04380 | -0.06709 | +0.02328 |
| topo_violation_fraction | 0.00112 | 0.00078 | 0.00065 | -0.00035 | -0.00047 | +0.00013 |

### Test

| Metric | Baseline | D8 real | D8 shuffled | Real-Baseline | Shuf-Baseline | Real-Shuffled |
|--------|----------|---------|-------------|---------------|---------------|---------------|
| mean_iou | 0.86147 | 0.86001 | 0.84603 | -0.00146 | -0.01544 | +0.01398 |
| iou_water | 0.76084 | 0.75693 | 0.73279 | -0.00391 | -0.02805 | +0.02414 |
| f1_water | 0.86418 | 0.86165 | 0.84579 | -0.00253 | -0.01839 | +0.01586 |
| precision_water | 0.86796 | 0.90259 | 0.89270 | +0.03463 | +0.02474 | +0.00989 |
| recall_water | 0.86043 | 0.82426 | 0.80357 | -0.03617 | -0.05686 | +0.02069 |
| topo_violation_fraction | 0.00110 | 0.00091 | 0.00079 | -0.00019 | -0.00031 | +0.00012 |

### Bolivia

| Metric | Baseline | D8 real | D8 shuffled | Real-Baseline | Shuf-Baseline | Real-Shuffled |
|--------|----------|---------|-------------|---------------|---------------|---------------|
| mean_iou | 0.84338 | 0.84194 | 0.83788 | -0.00143 | -0.00549 | +0.00406 |
| iou_water | 0.74230 | 0.73631 | 0.73065 | -0.00599 | -0.01165 | +0.00566 |
| f1_water | 0.85209 | 0.84813 | 0.84436 | -0.00396 | -0.00773 | +0.00377 |
| precision_water | 0.83527 | 0.89601 | 0.87366 | +0.06074 | +0.03840 | +0.02235 |
| recall_water | 0.86960 | 0.80511 | 0.81697 | -0.06449 | -0.05264 | -0.01186 |
| topo_violation_fraction | 0.00189 | 0.00155 | 0.00128 | -0.00033 | -0.00060 | +0.00027 |

## Training Loss at Best Epoch

| Condition | loss_total | loss_d8 | eff_d8_contribution | best_ep |
|-----------|------------|---------|---------------------|---------|
| dice_ce | 0.45663 | 0.000000 | 0.0000 | 34 |
| dice_ce_d8 | 0.43864 | 0.000031 | 0.0001 | 31 |
| dice_ce_d8_dem_shuffled | 0.50499 | 0.000038 | 0.0001 | 22 |

## Interpretation

- **Real-Shuffled > 0**: real DEM provides benefit (D8 captured DEM-specific signal).
- **Real-Shuffled <= 0**: shuffled ~= real (D8 still DEM-agnostic; rethink formulation).
- **Real-Baseline > 0**: D8 loss improves over Dice+CE (strong success).
- **Real-Baseline < 0 and Real > Shuffled**: D8 captures signal but hurts overall mIoU
  (possible over-suppression of recall; try smaller lambda or different tau).

## Notes

- Baseline: `segman_n100_dice_ce_seed0`
- GT diagnostic confirmed strong DEM-specific signal (AUC 0.79-0.86 real vs 0.48-0.51 shuffled).
- V1 topo loss (4-neighbour, lambda 1/2/4) showed no robust real>shuffled signal.
- D8 loss uses 8-neighbour steepest descent; shuffled DEM reroutes downstream direction.
- Aggregation: `experiments_cvpr/segman/aggregate_segman_n100_d8_seed0.py`
- DEM is NOT a model input. Used only in loss and eval topo metrics.
