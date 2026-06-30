# SegMAN-S N=100 Lambda Stress Test — seed0 Results

**N_train**: 100  |  **Seed**: 0  |  **Lambdas tested**: 1.0, 2.0, 4.0
**Baseline** (`segman_n100_dice_ce_seed0`): val=0.85456  test=0.86147  bolivia=0.84338

## Scientific Question

> Does a stronger lambda_topo reveal a real DEM effect that was absent at lambda=0.5?
> Key criterion: **Topo real > Topo shuffled** at some lambda, consistently.

## Val mIoU by Lambda

| Lambda | Topo real | Topo shuffled | Real−Baseline | Shuf−Baseline | Real−Shuffled |
|--------|-----------|---------------|---------------|---------------|---------------|
| 1.0 | 0.84935 | 0.84626 | -0.00520 | -0.00830 | 0.00310 |
| 2.0 | 0.84713 | 0.84948 | -0.00743 | -0.00508 | -0.00235 |
| 4.0 | 0.84731 | 0.84491 | -0.00725 | -0.00965 | 0.00240 |

## Test mIoU by Lambda

| Lambda | Topo real | Topo shuffled | Real−Baseline | Shuf−Baseline | Real−Shuffled |
|--------|-----------|---------------|---------------|---------------|---------------|
| 1.0 | 0.85143 | 0.85829 | -0.01003 | -0.00318 | -0.00686 |
| 2.0 | 0.86412 | 0.86079 | 0.00265 | -0.00067 | 0.00333 |
| 4.0 | 0.85162 | 0.85510 | -0.00985 | -0.00637 | -0.00349 |

## Bolivia mIoU by Lambda

| Lambda | Topo real | Topo shuffled | Real−Baseline | Shuf−Baseline | Real−Shuffled |
|--------|-----------|---------------|---------------|---------------|---------------|
| 1.0 | 0.84421 | 0.84149 | 0.00083 | -0.00188 | 0.00271 |
| 2.0 | 0.84352 | 0.84843 | 0.00014 | 0.00505 | -0.00491 |
| 4.0 | 0.84197 | 0.83700 | -0.00141 | -0.00638 | 0.00497 |

## Loss at Best Epoch (train, by lambda)

| Lambda | Condition | loss_total | loss_topo | eff_topo_contribution | best_ep |
|--------|-----------|------------|-----------|----------------------|---------|
| 1.0 | dice_ce_topo | 0.43915 | 0.01310 | 0.0307 | 40 |
| 1.0 | dice_ce_topo_dem_shuffled | 0.50805 | 0.02135 | 0.0439 | 30 |
| 2.0 | dice_ce_topo | 0.51119 | 0.01561 | 0.0651 | 34 |
| 2.0 | dice_ce_topo_dem_shuffled | 0.49112 | 0.01911 | 0.0844 | 34 |
| 4.0 | dice_ce_topo | 0.49277 | 0.01306 | 0.1186 | 32 |
| 4.0 | dice_ce_topo_dem_shuffled | 0.55736 | 0.02032 | 0.1708 | 34 |

## Interpretation

- **Real−Shuffled > 0**: real DEM provides benefit over random noise at this lambda.
- **Real−Shuffled ≤ 0**: shuffled control is equal or better → DEM content not captured.
- **Real−Baseline < 0**: lambda too strong; topo loss hurts mIoU.

Interpretation by lambda:

- **lambda=1.0**: Shuffled > Real (-0.0069) — DEM content still not captured [WARN: lambda may be too strong, degrades mIoU vs baseline]
- **lambda=2.0**: No clear signal (Real−Shuffled=+0.0033, within noise)
- **lambda=4.0**: No clear signal (Real−Shuffled=-0.0035, within noise)

## Notes

- Baseline: `segman_n100_dice_ce_seed0` (lambda_topo=0.0 throughout).
- This is a single-seed stress test. Multi-seed replication needed before final conclusions.
- DEM is NOT a model input. Used only in topographic loss and eval topo metrics.
- Aggregation: `experiments_cvpr/segman/aggregate_segman_n100_lambda_sweep.py`
