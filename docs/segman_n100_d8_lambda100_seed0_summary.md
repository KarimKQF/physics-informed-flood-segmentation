# SegMAN-S N=100 D8 Downstream Loss -- lambda=100 vs lambda=1 Comparison (seed0)

**Scientific question**: Does lambda=100 give D8 contribution in the 1-5% target range?
Does D8 real still beat D8 shuffled? Does precision/recall rebalance?

**DEM note**: DEM is NEVER a model input. Used only in the D8 loss and topo eval metrics.

## mIoU Summary

| Condition | lambda | Val mIoU | Test mIoU | Bolivia mIoU | Best ep |
|-----------|--------|----------|-----------|--------------|---------|
| Dice+CE baseline | 0.0 | 0.85456 | 0.86147 | 0.84338 | 34 |
| D8 real  λ=1 | 1.0 | 0.85860 | 0.86001 | 0.84194 | 31 |
| D8 shuf  λ=1 | 1.0 | 0.84013 | 0.84603 | 0.83788 | 22 |
| D8 real  λ=100 | 100.0 | 0.83984 | 0.85789 | 0.84084 | 30 |
| D8 shuf  λ=100 | 100.0 | 0.84328 | 0.84799 | 0.83669 | 58 |

## Key Deltas (val mIoU, water metrics)

| Comparison | Val mIoU | Val water IoU | Val Precision | Val Recall | Val Topo VF |
|------------|----------|---------------|---------------|------------|------------|
| λ=100 real − baseline | -0.01472 | -0.02499 | -0.02506 | -0.00793 | -0.000072 |
| λ=100 shuf − baseline | -0.01127 | -0.01852 | -0.03219 | +0.00893 | -0.000136 |
| λ=100 real − λ=100 shuf | -0.00344 | -0.00647 | +0.00713 | -0.01686 | +0.000064 |
| λ=100 real − λ=1 real | -0.01876 | -0.03015 | -0.08059 | +0.03588 | +0.000274 |
| λ=100 shuf − λ=1 shuf | +0.00316 | +0.00857 | -0.06869 | +0.07602 | +0.000337 |
| λ=1   real − baseline | +0.00404 | +0.00517 | +0.05553 | -0.04380 | -0.000346 |
| λ=1   shuf − baseline | -0.01443 | -0.02709 | +0.03650 | -0.06709 | -0.000474 |

## Effective D8 Contribution at Best Epoch

| Condition | lambda | Raw D8 loss | lambda*D8 | dice+ce | Eff D8% | Best ep |
|-----------|--------|-------------|-----------|---------|---------|---------|
| Dice+CE baseline | 0.0 | 0.0000000 | 0.0000000 | 0.45663 | 0.00% | 34 |
| D8 real  λ=1 | 1.0 | 0.0000312 | 0.0000312 | 0.43861 | 0.01% | 31 |
| D8 shuf  λ=1 | 1.0 | 0.0000379 | 0.0000379 | 0.50495 | 0.01% | 22 |
| D8 real  λ=100 | 100.0 | 0.0000148 | 0.0014850 | 0.49330 | 0.30% | 30 |
| D8 shuf  λ=100 | 100.0 | 0.0000477 | 0.0047748 | 0.46746 | 1.02% | 58 |

## Detailed Metrics by Split

### Valid

| Metric | Baseline | D8 real λ=1 | D8 shuf λ=1 | D8 real λ=100 | D8 shuf λ=100 |
|---|---|---|---|---|---|
| mIoU | 0.85456 | 0.85860 | 0.84013 | 0.83984 | 0.84328 |
| IoU water | 0.74486 | 0.75003 | 0.71777 | 0.71988 | 0.72634 |
| IoU background | 0.96425 | 0.96717 | 0.96248 | 0.95980 | 0.96022 |
| F1 water | 0.85378 | 0.85716 | 0.83570 | 0.83713 | 0.84148 |
| Precision water | 0.85043 | 0.90596 | 0.88692 | 0.82536 | 0.81823 |
| Recall water | 0.85716 | 0.81335 | 0.79007 | 0.84923 | 0.86609 |
| Specificity | 0.98132 | 0.98954 | 0.98752 | 0.97773 | 0.97616 |
| Pixel accuracy | 0.96763 | 0.97011 | 0.96575 | 0.96357 | 0.96402 |
| Balanced accuracy | 0.91924 | 0.90145 | 0.88880 | 0.91348 | 0.92112 |
| FPR | 0.01868 | 0.01046 | 0.01248 | 0.02227 | 0.02384 |
| FNR | 0.14284 | 0.18665 | 0.20993 | 0.15077 | 0.13391 |
| GT water ratio | 0.1103 | 0.1103 | 0.1103 | 0.1103 | 0.1103 |
| Pred water ratio | 0.1111 | 0.0990 | 0.0982 | 0.1134 | 0.1167 |
| Pred/GT ratio | 1.0079 | 0.8978 | 0.8908 | 1.0289 | 1.0585 |
| Topo violation | 0.001122 | 0.000775 | 0.000648 | 0.001049 | 0.000985 |

### Test

| Metric | Baseline | D8 real λ=1 | D8 shuf λ=1 | D8 real λ=100 | D8 shuf λ=100 |
|---|---|---|---|---|---|
| mIoU | 0.86147 | 0.86001 | 0.84603 | 0.85789 | 0.84799 |
| IoU water | 0.76084 | 0.75693 | 0.73279 | 0.75485 | 0.73867 |
| IoU background | 0.96209 | 0.96309 | 0.95926 | 0.96093 | 0.95731 |
| F1 water | 0.86418 | 0.86165 | 0.84579 | 0.86030 | 0.84970 |
| Precision water | 0.86796 | 0.90259 | 0.89270 | 0.86197 | 0.83884 |
| Recall water | 0.86043 | 0.82426 | 0.80357 | 0.85864 | 0.86084 |
| Specificity | 0.98129 | 0.98728 | 0.98619 | 0.98035 | 0.97636 |
| Pixel accuracy | 0.96617 | 0.96690 | 0.96335 | 0.96512 | 0.96191 |
| Balanced accuracy | 0.92086 | 0.90577 | 0.89488 | 0.91949 | 0.91860 |
| FPR | 0.01871 | 0.01272 | 0.01381 | 0.01965 | 0.02364 |
| FNR | 0.13957 | 0.17574 | 0.19643 | 0.14136 | 0.13916 |
| GT water ratio | 0.1251 | 0.1251 | 0.1251 | 0.1251 | 0.1251 |
| Pred water ratio | 0.1240 | 0.1142 | 0.1126 | 0.1246 | 0.1283 |
| Pred/GT ratio | 0.9913 | 0.9132 | 0.9002 | 0.9961 | 1.0262 |
| Topo violation | 0.001099 | 0.000911 | 0.000794 | 0.000941 | 0.000946 |

### Bolivia

| Metric | Baseline | D8 real λ=1 | D8 shuf λ=1 | D8 real λ=100 | D8 shuf λ=100 |
|---|---|---|---|---|---|
| mIoU | 0.84338 | 0.84194 | 0.83788 | 0.84084 | 0.83669 |
| IoU water | 0.74230 | 0.73631 | 0.73065 | 0.73857 | 0.72945 |
| IoU background | 0.94445 | 0.94757 | 0.94512 | 0.94311 | 0.94394 |
| F1 water | 0.85209 | 0.84813 | 0.84436 | 0.84963 | 0.84356 |
| Precision water | 0.83527 | 0.89601 | 0.87366 | 0.82749 | 0.85987 |
| Recall water | 0.86960 | 0.80511 | 0.81697 | 0.87298 | 0.82785 |
| Specificity | 0.96767 | 0.98239 | 0.97773 | 0.96569 | 0.97457 |
| Pixel accuracy | 0.95212 | 0.95427 | 0.95223 | 0.95099 | 0.95130 |
| Balanced accuracy | 0.91864 | 0.89375 | 0.89735 | 0.91934 | 0.90121 |
| FPR | 0.03233 | 0.01761 | 0.02227 | 0.03431 | 0.02543 |
| FNR | 0.13040 | 0.19489 | 0.18303 | 0.12702 | 0.17215 |
| GT water ratio | 0.1586 | 0.1586 | 0.1586 | 0.1586 | 0.1586 |
| Pred water ratio | 0.1651 | 0.1425 | 0.1483 | 0.1673 | 0.1527 |
| Pred/GT ratio | 1.0411 | 0.8985 | 0.9351 | 1.0550 | 0.9628 |
| Topo violation | 0.001887 | 0.001552 | 0.001284 | 0.001606 | 0.001492 |

## Interpretation Guide

- **Eff D8% < 0.1%**: still underpowered at lambda=100 (unexpected — review loss code).
- **Eff D8% 1-5%**: target range — meaningful gradient influence.
- **Eff D8% > 10%**: potentially overregularised — check if recall collapses further.
- **Real > Shuffled (mIoU, water IoU)**: D8 is DEM-specific at lambda=100.
- **Real > Baseline**: D8 provides performance benefit.
- **Recall drops further vs lambda=1**: lambda=100 may be over-suppressing water predictions.

## Notes

- DEM is NOT a model input. Used only in D8 loss and topo eval metrics.
- Baseline: `segman_n100_dice_ce_seed0`
- Aggregation: `experiments_cvpr/segman/aggregate_segman_n100_d8_lambda100_seed0.py`
