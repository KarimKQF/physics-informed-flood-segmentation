# Elevation AUC Diagnostic — segman_n100_d8_lambda100p0_seed0

**Read-only. No training. No config/loss/model changes.**

- Config: `configs\segman\d8_n100_seed0\n100_seed0_dice_ce_d8_lambda100p0.yaml`
- Checkpoint: `E:\flood_research\experiments\segman\runs\segman_n100_d8_lambda100p0_seed0\checkpoints\best_checkpoint.pt`

## Metrics

| Metric | Description |
|--------|-------------|
| `AUC(-elev -> GT water)` | AUROC of negative elevation as predictor of GT flood label. ~0.8 in train data. |
| `AUC(-elev -> pred binary)` | Same but predicting model's binary output. If < AUC_GT -> model under-uses elevation. |
| `AUC(p_water -> GT)` | Standard model discrimination AUC. |
| `AUC shuffled GT` | Sanity check: permuted elevation -> GT label. Should be ~0.5. |
| `elev_gap_GT` | mean(h dry) - mean(h water) for GT labels. Positive -> water is lower. |
| `elev_gap_pred` | Same for model predictions. |
| `pi_GT(b)` | P(GT water | elevation-rank bin b). Should decrease as elevation increases. |
| `pi_pred(b)` | P(pred water | elevation-rank bin b). |

## Decision Rule

- `AUC_pred >= AUC_GT` OR `pi_pred >= pi_GT` -> model already captures elevation structure -> **stop loss engineering**
- `AUC_pred < AUC_GT` AND `pi_pred flatter than pi_GT` -> model under-uses elevation -> **propose ElevationPriorLoss plan**

## Split: `val` (86 tiles, 86 with valid stats)

### Pooled AUC

| Metric | Value |
|--------|-------|
| AUC(-elev -> GT water) [pooled] | 0.7537 |
| AUC(-elev -> pred binary) [pooled] | 0.7479 |
| AUC(p_water -> GT) [pooled] | 0.9805 |
| AUC(shuffled elev -> GT) [sanity ~0.5] | 0.5001 |
| Elevation gap GT [m] | 95.6850 |
| Elevation gap pred [m] | 95.8957 |

### Per-tile AUC (mean +/- std)

| Metric | Per-tile mean +/- std |
|--------|----------------------|
| AUC(-elev -> GT water) | 0.8024 +/- 0.1844 (n=77) |
| AUC(-elev -> pred binary) | 0.8074 +/- 0.1896 (n=76) |
| AUC(p_water -> GT) | 0.9080 +/- 0.1155 (n=77) |
| AUC(shuffled -> GT) | 0.4986 +/- 0.0125 (n=77) |
| Elevation gap GT [m] | 12.1536 +/- 31.4703 (n=77) |
| Elevation gap pred [m] | 11.9824 +/- 21.2671 (n=76) |

### Pooled Pi Curves (P(water | elevation rank bin))
Bin 0 = lowest elevation, Bin N-1 = highest elevation.

| Bin (low->high elev) | pi_GT | pi_pred | pi_GT - pi_pred |
|---------------------|-------|---------|----------------|
| [0/10] | 0.2420 | 0.2443 | -0.0023 |
| [1/10] | 0.2596 | 0.2431 | 0.0165 |
| [2/10] | 0.2298 | 0.2518 | -0.0220 |
| [3/10] | 0.0761 | 0.0835 | -0.0074 |
| [4/10] | 0.1144 | 0.0972 | 0.0173 |
| [5/10] | 0.0853 | 0.1218 | -0.0364 |
| [6/10] | 0.0400 | 0.0469 | -0.0069 |
| [7/10] | 0.0465 | 0.0416 | 0.0049 |
| [8/10] | 0.0066 | 0.0030 | 0.0036 |
| [9/10] | 0.0022 | 0.0013 | 0.0008 |

## Split: `test` (89 tiles, 89 with valid stats)

### Pooled AUC

| Metric | Value |
|--------|-------|
| AUC(-elev -> GT water) [pooled] | 0.7615 |
| AUC(-elev -> pred binary) [pooled] | 0.7726 |
| AUC(p_water -> GT) [pooled] | 0.9833 |
| AUC(shuffled elev -> GT) [sanity ~0.5] | 0.5000 |
| Elevation gap GT [m] | 92.5569 |
| Elevation gap pred [m] | 93.9975 |

### Per-tile AUC (mean +/- std)

| Metric | Per-tile mean +/- std |
|--------|----------------------|
| AUC(-elev -> GT water) | 0.8110 +/- 0.1755 (n=82) |
| AUC(-elev -> pred binary) | 0.8026 +/- 0.1920 (n=74) |
| AUC(p_water -> GT) | 0.9169 +/- 0.1390 (n=82) |
| AUC(shuffled -> GT) | 0.5002 +/- 0.0129 (n=82) |
| Elevation gap GT [m] | 16.7130 +/- 41.8811 (n=82) |
| Elevation gap pred [m] | 3.9339 +/- 49.1564 (n=74) |

### Pooled Pi Curves (P(water | elevation rank bin))
Bin 0 = lowest elevation, Bin N-1 = highest elevation.

| Bin (low->high elev) | pi_GT | pi_pred | pi_GT - pi_pred |
|---------------------|-------|---------|----------------|
| [0/10] | 0.4386 | 0.4566 | -0.0180 |
| [1/10] | 0.0258 | 0.0268 | -0.0009 |
| [2/10] | 0.3270 | 0.3569 | -0.0299 |
| [3/10] | 0.1433 | 0.1201 | 0.0232 |
| [4/10] | 0.1097 | 0.0769 | 0.0328 |
| [5/10] | 0.0414 | 0.0346 | 0.0068 |
| [6/10] | 0.0946 | 0.1074 | -0.0128 |
| [7/10] | 0.0226 | 0.0194 | 0.0031 |
| [8/10] | 0.0424 | 0.0412 | 0.0012 |
| [9/10] | 0.0053 | 0.0059 | -0.0007 |

## Split: `bolivia` (15 tiles, 15 with valid stats)

### Pooled AUC

| Metric | Value |
|--------|-------|
| AUC(-elev -> GT water) [pooled] | 0.5664 |
| AUC(-elev -> pred binary) [pooled] | 0.5869 |
| AUC(p_water -> GT) [pooled] | 0.9818 |
| AUC(shuffled elev -> GT) [sanity ~0.5] | 0.4995 |
| Elevation gap GT [m] | 2.5837 |
| Elevation gap pred [m] | 3.2266 |

### Per-tile AUC (mean +/- std)

| Metric | Per-tile mean +/- std |
|--------|----------------------|
| AUC(-elev -> GT water) | 0.8556 +/- 0.1010 (n=15) |
| AUC(-elev -> pred binary) | 0.8056 +/- 0.1937 (n=14) |
| AUC(p_water -> GT) | 0.9395 +/- 0.0777 (n=15) |
| AUC(shuffled -> GT) | 0.4944 +/- 0.1028 (n=15) |
| Elevation gap GT [m] | 2.5547 +/- 1.6047 (n=15) |
| Elevation gap pred [m] | 1.7160 +/- 2.3971 (n=14) |

### Pooled Pi Curves (P(water | elevation rank bin))
Bin 0 = lowest elevation, Bin N-1 = highest elevation.

| Bin (low->high elev) | pi_GT | pi_pred | pi_GT - pi_pred |
|---------------------|-------|---------|----------------|
| [0/10] | 0.2084 | 0.2788 | -0.0704 |
| [1/10] | 0.0350 | 0.0343 | 0.0007 |
| [2/10] | 0.0476 | 0.0433 | 0.0043 |
| [3/10] | 0.0148 | 0.0103 | 0.0044 |
| [4/10] | 0.5105 | 0.5141 | -0.0036 |
| [5/10] | 0.6569 | 0.6920 | -0.0352 |
| [6/10] | 0.0697 | 0.0765 | -0.0068 |
| [7/10] | 0.0257 | 0.0174 | 0.0084 |
| [8/10] | 0.0116 | 0.0064 | 0.0052 |
| [9/10] | 0.0058 | 0.0001 | 0.0058 |

---
*Generated by `experiments_cvpr/segman/diagnose_elevation_auc_predictions.py`.*
