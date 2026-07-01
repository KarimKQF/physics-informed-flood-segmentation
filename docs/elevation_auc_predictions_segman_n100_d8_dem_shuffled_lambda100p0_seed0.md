# Elevation AUC Diagnostic — segman_n100_d8_dem_shuffled_lambda100p0_seed0

**Read-only. No training. No config/loss/model changes.**

- Config: `configs\segman\d8_n100_seed0\n100_seed0_dice_ce_d8_dem_shuffled_lambda100p0.yaml`
- Checkpoint: `E:\flood_research\experiments\segman\runs\segman_n100_d8_dem_shuffled_lambda100p0_seed0\checkpoints\best_checkpoint.pt`

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
| AUC(-elev -> pred binary) [pooled] | 0.7466 |
| AUC(p_water -> GT) [pooled] | 0.9823 |
| AUC(shuffled elev -> GT) [sanity ~0.5] | 0.5001 |
| Elevation gap GT [m] | 95.6850 |
| Elevation gap pred [m] | 95.9584 |

### Per-tile AUC (mean +/- std)

| Metric | Per-tile mean +/- std |
|--------|----------------------|
| AUC(-elev -> GT water) | 0.8024 +/- 0.1844 (n=77) |
| AUC(-elev -> pred binary) | 0.8143 +/- 0.1893 (n=75) |
| AUC(p_water -> GT) | 0.9237 +/- 0.0970 (n=77) |
| AUC(shuffled -> GT) | 0.4986 +/- 0.0125 (n=77) |
| Elevation gap GT [m] | 12.1536 +/- 31.4703 (n=77) |
| Elevation gap pred [m] | 12.3487 +/- 21.4861 (n=75) |

### Pooled Pi Curves (P(water | elevation rank bin))
Bin 0 = lowest elevation, Bin N-1 = highest elevation.

| Bin (low->high elev) | pi_GT | pi_pred | pi_GT - pi_pred |
|---------------------|-------|---------|----------------|
| [0/10] | 0.2420 | 0.2488 | -0.0068 |
| [1/10] | 0.2596 | 0.2464 | 0.0133 |
| [2/10] | 0.2298 | 0.2697 | -0.0400 |
| [3/10] | 0.0761 | 0.0744 | 0.0017 |
| [4/10] | 0.1144 | 0.0999 | 0.0145 |
| [5/10] | 0.0853 | 0.1224 | -0.0371 |
| [6/10] | 0.0400 | 0.0567 | -0.0167 |
| [7/10] | 0.0465 | 0.0438 | 0.0028 |
| [8/10] | 0.0066 | 0.0035 | 0.0031 |
| [9/10] | 0.0022 | 0.0012 | 0.0009 |

## Split: `test` (89 tiles, 89 with valid stats)

### Pooled AUC

| Metric | Value |
|--------|-------|
| AUC(-elev -> GT water) [pooled] | 0.7615 |
| AUC(-elev -> pred binary) [pooled] | 0.7720 |
| AUC(p_water -> GT) [pooled] | 0.9656 |
| AUC(shuffled elev -> GT) [sanity ~0.5] | 0.5000 |
| Elevation gap GT [m] | 92.5569 |
| Elevation gap pred [m] | 93.5481 |

### Per-tile AUC (mean +/- std)

| Metric | Per-tile mean +/- std |
|--------|----------------------|
| AUC(-elev -> GT water) | 0.8110 +/- 0.1755 (n=82) |
| AUC(-elev -> pred binary) | 0.8213 +/- 0.1764 (n=77) |
| AUC(p_water -> GT) | 0.9212 +/- 0.1402 (n=82) |
| AUC(shuffled -> GT) | 0.5002 +/- 0.0129 (n=82) |
| Elevation gap GT [m] | 16.7130 +/- 41.8811 (n=82) |
| Elevation gap pred [m] | 7.8589 +/- 33.3516 (n=77) |

### Pooled Pi Curves (P(water | elevation rank bin))
Bin 0 = lowest elevation, Bin N-1 = highest elevation.

| Bin (low->high elev) | pi_GT | pi_pred | pi_GT - pi_pred |
|---------------------|-------|---------|----------------|
| [0/10] | 0.4386 | 0.4670 | -0.0285 |
| [1/10] | 0.0258 | 0.0486 | -0.0228 |
| [2/10] | 0.3270 | 0.3488 | -0.0218 |
| [3/10] | 0.1433 | 0.1198 | 0.0235 |
| [4/10] | 0.1097 | 0.0781 | 0.0316 |
| [5/10] | 0.0414 | 0.0313 | 0.0101 |
| [6/10] | 0.0946 | 0.1136 | -0.0190 |
| [7/10] | 0.0226 | 0.0248 | -0.0023 |
| [8/10] | 0.0424 | 0.0446 | -0.0022 |
| [9/10] | 0.0053 | 0.0067 | -0.0014 |

## Split: `bolivia` (15 tiles, 15 with valid stats)

### Pooled AUC

| Metric | Value |
|--------|-------|
| AUC(-elev -> GT water) [pooled] | 0.5664 |
| AUC(-elev -> pred binary) [pooled] | 0.5799 |
| AUC(p_water -> GT) [pooled] | 0.9816 |
| AUC(shuffled elev -> GT) [sanity ~0.5] | 0.4995 |
| Elevation gap GT [m] | 2.5837 |
| Elevation gap pred [m] | 3.0278 |

### Per-tile AUC (mean +/- std)

| Metric | Per-tile mean +/- std |
|--------|----------------------|
| AUC(-elev -> GT water) | 0.8556 +/- 0.1010 (n=15) |
| AUC(-elev -> pred binary) | 0.8146 +/- 0.1713 (n=14) |
| AUC(p_water -> GT) | 0.9442 +/- 0.0602 (n=15) |
| AUC(shuffled -> GT) | 0.4944 +/- 0.1028 (n=15) |
| Elevation gap GT [m] | 2.5547 +/- 1.6047 (n=15) |
| Elevation gap pred [m] | 1.8897 +/- 2.2023 (n=14) |

### Pooled Pi Curves (P(water | elevation rank bin))
Bin 0 = lowest elevation, Bin N-1 = highest elevation.

| Bin (low->high elev) | pi_GT | pi_pred | pi_GT - pi_pred |
|---------------------|-------|---------|----------------|
| [0/10] | 0.2084 | 0.2337 | -0.0253 |
| [1/10] | 0.0350 | 0.0199 | 0.0151 |
| [2/10] | 0.0476 | 0.0433 | 0.0044 |
| [3/10] | 0.0148 | 0.0102 | 0.0046 |
| [4/10] | 0.5105 | 0.4983 | 0.0122 |
| [5/10] | 0.6569 | 0.6512 | 0.0056 |
| [6/10] | 0.0697 | 0.0514 | 0.0183 |
| [7/10] | 0.0257 | 0.0132 | 0.0125 |
| [8/10] | 0.0116 | 0.0057 | 0.0059 |
| [9/10] | 0.0058 | 0.0000 | 0.0058 |

---
*Generated by `experiments_cvpr/segman/diagnose_elevation_auc_predictions.py`.*
