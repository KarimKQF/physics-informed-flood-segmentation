# Elevation AUC Diagnostic — segman_n100_d8_lambda1p0_seed0

**Read-only. No training. No config/loss/model changes.**

- Config: `configs\segman\d8_n100_seed0\n100_seed0_dice_ce_d8_lambda1p0.yaml`
- Checkpoint: `E:\flood_research\experiments\segman\runs\segman_n100_d8_lambda1p0_seed0\checkpoints\best_checkpoint.pt`

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
| AUC(-elev -> pred binary) [pooled] | 0.7547 |
| AUC(p_water -> GT) [pooled] | 0.9845 |
| AUC(shuffled elev -> GT) [sanity ~0.5] | 0.5001 |
| Elevation gap GT [m] | 95.6850 |
| Elevation gap pred [m] | 94.7447 |

### Per-tile AUC (mean +/- std)

| Metric | Per-tile mean +/- std |
|--------|----------------------|
| AUC(-elev -> GT water) | 0.8024 +/- 0.1844 (n=77) |
| AUC(-elev -> pred binary) | 0.8152 +/- 0.1960 (n=72) |
| AUC(p_water -> GT) | 0.9275 +/- 0.0971 (n=77) |
| AUC(shuffled -> GT) | 0.4986 +/- 0.0125 (n=77) |
| Elevation gap GT [m] | 12.1536 +/- 31.4703 (n=77) |
| Elevation gap pred [m] | 12.3603 +/- 23.0154 (n=72) |

### Pooled Pi Curves (P(water | elevation rank bin))
Bin 0 = lowest elevation, Bin N-1 = highest elevation.

| Bin (low->high elev) | pi_GT | pi_pred | pi_GT - pi_pred |
|---------------------|-------|---------|----------------|
| [0/10] | 0.2420 | 0.2355 | 0.0065 |
| [1/10] | 0.2596 | 0.2363 | 0.0233 |
| [2/10] | 0.2298 | 0.1967 | 0.0330 |
| [3/10] | 0.0761 | 0.0550 | 0.0211 |
| [4/10] | 0.1144 | 0.0859 | 0.0286 |
| [5/10] | 0.0853 | 0.0903 | -0.0050 |
| [6/10] | 0.0400 | 0.0430 | -0.0029 |
| [7/10] | 0.0465 | 0.0424 | 0.0041 |
| [8/10] | 0.0066 | 0.0031 | 0.0035 |
| [9/10] | 0.0022 | 0.0016 | 0.0006 |

## Split: `test` (89 tiles, 89 with valid stats)

### Pooled AUC

| Metric | Value |
|--------|-------|
| AUC(-elev -> GT water) [pooled] | 0.7615 |
| AUC(-elev -> pred binary) [pooled] | 0.7814 |
| AUC(p_water -> GT) [pooled] | 0.9862 |
| AUC(shuffled elev -> GT) [sanity ~0.5] | 0.5000 |
| Elevation gap GT [m] | 92.5569 |
| Elevation gap pred [m] | 96.1082 |

### Per-tile AUC (mean +/- std)

| Metric | Per-tile mean +/- std |
|--------|----------------------|
| AUC(-elev -> GT water) | 0.8110 +/- 0.1755 (n=82) |
| AUC(-elev -> pred binary) | 0.8107 +/- 0.1905 (n=77) |
| AUC(p_water -> GT) | 0.9246 +/- 0.1374 (n=82) |
| AUC(shuffled -> GT) | 0.5002 +/- 0.0129 (n=82) |
| Elevation gap GT [m] | 16.7130 +/- 41.8811 (n=82) |
| Elevation gap pred [m] | 7.0441 +/- 38.3237 (n=77) |

### Pooled Pi Curves (P(water | elevation rank bin))
Bin 0 = lowest elevation, Bin N-1 = highest elevation.

| Bin (low->high elev) | pi_GT | pi_pred | pi_GT - pi_pred |
|---------------------|-------|---------|----------------|
| [0/10] | 0.4386 | 0.4470 | -0.0084 |
| [1/10] | 0.0258 | 0.0195 | 0.0063 |
| [2/10] | 0.3270 | 0.3182 | 0.0088 |
| [3/10] | 0.1433 | 0.1129 | 0.0304 |
| [4/10] | 0.1097 | 0.0704 | 0.0392 |
| [5/10] | 0.0414 | 0.0286 | 0.0128 |
| [6/10] | 0.0946 | 0.0784 | 0.0163 |
| [7/10] | 0.0226 | 0.0192 | 0.0034 |
| [8/10] | 0.0424 | 0.0433 | -0.0008 |
| [9/10] | 0.0053 | 0.0047 | 0.0006 |

## Split: `bolivia` (15 tiles, 15 with valid stats)

### Pooled AUC

| Metric | Value |
|--------|-------|
| AUC(-elev -> GT water) [pooled] | 0.5664 |
| AUC(-elev -> pred binary) [pooled] | 0.5756 |
| AUC(p_water -> GT) [pooled] | 0.9837 |
| AUC(shuffled elev -> GT) [sanity ~0.5] | 0.4995 |
| Elevation gap GT [m] | 2.5837 |
| Elevation gap pred [m] | 2.8721 |

### Per-tile AUC (mean +/- std)

| Metric | Per-tile mean +/- std |
|--------|----------------------|
| AUC(-elev -> GT water) | 0.8556 +/- 0.1010 (n=15) |
| AUC(-elev -> pred binary) | 0.8047 +/- 0.1779 (n=14) |
| AUC(p_water -> GT) | 0.9593 +/- 0.0386 (n=15) |
| AUC(shuffled -> GT) | 0.4944 +/- 0.1028 (n=15) |
| Elevation gap GT [m] | 2.5547 +/- 1.6047 (n=15) |
| Elevation gap pred [m] | 1.8652 +/- 2.0002 (n=14) |

### Pooled Pi Curves (P(water | elevation rank bin))
Bin 0 = lowest elevation, Bin N-1 = highest elevation.

| Bin (low->high elev) | pi_GT | pi_pred | pi_GT - pi_pred |
|---------------------|-------|---------|----------------|
| [0/10] | 0.2084 | 0.1960 | 0.0124 |
| [1/10] | 0.0350 | 0.0251 | 0.0099 |
| [2/10] | 0.0476 | 0.0447 | 0.0029 |
| [3/10] | 0.0148 | 0.0103 | 0.0044 |
| [4/10] | 0.5105 | 0.4790 | 0.0315 |
| [5/10] | 0.6569 | 0.6202 | 0.0367 |
| [6/10] | 0.0697 | 0.0339 | 0.0358 |
| [7/10] | 0.0257 | 0.0100 | 0.0158 |
| [8/10] | 0.0116 | 0.0061 | 0.0055 |
| [9/10] | 0.0058 | 0.0000 | 0.0058 |

---
*Generated by `experiments_cvpr/segman/diagnose_elevation_auc_predictions.py`.*
