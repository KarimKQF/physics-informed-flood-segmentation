# Elevation AUC Diagnostic — segman_n100_d8_dem_shuffled_lambda1p0_seed0

**Read-only. No training. No config/loss/model changes.**

- Config: `configs\segman\d8_n100_seed0\n100_seed0_dice_ce_d8_dem_shuffled_lambda1p0.yaml`
- Checkpoint: `E:\flood_research\experiments\segman\runs\segman_n100_d8_dem_shuffled_lambda1p0_seed0\checkpoints\best_checkpoint.pt`

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
| AUC(-elev -> pred binary) [pooled] | 0.7536 |
| AUC(p_water -> GT) [pooled] | 0.9767 |
| AUC(shuffled elev -> GT) [sanity ~0.5] | 0.5001 |
| Elevation gap GT [m] | 95.6850 |
| Elevation gap pred [m] | 93.5078 |

### Per-tile AUC (mean +/- std)

| Metric | Per-tile mean +/- std |
|--------|----------------------|
| AUC(-elev -> GT water) | 0.8024 +/- 0.1844 (n=77) |
| AUC(-elev -> pred binary) | 0.8013 +/- 0.2100 (n=69) |
| AUC(p_water -> GT) | 0.8917 +/- 0.1345 (n=77) |
| AUC(shuffled -> GT) | 0.4986 +/- 0.0125 (n=77) |
| Elevation gap GT [m] | 12.1536 +/- 31.4703 (n=77) |
| Elevation gap pred [m] | 9.5613 +/- 20.1521 (n=69) |

### Pooled Pi Curves (P(water | elevation rank bin))
Bin 0 = lowest elevation, Bin N-1 = highest elevation.

| Bin (low->high elev) | pi_GT | pi_pred | pi_GT - pi_pred |
|---------------------|-------|---------|----------------|
| [0/10] | 0.2420 | 0.2354 | 0.0066 |
| [1/10] | 0.2596 | 0.2344 | 0.0252 |
| [2/10] | 0.2298 | 0.1912 | 0.0386 |
| [3/10] | 0.0761 | 0.0542 | 0.0219 |
| [4/10] | 0.1144 | 0.0799 | 0.0345 |
| [5/10] | 0.0853 | 0.0956 | -0.0103 |
| [6/10] | 0.0400 | 0.0479 | -0.0079 |
| [7/10] | 0.0465 | 0.0388 | 0.0077 |
| [8/10] | 0.0066 | 0.0022 | 0.0044 |
| [9/10] | 0.0022 | 0.0025 | -0.0003 |

## Split: `test` (89 tiles, 89 with valid stats)

### Pooled AUC

| Metric | Value |
|--------|-------|
| AUC(-elev -> GT water) [pooled] | 0.7615 |
| AUC(-elev -> pred binary) [pooled] | 0.7843 |
| AUC(p_water -> GT) [pooled] | 0.9665 |
| AUC(shuffled elev -> GT) [sanity ~0.5] | 0.5000 |
| Elevation gap GT [m] | 92.5569 |
| Elevation gap pred [m] | 96.4390 |

### Per-tile AUC (mean +/- std)

| Metric | Per-tile mean +/- std |
|--------|----------------------|
| AUC(-elev -> GT water) | 0.8110 +/- 0.1755 (n=82) |
| AUC(-elev -> pred binary) | 0.7630 +/- 0.2305 (n=71) |
| AUC(p_water -> GT) | 0.9038 +/- 0.1375 (n=82) |
| AUC(shuffled -> GT) | 0.5002 +/- 0.0129 (n=82) |
| Elevation gap GT [m] | 16.7130 +/- 41.8811 (n=82) |
| Elevation gap pred [m] | 2.4310 +/- 44.8225 (n=71) |

### Pooled Pi Curves (P(water | elevation rank bin))
Bin 0 = lowest elevation, Bin N-1 = highest elevation.

| Bin (low->high elev) | pi_GT | pi_pred | pi_GT - pi_pred |
|---------------------|-------|---------|----------------|
| [0/10] | 0.4386 | 0.4485 | -0.0099 |
| [1/10] | 0.0258 | 0.0161 | 0.0097 |
| [2/10] | 0.3270 | 0.3149 | 0.0121 |
| [3/10] | 0.1433 | 0.1115 | 0.0317 |
| [4/10] | 0.1097 | 0.0675 | 0.0422 |
| [5/10] | 0.0414 | 0.0265 | 0.0149 |
| [6/10] | 0.0946 | 0.0755 | 0.0191 |
| [7/10] | 0.0226 | 0.0200 | 0.0025 |
| [8/10] | 0.0424 | 0.0396 | 0.0028 |
| [9/10] | 0.0053 | 0.0056 | -0.0003 |

## Split: `bolivia` (15 tiles, 15 with valid stats)

### Pooled AUC

| Metric | Value |
|--------|-------|
| AUC(-elev -> GT water) [pooled] | 0.5664 |
| AUC(-elev -> pred binary) [pooled] | 0.5771 |
| AUC(p_water -> GT) [pooled] | 0.9773 |
| AUC(shuffled elev -> GT) [sanity ~0.5] | 0.4995 |
| Elevation gap GT [m] | 2.5837 |
| Elevation gap pred [m] | 2.9277 |

### Per-tile AUC (mean +/- std)

| Metric | Per-tile mean +/- std |
|--------|----------------------|
| AUC(-elev -> GT water) | 0.8556 +/- 0.1010 (n=15) |
| AUC(-elev -> pred binary) | 0.7945 +/- 0.2045 (n=14) |
| AUC(p_water -> GT) | 0.9444 +/- 0.0591 (n=15) |
| AUC(shuffled -> GT) | 0.4944 +/- 0.1028 (n=15) |
| Elevation gap GT [m] | 2.5547 +/- 1.6047 (n=15) |
| Elevation gap pred [m] | 1.7621 +/- 2.0950 (n=14) |

### Pooled Pi Curves (P(water | elevation rank bin))
Bin 0 = lowest elevation, Bin N-1 = highest elevation.

| Bin (low->high elev) | pi_GT | pi_pred | pi_GT - pi_pred |
|---------------------|-------|---------|----------------|
| [0/10] | 0.2084 | 0.2237 | -0.0153 |
| [1/10] | 0.0350 | 0.0154 | 0.0197 |
| [2/10] | 0.0476 | 0.0405 | 0.0071 |
| [3/10] | 0.0148 | 0.0096 | 0.0052 |
| [4/10] | 0.5105 | 0.4922 | 0.0184 |
| [5/10] | 0.6569 | 0.6364 | 0.0205 |
| [6/10] | 0.0697 | 0.0479 | 0.0218 |
| [7/10] | 0.0257 | 0.0113 | 0.0144 |
| [8/10] | 0.0116 | 0.0061 | 0.0055 |
| [9/10] | 0.0058 | 0.0000 | 0.0058 |

---
*Generated by `experiments_cvpr/segman/diagnose_elevation_auc_predictions.py`.*
