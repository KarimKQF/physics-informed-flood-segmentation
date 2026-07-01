# Elevation AUC Diagnostic — segman_n100_dice_ce_seed0

**Read-only. No training. No config/loss/model changes.**

- Config: `configs\segman\multiseed_n100\n100_seed0_dice_ce.yaml`
- Checkpoint: `E:\flood_research\experiments\segman\runs\segman_n100_dice_ce_seed0\checkpoints\best_checkpoint.pt`

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
| AUC(-elev -> pred binary) [pooled] | 0.7453 |
| AUC(p_water -> GT) [pooled] | 0.9848 |
| AUC(shuffled elev -> GT) [sanity ~0.5] | 0.5001 |
| Elevation gap GT [m] | 95.6850 |
| Elevation gap pred [m] | 94.5335 |

### Per-tile AUC (mean +/- std)

| Metric | Per-tile mean +/- std |
|--------|----------------------|
| AUC(-elev -> GT water) | 0.8024 +/- 0.1844 (n=77) |
| AUC(-elev -> pred binary) | 0.8019 +/- 0.2001 (n=75) |
| AUC(p_water -> GT) | 0.9300 +/- 0.0896 (n=77) |
| AUC(shuffled -> GT) | 0.4986 +/- 0.0125 (n=77) |
| Elevation gap GT [m] | 12.1536 +/- 31.4703 (n=77) |
| Elevation gap pred [m] | 11.9095 +/- 21.2327 (n=75) |

### Pooled Pi Curves (P(water | elevation rank bin))
Bin 0 = lowest elevation, Bin N-1 = highest elevation.

| Bin (low->high elev) | pi_GT | pi_pred | pi_GT - pi_pred |
|---------------------|-------|---------|----------------|
| [0/10] | 0.2420 | 0.2417 | 0.0003 |
| [1/10] | 0.2596 | 0.2441 | 0.0155 |
| [2/10] | 0.2298 | 0.2422 | -0.0124 |
| [3/10] | 0.0761 | 0.0654 | 0.0108 |
| [4/10] | 0.1144 | 0.1019 | 0.0126 |
| [5/10] | 0.0853 | 0.1114 | -0.0261 |
| [6/10] | 0.0400 | 0.0542 | -0.0141 |
| [7/10] | 0.0465 | 0.0439 | 0.0026 |
| [8/10] | 0.0066 | 0.0049 | 0.0017 |
| [9/10] | 0.0022 | 0.0017 | 0.0005 |

## Split: `test` (89 tiles, 89 with valid stats)

### Pooled AUC

| Metric | Value |
|--------|-------|
| AUC(-elev -> GT water) [pooled] | 0.7615 |
| AUC(-elev -> pred binary) [pooled] | 0.7684 |
| AUC(p_water -> GT) [pooled] | 0.9865 |
| AUC(shuffled elev -> GT) [sanity ~0.5] | 0.5000 |
| Elevation gap GT [m] | 92.5569 |
| Elevation gap pred [m] | 92.8088 |

### Per-tile AUC (mean +/- std)

| Metric | Per-tile mean +/- std |
|--------|----------------------|
| AUC(-elev -> GT water) | 0.8110 +/- 0.1755 (n=82) |
| AUC(-elev -> pred binary) | 0.8149 +/- 0.1772 (n=77) |
| AUC(p_water -> GT) | 0.9285 +/- 0.1300 (n=82) |
| AUC(shuffled -> GT) | 0.5002 +/- 0.0129 (n=82) |
| Elevation gap GT [m] | 16.7130 +/- 41.8811 (n=82) |
| Elevation gap pred [m] | 7.5501 +/- 32.7449 (n=77) |

### Pooled Pi Curves (P(water | elevation rank bin))
Bin 0 = lowest elevation, Bin N-1 = highest elevation.

| Bin (low->high elev) | pi_GT | pi_pred | pi_GT - pi_pred |
|---------------------|-------|---------|----------------|
| [0/10] | 0.4386 | 0.4512 | -0.0126 |
| [1/10] | 0.0258 | 0.0328 | -0.0069 |
| [2/10] | 0.3270 | 0.3420 | -0.0150 |
| [3/10] | 0.1433 | 0.1174 | 0.0259 |
| [4/10] | 0.1097 | 0.0788 | 0.0308 |
| [5/10] | 0.0414 | 0.0375 | 0.0039 |
| [6/10] | 0.0946 | 0.1102 | -0.0156 |
| [7/10] | 0.0226 | 0.0206 | 0.0020 |
| [8/10] | 0.0424 | 0.0445 | -0.0021 |
| [9/10] | 0.0053 | 0.0049 | 0.0004 |

## Split: `bolivia` (15 tiles, 15 with valid stats)

### Pooled AUC

| Metric | Value |
|--------|-------|
| AUC(-elev -> GT water) [pooled] | 0.5664 |
| AUC(-elev -> pred binary) [pooled] | 0.5929 |
| AUC(p_water -> GT) [pooled] | 0.9831 |
| AUC(shuffled elev -> GT) [sanity ~0.5] | 0.4995 |
| Elevation gap GT [m] | 2.5837 |
| Elevation gap pred [m] | 3.3799 |

### Per-tile AUC (mean +/- std)

| Metric | Per-tile mean +/- std |
|--------|----------------------|
| AUC(-elev -> GT water) | 0.8556 +/- 0.1010 (n=15) |
| AUC(-elev -> pred binary) | 0.8110 +/- 0.1627 (n=15) |
| AUC(p_water -> GT) | 0.9473 +/- 0.0747 (n=15) |
| AUC(shuffled -> GT) | 0.4944 +/- 0.1028 (n=15) |
| Elevation gap GT [m] | 2.5547 +/- 1.6047 (n=15) |
| Elevation gap pred [m] | 1.9657 +/- 1.8185 (n=15) |

### Pooled Pi Curves (P(water | elevation rank bin))
Bin 0 = lowest elevation, Bin N-1 = highest elevation.

| Bin (low->high elev) | pi_GT | pi_pred | pi_GT - pi_pred |
|---------------------|-------|---------|----------------|
| [0/10] | 0.2084 | 0.2727 | -0.0642 |
| [1/10] | 0.0350 | 0.0560 | -0.0209 |
| [2/10] | 0.0476 | 0.0489 | -0.0013 |
| [3/10] | 0.0148 | 0.0127 | 0.0021 |
| [4/10] | 0.5105 | 0.5080 | 0.0026 |
| [5/10] | 0.6569 | 0.6603 | -0.0035 |
| [6/10] | 0.0697 | 0.0596 | 0.0101 |
| [7/10] | 0.0257 | 0.0222 | 0.0035 |
| [8/10] | 0.0116 | 0.0095 | 0.0021 |
| [9/10] | 0.0058 | 0.0014 | 0.0044 |

---
*Generated by `experiments_cvpr/segman/diagnose_elevation_auc_predictions.py`.*
