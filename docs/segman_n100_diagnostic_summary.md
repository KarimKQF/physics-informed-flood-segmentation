# SegMAN-S N=100 Diagnostic — Aggregated Results

**N_train**: 100  |  **Seeds**: 0, 1, 2  |  **Conditions**: Dice+CE / Topo real / Topo shuffled

## Key Scientific Question

> Does the real DEM topographic loss beat the shuffled-DEM control at N=100?
> Main criterion: **Topo real > Topo shuffled consistently across seeds.**

## Val Split

| Condition | mIoU | IoU_water | F1_water | Prec_water | Rec_water | Topo_viol |
|-----------|------|-----------|----------|------------|-----------|-----------|
| Dice+CE | 0.8390 +/- 0.0172 | 0.7167 +/- 0.0311 | 0.8347 +/- 0.0212 | 0.8629 +/- 0.0126 | 0.8099 +/- 0.0504 | 0.0009 +/- 0.0002 |
| Dice+CE+Topo | 0.8414 +/- 0.0111 | 0.7206 +/- 0.0198 | 0.8375 +/- 0.0135 | 0.8720 +/- 0.0115 | 0.8062 +/- 0.0286 | 0.0009 +/- 0.0001 |
| Dice+CE+Topo+Shuffled | 0.8430 +/- 0.0085 | 0.7241 +/- 0.0144 | 0.8399 +/- 0.0096 | 0.8584 +/- 0.0301 | 0.8232 +/- 0.0241 | 0.0010 +/- 0.0001 |

## Test Split

| Condition | mIoU | IoU_water | F1_water | Prec_water | Rec_water | Topo_viol |
|-----------|------|-----------|----------|------------|-----------|-----------|
| Dice+CE | 0.8512 +/- 0.0090 | 0.7425 +/- 0.0160 | 0.8522 +/- 0.0105 | 0.8761 +/- 0.0124 | 0.8300 +/- 0.0269 | 0.0010 +/- 0.0001 |
| Dice+CE+Topo | 0.8526 +/- 0.0056 | 0.7447 +/- 0.0101 | 0.8536 +/- 0.0066 | 0.8835 +/- 0.0096 | 0.8261 +/- 0.0208 | 0.0010 +/- 0.0000 |
| Dice+CE+Topo+Shuffled | 0.8552 +/- 0.0058 | 0.7497 +/- 0.0091 | 0.8569 +/- 0.0059 | 0.8735 +/- 0.0339 | 0.8422 +/- 0.0242 | 0.0011 +/- 0.0001 |

## Bolivia Split

| Condition | mIoU | IoU_water | F1_water | Prec_water | Rec_water | Topo_viol |
|-----------|------|-----------|----------|------------|-----------|-----------|
| Dice+CE | 0.8271 +/- 0.0221 | 0.7111 +/- 0.0398 | 0.8307 +/- 0.0275 | 0.8997 +/- 0.0559 | 0.7785 +/- 0.0862 | 0.0015 +/- 0.0003 |
| Dice+CE+Topo | 0.8189 +/- 0.0154 | 0.6969 +/- 0.0278 | 0.8211 +/- 0.0191 | 0.9040 +/- 0.0301 | 0.7548 +/- 0.0550 | 0.0017 +/- 0.0001 |
| Dice+CE+Topo+Shuffled | 0.8138 +/- 0.0313 | 0.6890 +/- 0.0563 | 0.8150 +/- 0.0402 | 0.8888 +/- 0.0589 | 0.7623 +/- 0.1063 | 0.0018 +/- 0.0002 |

## Per-Seed Test mIoU

| Condition | seed0 | seed1 | seed2 |
|-----------|-------|-------|-------|
| Dice+CE | 0.8615 | 0.8444 | 0.8477 |
| Dice+CE+Topo | 0.8580 | 0.8469 | 0.8529 |
| Dice+CE+Topo+Shuffled | 0.8578 | 0.8485 | 0.8592 |

## Paired Deltas (test mIoU)

| Comparison | seed0 | seed1 | seed2 | Mean +/- Std |
|------------|-------|-------|-------|-------------|
| Topo real - Dice+CE | -0.00343 | 0.00252 | 0.00516 | 0.0014 +/- 0.0044 |
| Shuffled  - Topo real | -0.00020 | 0.00157 | 0.00629 | 0.0026 +/- 0.0034 |
| Shuffled  - Dice+CE | -0.00363 | 0.00408 | 0.01146 | 0.0040 +/- 0.0075 |

## Interpretation

Compared to N=50 (where std dominated signal):

  - Topo real - Dice+CE:    does topo help over baseline?: +0.0014 +/- 0.0044  ->  **neutral (noise)**
  - Shuffled  - Topo real:  is shuffled >= real? (null effect): +0.0026 +/- 0.0034  ->  **neutral (noise)**
  - Shuffled  - Dice+CE:    does any topo term help?: +0.0040 +/- 0.0075  ->  **neutral (noise)**

**DEM effect established** if: Topo real - Dice+CE > 0 AND Shuffled - Topo real <= 0 consistently.

## Notes

- N=100 is a strict superset of N=50 (same seed): positions 0-49 in shuffled order are identical.
- DEM is NOT a model input. Used only in topographic loss during training.
- Eval-time topo metrics always use the real DEM for physical coherence measurement.
- Aggregation script: `experiments_cvpr/segman/aggregate_segman_n100_diagnostic.py`
