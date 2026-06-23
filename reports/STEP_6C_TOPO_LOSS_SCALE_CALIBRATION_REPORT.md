# STEP 6C Topographic Loss Scale Calibration

Generated: 2026-06-22T11:13:16+00:00

## Scope

This calibration used the STEP 6C TerraMind-L + UPerNet setup with Dice segmentation loss and `TopographicInconsistencyLoss`. It ran forward passes only, did not create an optimizer, did not call `optimizer.step()`, did not update weights, did not use DEM as model input, and did not modify raw data. DEM was loaded only for the loss.

The train transform family was preserved as D4 + ToTensorV2. For calibration, ToTensorV2 was handled by the dataloader and the D4 operation was applied explicitly to images, masks, and DEM together so the augmented DEM stayed aligned with the augmented labels.

## Configuration

- Config: `C:/flood_research/repos/physics-informed-flood-segmentation/configs/step6c_terramind_l_upernet_dice_topographic_lambda005.yaml`
- Script: `C:/flood_research/repos/physics-informed-flood-segmentation/scripts/step6c_topo_loss_scale_calibration.py`
- JSON metrics: `E:/flood_research/experiments/terramind_baseline/runs/step6c_terramind_l_upernet_dice_topographic_lambda005/metrics/topo_loss_scale_calibration.json`
- CSV metrics: `E:/flood_research/experiments/terramind_baseline/runs/step6c_terramind_l_upernet_dice_topographic_lambda005/metrics/topo_loss_scale_calibration.csv`
- Log: `E:/flood_research/experiments/terramind_baseline/runs/step6c_terramind_l_upernet_dice_topographic_lambda005/logs/topo_loss_scale_calibration.log`
- Batches requested: 30
- Batches used: 30
- Batch size: 2
- Samples used: 60
- Device: cuda
- Backbone initialization: original TerraMind pretrained checkpoint
- Feature indices: `[5, 11, 17, 23]`
- DEM as model input: false

## Loss Scale

| Metric | Mean | Median | Min | Max | Std |
|---|---:|---:|---:|---:|---:|
| loss_dice | 0.94158655 | 0.97241098 | 0.59971678 | 0.99997461 | 0.08843762 |
| loss_topo | 0.00001003 | 0.00000523 | 0.00000014 | 0.00006661 | 0.00001308 |

## Lambda Sweep

| lambda_topo | mean contribution | mean loss ratio | median loss ratio | mean scaled grad ratio |
|---:|---:|---:|---:|---:|
| 0.05 | 0.00000050 | 0.000052% | 0.000027% | 0.0253 |
| 0.10 | 0.00000100 | 0.000105% | 0.000054% | 0.0506 |
| 0.50 | 0.00000501 | 0.000525% | 0.000271% | 0.2529 |
| 1.00 | 0.00001003 | 0.001049% | 0.000542% | 0.5057 |
| 5.00 | 0.00005013 | 0.005247% | 0.002711% | 2.5285 |
| 10.00 | 0.00010026 | 0.010494% | 0.005421% | 5.0571 |
| 20.00 | 0.00020053 | 0.020988% | 0.010842% | 10.1142 |
| 50.00 | 0.00050132 | 0.052470% | 0.027105% | 25.2854 |

## Recommendations

Using `lambda_needed = target_ratio * mean(loss_dice) / mean(loss_topo)`:

| Target topo contribution | Recommended lambda |
|---:|---:|
| 0.1% of Dice | 93.9099 |
| 1% of Dice | 939.0994 |
| 5% of Dice | 4695.4968 |

Original `lambda_topo=0.05` mean scalar-loss ratio: 0.00005247%. Assessment by scalar loss: **too weak**. Its mean scaled logit-gradient ratio is 0.0253, so it is not a zero-gradient term, but it is very weak as a contribution to the reported scalar loss.

The scalar-loss lambdas above are very large because the initial topographic loss value is tiny. They should not be used blindly: at `lambda_topo=939.0994`, the initial topographic logit-gradient norm would be hundreds of times larger than Dice on average.

Recommended first full STEP 6C run: **lambda_topo=0.5000**, chosen from the tested sweep as a gradient-aware first run.

## Topographic Pair Diagnostics

| Metric | Mean | Median | Min | Max | Std |
|---|---:|---:|---:|---:|---:|
| valid descending pairs | 745325.67 | 858834.00 | 70698.00 | 1043342.00 | 269014.42 |
| descending pair fraction | 0.979032 | 0.999304 | 0.564041 | 0.999985 | 0.079949 |
| positive elevation delta mean | 0.52577939 | 0.25534516 | 0.13459121 | 2.39512375 | 0.55114074 |

- Fraction of batches with `loss_topo == 0`: 0.00%
- Fraction of batches with `loss_topo < 1e-06`: 6.67%

## Gradient Diagnostics

Gradients were measured with respect to logits, not model parameters. This checks loss scale without updating weights.

| Metric | Mean | Median | Min | Max | Std |
|---|---:|---:|---:|---:|---:|
| Dice grad L2 | 0.00000181 | 0.00000154 | 0.00000000 | 0.00001294 | 0.00000221 |
| Topo grad L2 | 0.00000062 | 0.00000027 | 0.00000000 | 0.00000375 | 0.00000082 |
| Topo/Dice grad ratio | 0.50570853 | 0.24041461 | 0.00943484 | 3.66594750 | 0.75840106 |

- Topographic-loss gradients finite in all batches: True
- Topographic-loss gradient norm nonzero in all batches: True

## Probability Diagnostics

| Metric | Mean | Median | Min | Max | Std |
|---|---:|---:|---:|---:|---:|
| p_water mean per batch | 0.98791565 | 0.99996364 | 0.67919803 | 0.99999875 | 0.05778945 |
| p_water std per batch | 0.01380201 | 0.00594687 | 0.00060901 | 0.15274899 | 0.03169824 |

## Interpretation

`loss_topo` is small mainly because the original TerraMind pretrained initialization is highly saturated toward the water class on these batches. Since the loss term contains `p_high_water * (1 - p_low_water)`, predictions that are nearly water everywhere make the scalar topo loss tiny even when many descending DEM pairs exist. The loss also averages over valid descending neighbor pairs. The valid pair mask is not sparse, positive elevation deltas are nonzero, and the topographic gradient is finite and nonzero, so this looks like a scale-calibration issue rather than an obvious bug.

STEP 6C full training is methodologically ready after choosing a calibrated lambda. Do not use the scalar-loss 1% lambda directly as the first run; use **lambda_topo=0.5000** first, then sweep upward/downward after checking stability and physical metrics.
