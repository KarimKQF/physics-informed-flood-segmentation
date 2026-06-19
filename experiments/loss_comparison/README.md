# Loss Comparison

This experiment compares several masked binary segmentation losses on the
compact real Sen1Floods11 subset with aligned DEMs.

It is designed to compare training losses on a compact real subset. It is not a
final performance evaluation.

## Compared Blocks

Block A trains with classical segmentation losses only:

```text
A1 Masked BCE
A2 Masked Dice
A3 Masked BCE + Dice
A4 Masked Focal
A5 Masked Tversky
```

Block B uses the same base losses plus DEM-only Topographic Loss:

```text
total_loss = base_loss + alpha_topo * topographic_loss
```

This does not use buildings and does not create `q_i`.

## Metrics

All segmentation metrics ignore invalid pixels through `valid_mask`. The
Sen1Floods11 mask value `-1` is therefore excluded from losses and metrics.

Reported metrics:

```text
IoU
Dice
F1
Recall
Precision
ViolationRateTopo
train loss
val loss
```

`ViolationRateTopo` is a simple diagnostic metric. It computes a DEM slope map,
sets a high-slope threshold from the valid pixels, and reports the proportion of
valid predicted-water pixels that fall above that slope threshold. It is a
stable comparison metric for these mini-runs, not a definitive hydrological
criterion.

## Run

```powershell
python experiments\loss_comparison\run_loss_comparison.py `
  --manifest D:\urban_runoff_data\manifests\sen1floods11_compact_with_dem_manifest.csv `
  --output-dir experiments\results\loss_comparison `
  --epochs 10 `
  --batch-size 2 `
  --lr 0.001 `
  --alpha-topo 0.1 `
  --max-samples 8 `
  --seed 42
```

Outputs:

```text
experiments/results/loss_comparison/metrics_per_epoch.csv
experiments/results/loss_comparison/summary_results.csv
experiments/results/loss_comparison/loss_comparison_table.md
experiments/results/loss_comparison/config.json
experiments/results/loss_comparison/predictions/
```

## Diagnostics

The first mini-run can look like an all-water baseline: recall close to 1,
precision close to the target positive rate, and `ViolationRateTopo` close to
the selected high-slope fraction. Do not interpret that as evidence for or
against Topographic Loss before checking trivial baselines and threshold
sensitivity.

Evaluate trivial baselines:

```powershell
python experiments\loss_comparison\evaluate_trivial_baselines.py `
  --manifest D:\urban_runoff_data\manifests\sen1floods11_compact_with_dem_manifest.csv `
  --output-dir experiments\results\loss_comparison
```

Run a threshold sweep on saved checkpoints:

```powershell
python experiments\loss_comparison\threshold_sweep.py `
  --manifest D:\urban_runoff_data\manifests\sen1floods11_compact_with_dem_manifest.csv `
  --results-dir experiments\results\loss_comparison `
  --output-dir experiments\results\loss_comparison\threshold_sweep
```

Export visual diagnostics:

```powershell
python experiments\loss_comparison\export_best_worst_predictions.py `
  --manifest D:\urban_runoff_data\manifests\sen1floods11_compact_with_dem_manifest.csv `
  --results-dir experiments\results\loss_comparison `
  --output-dir experiments\results\loss_comparison\predictions_diagnostics
```

Run short diagnostic experiments:

```powershell
python experiments\loss_comparison\run_loss_comparison.py `
  --manifest D:\urban_runoff_data\manifests\sen1floods11_compact_with_dem_manifest.csv `
  --output-dir experiments\results\loss_comparison_debug `
  --epochs 10 `
  --batch-size 2 `
  --lr 0.0003 `
  --alpha-topo 0.1 `
  --max-samples 8 `
  --seed 42 `
  --loss-debug

python experiments\loss_comparison\run_loss_comparison.py `
  --manifest D:\urban_runoff_data\manifests\sen1floods11_compact_with_dem_manifest.csv `
  --output-dir experiments\results\loss_comparison_pos_weight `
  --epochs 10 `
  --batch-size 2 `
  --lr 0.0003 `
  --alpha-topo 0.1 `
  --max-samples 8 `
  --seed 42 `
  --use-pos-weight `
  --loss-debug
```
