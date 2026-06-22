# STEP 5N Baseline Freeze and Physics-Loss Prep Report

Generated: 2026-06-20T17:42:03

## Summary

STEP 5N consolidated the validated TerraMind classical-loss baselines, froze the best baseline for future physics-informed work, inspected local physical/topographic inputs, and prepared the first physics-loss protocol. No physics-loss training, DARN training, STURM-Flood training, longer classical training, architecture experiment, raw-data mutation, or checkpoint creation was performed.

## Baseline Comparison

| Step | Model | Decoder | Best epoch | Valid mIoU | Valid IoU water | Test mIoU | Test IoU water | Bolivia mIoU | Bolivia IoU water |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5G | TerraMind tiny pretrained + UNetDecoder | UNetDecoder | 5 | 0.827025 | 0.694335 | 0.852610 | 0.744812 | 0.839058 | 0.735087 |
| 5H | TerraMind small pretrained + UNetDecoder | UNetDecoder | 5 | 0.829888 | 0.699929 | 0.858503 | 0.755579 | 0.845861 | 0.746593 |
| 5I | TerraMind base pretrained + UNetDecoder | UNetDecoder | 5 | 0.843346 | 0.722938 | 0.864234 | 0.764726 | 0.861401 | 0.768374 |
| 5K | TerraMind base pretrained + UPerNet | UperNetDecoder | 2 | 0.805948 | 0.656033 | 0.823169 | 0.692449 | 0.830989 | 0.720674 |
| 5M | TerraMind-L pretrained + UPerNet | UperNetDecoder | 5 | 0.807700 | 0.660359 | 0.830899 | 0.707269 | 0.827710 | 0.718633 |

Machine-readable comparison files:

- `E:/flood_research/experiments/terramind_baseline/runs/step5n_baseline_freeze_and_physics_loss_prep/metrics/baseline_model_comparison.csv`
- `E:/flood_research/experiments/terramind_baseline/runs/step5n_baseline_freeze_and_physics_loss_prep/metrics/baseline_model_comparison.json`

## Frozen Baseline

- Selected baseline: TerraMind base pretrained + UNetDecoder
- Source step: 5I
- Run path: `E:/flood_research/experiments/terramind_baseline/runs/step5i_base_unetdecoder_pretrained`
- Config path: `E:/flood_research/experiments/terramind_baseline/runs/step5i_base_unetdecoder_pretrained/configs/terramind_v1_base_unetdecoder_pretrained_train.yaml`
- Best checkpoint: `E:/flood_research/experiments/terramind_baseline/runs/step5i_base_unetdecoder_pretrained/checkpoints/best_checkpoint.pt`
- Best checkpoint size: 1098.04 MB
- Pretrained checkpoint: `E:/flood_research/experiments/terramind_baseline/checkpoints/pretrained/TerraMind_v1_base.pt`

Validation/test/Bolivia metrics:

- Valid: mIoU 0.843346, IoU water 0.722938, F1 water 0.839192
- Test: mIoU 0.864234, IoU water 0.764726, F1 water 0.866680
- Bolivia: mIoU 0.861401, IoU water 0.768374, F1 water 0.869017

Justification:

- STEP 5I has the highest validation mIoU among tested configs.
- STEP 5I has the highest test mIoU in this comparison set, satisfying the highest or near-highest test mIoU requirement.
- STEP 5I has the strongest overall balance across valid/test/Bolivia.
- UPerNet variants underperformed the UNetDecoder baseline.
- TerraMind-L + UPerNet was feasible but did not outperform base UNetDecoder.

## Topographic and Physical Input Availability

- DEM/SRTM/elevation: not found locally.
- HAND as Height Above Nearest Drainage: not found locally. Sen1Floods11 `HandLabeled` naming is human-label terminology, not hydrologic HAND.
- Slope/flow direction: not found locally.
- Water-distance maps: not found locally.
- Permanent-water masks: available through Sen1Floods11 JRC water products.
- Geospatial metadata: available in Sen1Floods11 GeoTIFFs; sample LabelHand/S1Hand/S2Hand/JRCWaterHand grids are aligned in the sample metadata check.

Permanent-water details:

- JRCWaterHand file count: 446
- JRCPerm file count: 814
- S1Perm file count: 814
- Hand-labeled sample linkage: 446 hand-labeled Sen1Floods11 samples have matching LabelHand/S1Hand/S2Hand/JRCWaterHand IDs out of 446 labels.

Blocker: every sample cannot yet be linked to DEM/HAND/topographic rasters. Topographic physics loss is not training-ready until aligned DEM/HAND or derived slope/flow features are prepared as derived artifacts.

## Proposed Loss Formulation

The first future objective is:

```text
L_total = L_CE + lambda_phys L_topo
```

The detailed candidate losses are documented in:

- `E:/flood_research/experiments/terramind_baseline/runs/step5n_baseline_freeze_and_physics_loss_prep/physics_loss_design/physics_loss_formalization.md`

## First Experiment Plan

Future first physics-loss experiment, after human validation:

- Target STEP 5I TerraMind base pretrained + UNetDecoder.
- Same train/valid/test/Bolivia policy, model, checkpoint, optimizer, batch size, precision, and 5-epoch count.
- Only change: add topographic physics term.
- Lambda sweep: 0.01, 0.05, 0.1.

Plan path:

- `E:/flood_research/experiments/terramind_baseline/runs/step5n_baseline_freeze_and_physics_loss_prep/physics_loss_design/first_physics_experiment_plan.md`

## Success Criteria

Future physics-loss runs must preserve or improve water IoU, water F1, mIoU, Bolivia/OOD generalization, and qualitative physical consistency. Accuracy alone is not sufficient because non-water dominates.

Criteria path:

- `E:/flood_research/experiments/terramind_baseline/runs/step5n_baseline_freeze_and_physics_loss_prep/physics_loss_design/success_criteria.md`

## Blockers

- No local DEM/SRTM/HAND/slope/flow/elevation rasters are currently available for Sen1Floods11 sample linkage.
- A topographic feature preparation workflow must be validated before any physics-loss training.
- Raw data must remain unchanged; any derived topographic features should be written to a separate derived-feature or experiment directory.

## Recommended Next Step

Wait for human validation before implementing the first physics-informed topographic loss.
