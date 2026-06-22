# STEP 5S-A - TerraMind-L UPerNet Corrected Indices Dice Run

Generated at: 2026-06-21T15:19:22+00:00

## Purpose

STEP 5R found that loss parity is mostly satisfied: the official IBM TerraMind Sen1Floods11 configs and our STEP 5I/5O/5P runs all use Dice. The high-confidence unresolved architecture issue is that STEP 5O/5P used UPerNet feature indices [2, 5, 8, 11], while the official config comment marks [5, 11, 17, 23] for the large backbone.

## Configuration

- Model: TerraMind-L pretrained + UPerNet
- Old STEP 5O/5P indices: [2, 5, 8, 11]
- Corrected large-backbone indices: [5, 11, 17, 23]
- Loss: Dice
- ignore_index: -1
- Classes: 2
- Water class index: 1
- Inputs: S2L1C + S1GRD only
- No physics loss, no topographic loss, no DEM input, no DARN, no STURM-Flood
- Config: `C:/Users/Karim/Desktop/flood-segmentation-training/physics-informed-flood-segmentation/configs/step5s_a_terramind_l_upernet_corrected_indices_dice.yaml`
- Run config: `E:/flood_research/experiments/terramind_baseline/runs/step5s_a_terramind_l_upernet_corrected_indices_dice/configs/step5s_a_terramind_l_upernet_corrected_indices_dice.yaml`

## Dataset Policy

- Train: 251
- Valid: 86
- Test: 89
- Bolivia: 15
- Excluded fully invalid tiles: Ghana_234935, Ghana_26376, Ghana_277, Ghana_5079, Ghana_83483
- keep no_water: true
- keep warning_review: true

## Training Recipe

- AdamW lr: 2e-5
- Weight decay: 1e-4
- Precision: 32
- Physical batch size: 1
- Gradient accumulation: 8
- Effective batch size: 8
- D4 augmentation: enabled for training
- BatchNorm policy: eval mode and affine parameters frozen for UPerNet BatchNorm modules
- Mixed precision note: 16-mixed smoke passed, but the first background training attempt produced non-finite epoch losses; fp32 is used for the stable run.
- Scheduler: ReduceLROnPlateau on validation_miou, mode=max, factor=0.5, patience=3
- Early stopping: validation_miou, min_epochs=30, patience=15, max_epochs=80

## Smoke Result

- Status: passed
- Passed: True
- Output shape: [1, 2, 512, 512]
- Loss finite: True
- Backward OK: True
- BatchNorm eval modules: 13
- GPU peak allocated MB: 6213.089

## Training Status

- Status: running
- Log: `E:/flood_research/experiments/terramind_baseline/runs/step5s_a_terramind_l_upernet_corrected_indices_dice/logs/step5s_a_training.log`

- Training started: True
- Training completed: False
- Best epoch: unknown
- Best validation mIoU: unknown

## Decision Notes

- This run should become the new TerraMind-L UPerNet classical baseline only if it improves validation mIoU without degrading test/Bolivia behavior.
- CE+Dice / weighted CE+Dice ablations are still deferred until this corrected-architecture run is reviewed.
- Physics-informed STEP 6C remains blocked pending human validation.

## Guardrails

- Physics-informed training started: false
- DARN started: false
- STURM-Flood training started: false
- Raw data modified: false
- Official split files modified: false

## Next Step

Human validation after completion before deciding STEP 5S-B loss ablations or STEP 6C.
