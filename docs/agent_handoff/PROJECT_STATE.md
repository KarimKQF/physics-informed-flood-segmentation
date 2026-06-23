# Project State

## Purpose

This file records the stable shared context for the `physics-informed-flood-segmentation` research repository.

Agents communicate indirectly through short structured markdown files committed in this repository.

- Claude: scientific director, protocol designer, research reviewer.
- Codex: implementation engineer, PyTorch developer, repository operator.
- Human user: final decision maker and validation gate.

## Standing Rules

1. Do not launch full training unless explicitly instructed.
2. If a long job is launched, it must run in background, report PID and log path, then stop immediately.
3. Do not monitor logs continuously.
4. Do not use DEM as model input unless explicitly approved.
5. DEM must currently be used only inside the topographic loss.
6. Do not modify raw data.
7. Do not start DARN or STURM-Flood training unless explicitly approved.
8. Preserve all previous logs, configs, metrics, reports, and checkpoints.
9. Every experiment must have a unique run directory.
10. Every implementation task must end with a short report.

## Current Stable Context

- Project: physics-informed flood segmentation.
- Main dataset: Sen1Floods11 filtered protocol.
- Split sizes:
  - train: 251
  - valid: 86
  - test: 89
  - Bolivia/OOD: 15
- Aligned DEM path:
  `E:/flood_research/data/derived/sen1floods11_topography/dem_aligned/`

## Current Best Baseline

STEP 5S-A is the current best in-domain baseline.

Setup:

- Backbone: TerraMind-L / `terramind_v1_large`
- Decoder: `UperNetDecoder`
- Feature indices: `[5, 11, 17, 23]`
- Segmentation loss: Dice
- Batch size: 2
- Gradient accumulation steps: 4
- Effective batch size: 8
- Precision: FP32
- Train augmentation: D4 augmentation
- Physics loss: none
- DEM input: none

Final metrics:

| Split | mIoU | IoU water | F1 water |
|---|---:|---:|---:|
| valid | 0.881233 | 0.791499 | 0.883616 |
| test | 0.873051 | 0.780194 | 0.876527 |
| Bolivia/OOD | 0.843955 | 0.738797 | 0.849779 |

Important note: STEP 5I remains better on Bolivia/OOD, with Bolivia mIoU `0.861401`.
