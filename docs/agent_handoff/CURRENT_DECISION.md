# Current Decision

## Active Gate

The STEP 6C physics run cannot be launched until the 6C runner reproduces STEP 5S-A behavior under `lambda_topo=0`.

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

## Current Issue

- STEP 6C/v2 collapsed even when `lambda_topo=0`, so the physics loss is not the direct cause.
- The current suspected issue is runner/data-path parity.
- D4 applied inside the training loop/post-collation collapses.
- STEP 5S-A used dataloader-side Albumentations D4 and worked.
- The next implementation objective is to build a STEP 6C/v3 runner that reuses the successful STEP 5S-A dataloader-side D4 path and only adds DEM loading for the loss.

## Decision Requirement

Before any physics-loss training is considered scientifically valid, STEP 6C with `lambda_topo=0` must reproduce STEP 5S-A behavior closely enough to pass human validation.
