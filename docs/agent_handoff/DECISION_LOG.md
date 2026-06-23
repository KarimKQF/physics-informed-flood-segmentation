# Decision Log

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

## Timeline

- STEP 5S-A completed and became the best in-domain baseline.
- STEP 6C lambda=0.5 direct collapsed.
- STEP 6C warmup also collapsed under `lambda_topo=0`.
- Collapse is now considered a runner/data-path parity problem, not directly a topographic-loss problem.
- Current requirement: STEP 6C with `lambda_topo=0` must reproduce STEP 5S-A before physics training.
- STEP 6C/v3 physics warmup completed 80/80 epochs without collapse. Best epoch was 54. Final evaluation showed small valid/test declines versus STEP 5S-A but a small Bolivia/OOD gain. STEP 5S-A topographic metrics are still missing and should be computed before deciding STEP 6D.

## Current Gate

No STEP 6D lambda sweep should be launched until STEP 5S-A topographic metrics are computed with the same loss-only DEM evaluation path and reviewed against STEP 6C/v3.
