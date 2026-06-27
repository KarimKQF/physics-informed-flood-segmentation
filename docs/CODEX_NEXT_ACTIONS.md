# Codex Next Actions

Date: 2026-06-26

This note is operational. It says what Codex should do only when asked, and how
to avoid interfering with the active SegMAN N=50 multi-seed chain.

## Default Posture

Do not launch or stop anything by default. The multi-seed chain is expected to
be running sequentially under:

- `scripts/launch_segman_n50_multiseed_chain.ps1`

Seed0 is complete. The active/new chain covers seeds `1`, `2`, `3`, and `42`,
with four variants per seed.

## Only When Asked: Light Status Check

A safe status check should be read-only and brief. Do not tail continuously.

Use one of these read-only checks:

```powershell
Get-Content E:/flood_research/experiments/segman/multiseed_n50_chain.log -Tail 40
```

or inspect completed summaries without touching checkpoints:

```powershell
Get-ChildItem E:/flood_research/experiments/segman/runs/*/metrics/*_summary.json
```

Do not:

- kill or restart a process
- pass `-KillOrphans`
- launch a new chain
- launch an individual training run
- open GPU notebooks
- repeatedly poll logs

## Only When Asked: Aggregate After Completion

After all 16 new runs are complete, aggregate the full seed set:

```powershell
E:/flood_research/venvs/terramind-gpu/Scripts/python.exe experiments_cvpr/segman/aggregate_multiseed_results.py
```

The aggregation script is intended to be CPU-only. It reads summary JSON and
epoch CSV files, skips missing/incomplete runs, and writes:

- `reports/segman_n50_multiseed_results.csv`
- `reports/segman_n50_multiseed_results.json`
- `docs/segman_n50_multiseed_summary.md`

Before interpreting results, verify how many runs were loaded. The target is
20 total runs when including seed0: 5 seeds x 4 variants.

## Reporting Rules

When reporting results:

- include validation, test, and Bolivia/OOD splits
- include all available segmentation, confusion, prediction-ratio, topo, and
  training metrics
- report mean +/- std across seeds where available
- mark unavailable metrics as `not logged` or `not available`
- do not infer or fabricate missing metrics

Key scientific comparison:

- real DEM topo vs Dice+CE
- real DEM topo vs DEM-shuffled topo
- stability of those differences across seeds

## Decisions To Defer

Do not start any of these until explicitly requested after multi-seed analysis:

- lambda sweep over `0.1`, `0.5`, `1.0`, `2.0`
- full-data SegMAN experiments
- official WSL/Linux/CUDA kernel migration
- EoMT integration
- VFMNet/VFMSeg integration
- ViT-P audit or integration
- TerraMind continuation

## If Something Looks Wrong

If a summary is missing, partial, or reports failure, do not repair by deleting
or restarting runs unless the user explicitly asks. Report:

- run tag
- summary status, if present
- last chain-log line for that run, if checked
- whether checkpoint and CSV files appear present
- the safest next read-only diagnostic

