# STEP 6C/v3 Low-Data N=50 — Test & Bolivia Inference-Only Evaluation

Generated: 2026-06-23  
Scope: **inference only. No training launched. No raw data modified.
No existing runs overwritten. DEM never used as model input.**

Checkpoint: `E:/.../step6c_v3_low_data_n50_seed42_lambda05_warmup/checkpoints/best_checkpoint.pt`  
Eval script: `scripts/step6c_v3_low_data_n50_eval.py`  
Command:
```
E:/flood_research/venvs/terramind-gpu/Scripts/python.exe \
    scripts/step6c_v3_low_data_n50_eval.py
```
Total elapsed: **40.1 s** (GPU inference only, no training)  
Outputs:
- `E:/.../step6c_v3_low_data_n50_seed42_lambda05_warmup/metrics/step6c_v3_low_data_n50_test_bolivia_eval.json`
- `reports/STEP_6C_LOW_DATA_N50_TEST_BOLIVIA_EVAL_SUMMARY.json`

---

## A. Results Table

### A.1 STEP 6C/v3 N=50 Physics — all three splits

| Split | mIoU | Water IoU | Water F1 | Precision | Recall | Pred water px | Pred water % | Topo viol. frac. | Topo viol. pairs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Valid** | 0.824251 | 0.691409 | 0.817554 | 0.840905 | 0.795464 | 2,116,688 | 10.43 % | 8.36e-04 | 32,317 |
| **Test** | **0.843556** | **0.731002** | **0.844600** | 0.837313 | 0.852014 | 2,611,157 | 12.73 % | 9.69e-04 | 37,290 |
| **Bolivia** | **0.822575** | **0.708192** | **0.829171** | 0.821874 | 0.836600 | 463,014 | 16.15 % | 2.13e-03 | 11,922 |

*(Valid metrics shown for consistency check; best checkpoint selected on validation.)*

### A.2 Baseline 5S-A N=50 Dice-only (collapsed) — all three splits

| Split | mIoU | Water IoU | Water F1 | Pred water px |
|---|---:|---:|---:|---:|
| Valid | 0.444924 | 0.000095 | 0.000189 | 212 |
| Test | 0.438036 | 0.001031 | 0.002059 | 2,645 |
| Bolivia | 0.421426 | 0.001297 | 0.002591 | 590 |

*(All three splits collapsed. Baseline mIoU is essentially `iou_background / 2` — pure all-background accuracy.)*

### A.3 Delta table (physics N=50 − baseline N=50)

| Split | Δ mIoU | Δ Water IoU | Δ Water F1 | Δ pred water px |
|---|---:|---:|---:|---:|
| Valid | **+0.379327** | **+0.691314** | **+0.817365** | +2,116,476 |
| Test | **+0.405520** | **+0.729971** | **+0.842541** | +2,608,512 |
| Bolivia | **+0.401149** | **+0.706895** | **+0.826580** | +462,424 |

The effect is **consistent and large across all three splits**, not confined to the validation split used for checkpoint selection.

---

## B. Full-Data Reference

| Model | Valid mIoU | Test mIoU | Bolivia mIoU |
|---|---:|---:|---:|
| 5S-A full-data (N=251, Dice) | 0.881233 | 0.873051 | 0.843955 |
| 6C/v3 full-data (N=251, Physics) | 0.878494 | 0.872132 | 0.846406 |
| **6C/v3 N=50 Physics** (this eval) | **0.824251** | **0.843556** | **0.822575** |
| 5S-A N=50 Dice (collapsed) | 0.444924 | 0.438036 | 0.421426 |

**Physics N=50 as % of full-data physics performance:**
- Valid: 93.8 % of full-data 6C/v3 valid mIoU
- Test: **96.7 %** of full-data 6C/v3 test mIoU
- Bolivia: **97.2 %** of full-data 6C/v3 Bolivia mIoU

With only 50 training samples, the physics model recovers 94–97 % of
full-data performance. The gap to the full-data baseline (5S-A N=251)
is likewise small: **2.9 pp on test, 2.2 pp on Bolivia**.

---

## C. Topographic Consistency

| | Valid topo viol. frac | Test topo viol. frac | Bolivia topo viol. frac |
|---|---:|---:|---:|
| **Physics N=50** (this eval) | 8.36e-04 | 9.69e-04 | 2.13e-03 |
| Full-data 5S-A (from topographic eval) | 1.611e-03 | 1.592e-03 | 2.530e-03 |

The N=50 physics model has **lower topographic violation fractions** than the
full-data Dice-only baseline on all splits. Even with 5× fewer training
samples, the physics constraint keeps predictions more physically consistent.

---

## D. Guardrails Verification

| Guardrail | Status |
|---|---|
| DEM as model input | **FALSE** — verified from config + runtime guard in `__getitem__` |
| DEM usage | **Topographic metrics only** — `batch["topography"]` used in loss and topo metric computation, never forwarded to model |
| Training launched | FALSE |
| DARN launched | FALSE |
| STURM-Flood launched | FALSE |
| Raw data modified | FALSE |
| Existing runs overwritten | FALSE |
| BN policy | 13 BatchNorm modules in eval mode (confirmed by `batchnorm_eval_modules=13` in all splits) |

The stderr log contained one informational warning:
> `Missing keys in ckpt_path TerraMind_v1_large.pt: []`

This is printed by `build_task` when loading the pretrained TerraMind backbone
and is expected — empty list means **no keys were missing** (all backbone
weights loaded). Fine-tuned checkpoint weights were then loaded via
`load_state_dict(strict=True)`.

---

## E. Consistency Check

The validation metrics from this eval **exactly match** the per-epoch
metrics logged during training at epoch 55 (the best checkpoint):
- `val_mIoU = 0.824251` ✓  (training log: 0.824251)
- `val_iou_water = 0.691409` ✓  (training log: 0.691409)
- `val_water_pred_pixels = 2,116,688` ✓  (training log: 2,116,688)

This confirms the eval pipeline reproduces training-time evaluation exactly.

---

## F. Scientific Interpretation

### Does the physics N=50 model generalise beyond validation?

**Yes, strongly.** The model achieves water IoU = 0.731 on the held-out test
set and water IoU = 0.708 on the Bolivia OOD split — both orders of magnitude
above the collapsed baseline (0.001 and 0.001). The delta on test
(+0.730 water IoU) is actually *larger* than on validation (+0.691), confirming
the effect is not a validation-overfitting artefact.

### Is collapse avoided on test and Bolivia/OOD?

**Yes, definitively.** The baseline collapses to near-zero water prediction
on all three splits (212 water pixels on valid, 2,645 on test, 590 on
Bolivia, out of millions of ground-truth water pixels). The physics model
predicts 2.1 M water pixels on valid, 2.6 M on test, and 463 K on Bolivia.
Bolivia in particular — the most challenging OOD split with entirely
different geography — is handled cleanly by the physics model: mIoU=0.823,
water IoU=0.708.

### What does this mean for the low-data hypothesis (H3)?

The result strongly supports H3: the topographic prior stabilises training
in very low-data regimes on all evaluation splits, not just validation. The
physics model's Bolivia performance (mIoU=0.823) is only 2.2 pp below the
full-data 5S-A baseline (0.844), which used 5× more training data. This
near-parity with full-data results, using 50 samples and a weak topographic
prior, is a scientifically notable finding.

### One-seed caution (unchanged)

This experiment used seed 42 only. The absolute performance gap (physics N=50
vs baseline N=50) is large enough that reversal is implausible, but
statistical significance requires multi-seed replication. The recommended
path remains: N=100 pilot (to establish collapse boundary), then multi-seed
(seeds 0 and 1) for both N=50 and N=100.

---

## G. Updated Complete Comparison Table

*(Incorporating valid metrics from training log, test/Bolivia from this eval.)*

| Model | Train N | Valid mIoU | Valid water IoU | Test mIoU | Test water IoU | Bolivia mIoU | Bolivia water IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5S-A (Dice, full data) | 251 | 0.8812 | 0.7915 | 0.8731 | 0.7802 | 0.8440 | 0.7388 |
| 6C/v3 (Physics, full data) | 251 | 0.8785 | 0.7893 | 0.8721 | 0.7786 | 0.8464 | 0.7392 |
| **6C/v3 (Physics, N=50)** | **50** | **0.8243** | **0.6914** | **0.8436** | **0.7310** | **0.8226** | **0.7082** |
| 5S-A (Dice, N=50, collapsed) | 50 | 0.4449 | 0.0001 | 0.4380 | 0.0010 | 0.4214 | 0.0013 |

---

## H. Next Steps

1. **Update the comparative report** with test/Bolivia metrics (now available).
2. **Launch paired N=100 baseline vs physics** (the next priority — confirms
   whether collapse also occurs at N=100 or whether 50 is a special boundary).
3. **Multi-seed low-data** (seeds 0 and 1 for N=50 and/or N=100) once the
   signal at each N is established.
4. **STURM-Flood as second dataset** for inter-dataset generalisation
   (after Sen1Floods11 consolidation).

---

## I. Files and Provenance

| Path | Content |
|---|---|
| `scripts/step6c_v3_low_data_n50_eval.py` | Eval script (inference only) |
| `E:/.../step6c_v3_low_data_n50_seed42_lambda05_warmup/checkpoints/best_checkpoint.pt` | Best checkpoint (epoch 55) |
| `E:/.../metrics/step6c_v3_low_data_n50_test_bolivia_eval.json` | Full metrics payload |
| `reports/STEP_6C_LOW_DATA_N50_TEST_BOLIVIA_EVAL_SUMMARY.json` | Machine-readable summary |
| `E:/.../logs/step6c_v3_low_data_n50_eval_stdout.log` | Eval stdout |
| `E:/.../logs/step6c_v3_low_data_n50_eval_stderr.log` | Eval stderr (informational only) |

No training was launched. No raw data was modified. No existing runs,
checkpoints, or logs were overwritten.
