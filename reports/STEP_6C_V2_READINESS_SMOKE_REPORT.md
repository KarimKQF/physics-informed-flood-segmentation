# STEP 6C v2 Readiness Smoke Report

Generated: 2026-06-22
Scope: smoke-test only. **No full training launched. No raw data modified. DEM
loss-only. No run-training artifacts created** (smoke output isolated under a
dedicated `step6c_v2_readiness_smoke/` directory).

Script: `scripts/step6c_v2_readiness_smoke.py`
Results JSON:
`E:/flood_research/experiments/terramind_baseline/runs/step6c_v2_readiness_smoke/metrics/smoke_results.json`

Configs under test:
1. `configs/step6c_v2_terramind_l_upernet_dice_topographic_lambda01.yaml`
2. `configs/step6c_v2_terramind_l_upernet_dice_topographic_lambda05_warmup.yaml`
3. `configs/step6c_secondary_finetune_from_5s_a_best_lambda01.yaml` (secondary)

Device: CUDA (RTX 4000 Ada). Precision FP32.

---

## 1. Lambda schedule verification — PASS

| Epoch | warmup λ | expected |
|---:|---:|---|
| 1 | 0.0 | 0 (warmup) ✅ |
| 2 | 0.0 | 0 ✅ |
| 5 | 0.0 | 0 (last warmup epoch) ✅ |
| 6 | 0.03333 | 0 < λ < 0.5 (ramp start) ✅ |
| 7 | 0.06667 | ramp ✅ |
| 13 | 0.26667 | 0 < λ < 0.5 ✅ |
| 20 | 0.5 | 0.5 (ramp end) ✅ |
| 21 | 0.5 | 0.5 (held) ✅ |
| 40 | 0.5 | 0.5 (held) ✅ |

Constant config λ = 0.1 at epochs {1, 5, 20, 80}. ✅

**All schedule checks passed** (`all_passed: true`). The warmup config satisfies the
requirement exactly: **λ = 0 at epoch 1, strictly between 0 and 0.5 during the ramp,
and exactly 0.5 after the ramp.**

---

## 2. Forward / backward / finite-gradient smoke — PASS

One real train batch + aligned DEM, manual DEM-aligned D4 path, FP32.

### λ=0.1 constant config (epoch 1)
| Quantity | Value |
|---|---:|
| loss_dice | 0.920769 |
| loss_topo | 0.000308 |
| loss_total | 0.920799 |
| total finite | ✅ |
| grads finite | ✅ |
| grad L2 | 2.142 |

### λ=0.5 warmup config
| Epoch | λ | loss_dice | loss_topo | loss_total | total==dice (λ=0) | total finite | grads finite | grad L2 |
|---:|---:|---:|---:|---:|:--:|:--:|:--:|---:|
| 1 | 0.0 | 0.516659 | 0.000182 | 0.516659 | **✅ exact** | ✅ | ✅ | 3.312 |
| 20 | 0.5 | 0.912543 | 0.000454 | 0.912770 | n/a | ✅ | ✅ | 2.181 |

Key check: **during warmup (λ=0) `loss_total` equals `loss_dice` exactly** → the model
trains as pure Dice (identical to STEP 5S-A) during the protected early epochs, then the
topographic term is introduced.

> Note on "water fraction before/after one step": the smoke shares a single model across
> all single-step probes, and a first bias-corrected AdamW step on a fresh init is large
> and non-representative, so those numbers bounce (e.g. 0.995→0.003 in one probe) and are
> **not** a training-dynamics signal. They only confirm the model parameters do update
> (non-zero param movement) — in contrast to the failed checkpoint, which had **zero**
> gradients and zero movement. Real dynamics are governed by accumulated mini-batches
> over a full epoch, addressed by the warmup schedule.

---

## 3. Secondary fine-tune warm-start — PASS

| Quantity | Value |
|---|---|
| warm-start checkpoint | STEP 5S-A `best_checkpoint.pt` |
| loaded (strict) | ✅ |
| pred water fraction from 5S-A init | 0.0773 (true ≈ 0.0801) |
| non-degenerate | ✅ |

The secondary config correctly warm-starts from a healthy, non-degenerate model
(predicts sensible water), so the topographic prior will refine rather than collapse.

---

## 4. Readiness verdict

| Check | Status |
|---|:--:|
| Lambda schedule (warmup + constant) | ✅ PASS |
| Dice parity (λ=0 ⇒ total == dice exactly) | ✅ PASS |
| Forward + backward, finite loss & grads (both primary configs) | ✅ PASS |
| Peak behaviour finite at λ=0.5 (warmup ep20) | ✅ PASS |
| Secondary warm-start loads & non-degenerate | ✅ PASS |

**STEP 6C v2 is smoke-clean and ready.** Recommended launch (only on explicit
instruction): the **warmup** config; conservative fallback is the **λ=0.1 constant**
config. Do not launch full training without explicit instruction.
