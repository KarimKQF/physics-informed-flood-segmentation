# STEP 6C lambda=0.5 Collapse — Diagnostic & Optimization Report

Generated: 2026-06-22
Scope: diagnose the all-background collapse of the STEP 6C `lambda_topo=0.5` run,
fix the Dice parity gap, and prepare a corrected STEP 6C v2. **No full training was
launched. No raw data modified. DEM used only inside the loss. The failed run's
logs/checkpoints/configs were preserved.**

Machine-readable diagnostics:
`E:/flood_research/experiments/terramind_baseline/runs/step6c_terramind_l_upernet_dice_topographic_lambda05/metrics/collapse_diagnostics.json`

---

## 0. TL;DR

* The failed run **collapsed to all-background (zero water) during epoch 1 and could
  never recover**, because it fell into a **gradient-dead absorbing state**
  (`p_water = 0` everywhere → saturated softmax → **all gradients exactly 0**).
* The collapse was **caused by the topographic prior added during the fragile early
  epochs**, not by the Dice implementation. STEP 5S-A (Dice-only) passes through the
  *same* near-collapsed point at epoch 1 but keeps a tiny water seed and recovers;
  the extra drying pressure from the topographic term erased that seed.
* The topographic loss `p_high·(1−p_low)` has **two trivial degenerate minima**
  (all-dry and all-wet). The all-dry minimum **coincides with the Dice majority-class
  basin and is an absorbing state**, so the combined loss is biased toward it early.
* A **Dice parity gap was found and fixed**: the failed runner used a hand-rolled
  Dice (`smooth=1.0`, no empty-class masking) that differs from STEP 5S-A's
  `smp.losses.DiceLoss("multiclass")` by ~1e-5 on dense batches (more on no-water
  batches). `CombinedDicePhysicsLoss` now uses `smp.losses.DiceLoss` directly →
  **bit-exact (abs diff = 0.0)**. This was *not* the cause of the collapse but is
  required for a clean fair comparison.
* **Recommendation: relaunch as STEP 6C v2 with a lambda warmup (λ=0 for epochs 1–5,
  linear ramp 0→0.5 over epochs 6–20, then 0.5).** Conservative `λ=0.1` constant is
  the fallback. Both use the parity-fixed Dice. **Safe to launch on explicit
  instruction only.**

---

## 0.1 CRITICAL UPDATE (2026-06-22, after launching the warmup run) — diagnosis revised

The recommended **warmup** run was launched and **collapsed at epoch 2 while
`lambda_topo` was still 0** (warmup keeps λ=0 for epochs 1–5). The built-in
gradient-dead guard caught it and stopped the run (exit 4); `last_checkpoint.pt` was
preserved.

```
epoch=1 lambda_topo=0.00000 train_loss_dice=0.548533 val_iou_water=0.000000
        val_water_pred_pixels=0 epoch_max_grad_norm=4.91e+01      (healthy grads, 0 water)
epoch=2 lambda_topo=0.00000 train_loss_dice=0.523479 val_iou_water=0.000000
        val_water_pred_pixels=0 epoch_max_grad_norm=0.00e+00  -> GRADIENT-DEAD, STOP
```

**Revised conclusion:**

* The collapse occurred under **pure Dice** (`lambda_topo = 0`, `λ·train_loss_topo = 0`).
* **Therefore the topographic loss is NOT the direct cause of the collapse.** The
  earlier Section-6/Section-7 attribution to the topographic prior was *confounded*:
  the failed `λ=0.5` run would very likely have collapsed at `λ=0` too.
* The remaining systematic difference between the **non-collapsing STEP 5S-A** runner
  and the **collapsing STEP 6C/v2** runner (when both are pure Dice) is the
  **runner / data / augmentation / RNG path** — specifically the **manual D4**
  (`torch.rot90/flip` in the train loop, independent `random.Random` RNG) vs 5S-A's
  **`albumentations.D4`** in the dataloader. The Dice implementation is now bit-exact,
  so that is excluded.
* Epoch 1 is a **fragile knife-edge**: 5S-A retains a tiny water seed
  (`val_iou_water = 0.0027`) and recovers; the v2/6C path lands at **exactly 0** and
  dies. The augmentation/RNG trajectory decides which side of the edge the run falls on.

**New gating requirement (supersedes the earlier recommendation):**
`STEP 6C/v2 with lambda_topo = 0 must first reproduce STEP 5S-A's non-collapse before
any physics (λ>0) training is meaningful.` The Dice-parity fix and the λ-warmup were
both correct and are retained, but they do **not** address this, because the failure
is not in the loss. The next actions are: (A) an A/B microtest isolating manual D4 vs
`albumentations.D4` at λ=0 (`reports/STEP_6C_V2_D4_AB_MICROTEST_REPORT.md`), then
(B) align the v2 data path to 5S-A (shared `albumentations.D4` for image+mask+DEM),
then (C) a v2 pure-Dice parity mini-run
(`reports/STEP_6C_V2_PURE_DICE_PARITY_AFTER_D4_FIX_REPORT.md`). LR warmup is kept only
as a fallback if data-path parity alone does not fix it.

---

## 1. Failed run process status (task 7.1)

| Item | Value |
|---|---|
| Parent PID 12444 | Not present (already exited) |
| Python child PID 11548 | Was still running, **stalled (CPU≈0.02)** at/after epoch 6–7 |
| Action taken | `Stop-Process -Id 11548 -Force` → confirmed **STOPPED** |
| Remaining python processes | none |
| Log preserved | Yes (`logs/step6c_lambda05_training.log`, untouched) |
| Checkpoints preserved | Yes (`best_checkpoint.pt` = epoch 1; `last_checkpoint.pt` = epoch 7) |

The command line confirmed PID 11548 was
`scripts\step6c_lambda05_train.py --config ...lambda05.yaml`.

---

## 2. Prediction-distribution diagnostics (task 7.3)

Computed on one real train batch and one real val batch.

### Fresh original-TerraMind init (before training)
| Split | true water frac | pred water frac (argmax) | mean p_water | median p_water | IoU water | mIoU |
|---|---:|---:|---:|---:|---:|---:|
| train | 0.0801 | **0.9984** | 0.9984 | 1.0000 | 0.0793 | 0.0401 |
| valid | 0.1954 | **0.9998** | 0.9998 | 1.0000 | 0.1953 | 0.0976 |

→ The pretrained initialization is **saturated toward WATER everywhere** (consistent
with the earlier calibration's `mean p_water ≈ 0.988`).

### Failed-6C collapsed checkpoint (`last_checkpoint.pt`, epoch 7)
| Split | true water frac | pred water frac (argmax) | mean p_water | min/max p_water | IoU bg | IoU water | mIoU | f1 water |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 0.0801 | **0.0000** | **0.0000** | 0.0 / 0.0 | 0.9199 | **0.0000** | 0.4599 | n/a |
| valid | 0.1954 | **0.0000** | **0.0000** | 0.0 / 0.0 | 0.8046 | **0.0000** | 0.4023 | n/a (NaN) |

* Predicted-class histogram (valid): `[495373, 0]` — **every valid pixel predicted
  background, zero water pixels.**
* `p_water` is **exactly 0.0** (min = max = 0) → the softmax is hard-saturated.

**Collapse to all-background / no-water is confirmed.** The `val_f1_water = nan` and
`val_iou_water = 0` and byte-identical validation metrics across epochs in the log are
fully explained by this saturated all-background state.

---

## 3. Loss parity with STEP 5S-A (task 7.4)

STEP 5S-A computes its loss as
`task.train_loss_handler.compute_loss(model_out, target, task.criterion, ...)`, where
`task.criterion` is built by terratorch from `loss: dice` as
**`smp.losses.DiceLoss("multiclass", ignore_index=-1)`** (defaults `smooth=0.0`,
`eps=1e-7`, plus zeroing of classes absent from the batch).

The failed STEP 6C used a **hand-rolled** `_soft_dice_loss` (`smooth=1.0`, **no**
empty-class masking) inside `CombinedDicePhysicsLoss`.

Measured on the same real batches (fresh init):

| Quantity | train | valid |
|---|---:|---:|
| terratorch dice (5S-A path) | 0.92563820 | 0.83664298 |
| smp DiceLoss (direct) | 0.92563820 | 0.83664298 |
| hand-rolled `_soft_dice_loss` (smooth=1) | 0.92563641 | 0.83664119 |
| **abs diff smp vs terratorch** | **0.0** | **0.0** |
| abs diff hand-rolled vs terratorch | 1.79e-6 | 1.79e-6 |

On the collapsed checkpoint the hand-rolled vs terratorch gap was ~1.2e-5.

### Verdict & fix
* The Dice gap was **real but tiny on dense batches** (~1e-5), driven by `smooth`
  (1.0 vs 0.0) and the missing empty-class masking. **It was not the cause of the
  collapse** (both losses give ~0.52 at the collapsed state, ~0.92 at init).
* Still, exact parity is required for the methodology. **Fixed**: `CombinedDicePhysicsLoss`
  now constructs `smp.losses.DiceLoss(mode="multiclass", ignore_index=ignore_index,
  smooth=dice_smooth)` with `dice_smooth` defaulting to `0.0`.
* **Verified bit-exact** after the fix (random logits incl. ignore pixels and a
  no-water image): `abs(combined_lambda0.loss_dice − smp_dice) = 0.0`, and
  `loss_total == loss_dice` exactly when `lambda_topo = 0`.

**Dice parity: PASS (exact) after the fix.**

---

## 4. Runner / config parity with STEP 5S-A (task 7.5)

| Aspect | STEP 5S-A | STEP 6C λ=0.5 (failed) | Match? |
|---|---|---|---|
| Backbone | terramind_v1_large | terramind_v1_large | ✅ |
| Decoder | UperNetDecoder (ch 256, pool 1/2/3/6, align_corners) | same | ✅ |
| Feature indices | [5,11,17,23] | [5,11,17,23] | ✅ |
| Init checkpoint | original TerraMind_v1_large.pt | same | ✅ |
| Optimizer / LR / WD | AdamW 2e-5 / 1e-4 | same | ✅ |
| Scheduler | ReduceLROnPlateau max 0.5 p3 | same | ✅ |
| batch_size / accum / eff | 2 / 4 / 8 | 2 / 4 / 8 | ✅ |
| Precision | FP32 | FP32 | ✅ |
| BN eval policy | eval+frozen, 13 modules | eval+frozen, 13 modules | ✅ |
| Split manifests | step5e filtered (251/86/89/15) | same | ✅ |
| ignore_index / water=1 | -1 / 1 | -1 / 1 | ✅ |
| Metric computation | identical confusion/IoU code | identical (reused) | ✅ |
| **Segmentation loss** | smp DiceLoss (terratorch) | **hand-rolled `_soft_dice_loss`** | ⚠️ ~1e-5 gap → **fixed** |
| **D4 augmentation** | `albumentations.D4` (dataloader) | **manual `torch.rot90/flip` on img+mask+DEM** | ⚠️ different impl |
| Topographic loss | none | Dice + 0.5·L_topo | (the experimental variable) |

Two intended/known divergences beyond the physics term:

1. **Dice implementation** — now fixed to exact parity (Section 3).
2. **D4 augmentation** — STEP 6C had to apply D4 manually because the DEM must be
   transformed *together* with image+mask (the dataloader's `albumentations.D4`
   cannot see the DEM). **The manual D4 is applied consistently to image, mask, and
   DEM per sample** (verified in `apply_d4_to_batch`), so labels and images stay
   aligned with the DEM — it is *correct*, but it is a different RNG/implementation
   than 5S-A and therefore yields a different (still valid) augmentation trajectory.
   This contributes to trajectory divergence at epoch 1 but is not itself a bug.

**Runner parity: PASS** on all scientific knobs; the only true defect (Dice) is fixed,
and the D4 difference is correct-but-different (kept, because DEM-aligned D4 is required).

---

## 5. Optimizer & gradient diagnostics (task 7.6)

| Quantity | Value |
|---|---:|
| Total parameters | 321,007,362 |
| Trainable parameters | 321,000,194 |
| Trainable note | BN affine frozen (13 BN modules in eval), rest trainable |

At the **collapsed checkpoint** on a real train batch:

| Gradient (L2 over all params) | Value |
|---|---:|
| after Dice-only backward | **0.0** |
| after topographic-only backward | **0.0** |
| after total-loss backward | **0.0** |
| param L2 delta after one AdamW step | **0.0** |
| model updates? | **NO** |

→ **The collapsed state is a perfectly gradient-dead absorbing state.** With
`p_water = 0` exactly everywhere, the softmax Jacobian `p·(1−p) = 0`, so *both* the
Dice and the topographic gradients vanish identically and the optimizer cannot move
the model. This is why the validation metrics were byte-identical from epoch 1 to 6 —
the run was frozen, not slowly improving.

(For contrast, at a healthy unsaturated state the gradients are large and finite — see
the smoke test, where `grad_l2 ≈ 2–3` on a fresh batch.)

---

## 6. Topographic-loss degeneracy analysis (task 7.7)

`L_topo = mean_{descending pairs} w_ij · p_i(water) · (1 − p_j(water))`.

### Degenerate minima
| State | L_topo | Notes |
|---|---|---|
| **All dry** (`p_water=0`) | **0** | `p_high = 0`. **Coincides with the Dice majority-class (all-background) basin and is a gradient-dead absorbing state.** |
| **All wet** (`p_water=1`) | **0** | `1 − p_low = 0`. *Opposed* by Dice (Dice is high here, ~0.9), so Dice keeps pushing away from it. |
| Physically-consistent fill | 0 | monotone water in low areas — the intended minimum. |

The loss is **symmetric** in form, but the two trivial minima are **not symmetric in
the combined landscape**: the all-dry minimum is reinforced by Dice and is absorbing;
the all-wet minimum is repelled by Dice. So the combined `Dice + λ·L_topo` is biased
toward all-dry, especially early when the model is far from a real flood map.

### Does `p_high·(1−p_low)` reduce water early in training? — **Yes.**
Synthetic probe at an *unsaturated* state (`p_water = 0.5` everywhere, real DEM):

| Quantity | Value |
|---|---:|
| L_topo value (unsaturated) | **0.0481** (≈1000× larger than at the saturated init!) |
| mean ∂L_topo/∂(water logit) on **high** pixels | **+3.03e-7** (>0 → GD **suppresses** water) |
| mean ∂L_topo/∂(water logit) on **low** pixels | **−2.82e-7** (<0 → GD **encourages** water) |
| fraction of high pixels pushed drier | **99.4%** |
| fraction of low pixels pushed wetter | **99.4%** |

So whenever the model is in the active (unsaturated) regime, the topographic term
**systematically dries high-elevation pixels**. Combined with Dice's pull toward the
majority background class during the fragile epoch-1 transition, the model **overshoots
past the correct solution into the all-dry absorbing state** and gets stuck.

### Why the earlier calibration under-estimated the danger
The calibration measured `L_topo ≈ 1e-5` and picked λ=0.5 — but it measured at the
**saturated all-water init**, where `(1−p_low) ≈ 0` makes L_topo artificially tiny.
The synthetic probe shows L_topo is ~**0.048** mid-training — about **1000× larger**.
The "λ=0.5 gives 25% scaled-grad ratio" figure was therefore taken at a misleading
operating point; λ applied during the *active* regime carries far more weight than the
calibration implied. This is the key reason a "gradient-aware" λ=0.5 still collapsed.

### Safer strategies (assessment)
| Option | Verdict |
|---|---|
| **A. λ=0.1 constant** | Conservative; reduces but does not eliminate early drying pressure. Good fallback. |
| **B. λ warmup (0 early → ramp → 0.5)** | **Recommended.** Directly removes topo pressure during the fragile epochs; the model first learns real water (5S-A reaches IoU_water>0.6 by ep~4), then the prior refines. Preserves the scientific objective. |
| C. Gate by `val_iou_water > 0.4` | Most robust, slightly more complex; supported in spirit by warmup. Optional upgrade. |
| D. Secondary fine-tune from 5S-A best | Scientifically useful (physical-consistency / OOD), but **not** the from-scratch fair comparison. Provided as a clearly-marked secondary config. |
| E. Re-formulate loss (normalize/gate) | **Not needed** — the degeneracy is real but is fully managed by scheduling. Avoid changing the core formula to keep the science clean. |

---

## 7. Early-behavior comparison: 5S-A vs failed 6C (task 7.8)

| Epoch | 5S-A val_mIoU | 5S-A val_IoU_water | 5S-A val_F1_water | 6C val_mIoU | 6C val_IoU_water | 6C val_F1_water |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.446341 | **0.002721** | 0.005427 | 0.444872 | **0.000000** | NaN |
| 2 | 0.758833 | 0.586468 | 0.739338 | 0.444872 | 0.000000 | NaN |
| 3 | 0.774414 | 0.615550 | 0.762031 | 0.444872 | 0.000000 | NaN |
| 4 | 0.806484 | 0.662786 | 0.797200 | 0.444872 | 0.000000 | NaN |
| 5 | 0.819183 | 0.684824 | 0.812932 | 0.444872 | 0.000000 | NaN |
| 6 | 0.818129 | 0.679043 | 0.808845 | 0.444872 | 0.000000 | NaN |

**Isolated difference:** Both runs reach the *same* near-collapsed point at epoch 1.
5S-A retains a **tiny water seed (IoU_water = 0.0027, F1 = 0.0054)** and explodes to
IoU_water = 0.586 at epoch 2. 6C reaches **exactly 0.0000** water at epoch 1 — the seed
is gone — and is then trapped in the gradient-dead absorbing state forever. The **only
new force that pushes the epoch-1 seed from 0.0027 to 0.0000 is the topographic drying
pressure** (Section 6). That is the collapse mechanism.

---

## 8. Fixes implemented (tasks 7.4, 7.9)

1. **Dice parity fix** — `src/losses/combined_loss.py`:
   * `CombinedDicePhysicsLoss` now uses `smp.losses.DiceLoss(mode="multiclass",
     ignore_index=ignore_index, smooth=dice_smooth)` with `dice_smooth` default `0.0`
     → **bit-exact with STEP 5S-A**.
   * Added `set_lambda_topo()` for epoch-wise schedules.
   * Legacy `_soft_dice_loss` retained (documented as non-parity) for old imports.
2. **New config-driven v2 runner** — `scripts/step6c_v2_train.py`:
   * Identical recipe to 5S-A/6C (reuses the proven data/metric/DEM/manual-D4 helpers).
   * **Epoch-wise lambda schedule** (`constant` or `warmup_linear`) via
     `lambda_for_epoch()`; sets `criterion.set_lambda_topo()` each epoch and logs it.
   * **Optional weight warm-start** (`initialization.from_checkpoint`) for the
     secondary experiment, model-weights-only, `strict=True`.
   * Config-driven `run_dir`/`run_tag` (never touches the failed run dir);
     refuses to overwrite existing artifacts unless `--resume`.
   * DEM-as-model-input guardrail enforced.

The failed-run script `scripts/step6c_lambda05_train.py` was **not modified**.

### Config paths created (task 7.9; not launched)
1. `configs/step6c_v2_terramind_l_upernet_dice_topographic_lambda01.yaml`
   (λ=0.1 constant, parity-fixed Dice).
2. `configs/step6c_v2_terramind_l_upernet_dice_topographic_lambda05_warmup.yaml`
   (λ schedule: epochs 1–5 = 0; 6–20 linear 0→0.5; 21+ = 0.5).
3. `configs/step6c_secondary_finetune_from_5s_a_best_lambda01.yaml`
   (**secondary**: warm-start from 5S-A best, lr=5e-6, λ=0.1 constant).

---

## 9. Smoke tests (task 7.10)

See `reports/STEP_6C_V2_READINESS_SMOKE_REPORT.md` and
`E:/.../runs/step6c_v2_readiness_smoke/metrics/smoke_results.json`. Summary:

* **Lambda schedule** — all checks PASS: warmup = {1:0, 5:0, 6:0.0333, 13:0.2667,
  20:0.5, 21:0.5, 40:0.5}; constant = 0.1 at all epochs.
* **lambda01 & warmup forward/backward** — losses + grads finite; at warmup epoch 1
  (λ=0) `loss_total == loss_dice` exactly (pure-Dice parity during warmup); at warmup
  epoch 20 (λ=0.5) loss/grads finite.
* **Secondary** — 5S-A best warm-start loads (strict) and predicts a non-degenerate
  water fraction (0.077 ≈ true 0.08).

---

## 10. Final recommendation (task 7.11) & answers (task 8)

* **Failed 6C process stopped?** Yes — PID 11548 force-stopped; PID 12444 already gone;
  log/checkpoints preserved.
* **Prediction collapse confirmed?** Yes — all-background, `p_water=0` exactly,
  `iou_water=0`, gradient-dead absorbing state.
* **Dice parity passed?** Yes — now **bit-exact** after switching to `smp.losses.DiceLoss`
  (the hand-rolled version differed by ~1e-5; not the cause, but fixed).
* **Runner parity passed?** Yes on all scientific knobs; the only true defect (Dice) is
  fixed; the manual DEM-aligned D4 is correct-but-different and is retained.
* **Likely cause of collapse:** the topographic prior, applied from epoch 1, added
  systematic drying pressure on high-elevation pixels during the fragile early
  transition (when both 5S-A and 6C briefly approach all-background). It erased the
  epoch-1 water seed that 5S-A used to recover, driving the model into the all-dry
  degenerate minimum, which coincides with the Dice majority-class basin and is a
  gradient-dead absorbing state — hence permanent, unrecoverable collapse.
* **Fixes implemented:** exact Dice parity in `CombinedDicePhysicsLoss`; config-driven
  v2 runner with lambda warmup/schedule + weight warm-start; three configs.
* **Recommended next run:** **STEP 6C v2 warmup** —
  `configs/step6c_v2_terramind_l_upernet_dice_topographic_lambda05_warmup.yaml`.
  Fallback if any instability remains: the conservative
  `..._lambda01.yaml` (λ=0.1 constant). The secondary fine-tune config is for the
  physical-consistency / OOD study, not the primary fair comparison.
* **Exact launch recommendation (when explicitly instructed):**
  ```
  E:/flood_research/venvs/terramind-gpu/Scripts/python.exe \
    scripts/step6c_v2_train.py \
    --config configs/step6c_v2_terramind_l_upernet_dice_topographic_lambda05_warmup.yaml
  ```
  (launch in background; report PID + log path; then stop.)
* **Is it safe to launch full training?** Technically ready and de-risked, **but per
  the standing constraint, do NOT launch without explicit instruction.** This report
  performs diagnosis, fixes, and smoke tests only.
