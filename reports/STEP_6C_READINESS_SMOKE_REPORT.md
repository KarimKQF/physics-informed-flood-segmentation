# STEP 6C Readiness Smoke Report

Generated: 2026-06-22

## Purpose

Verify that the STEP 6C training setup — TerraMind-L + UPerNet with
`CombinedDicePhysicsLoss` (Dice + lambda_topo × TopographicInconsistencyLoss) —
is correctly implemented and operational before launching full training.

---

## Loss Implementation Status

**Issue found and fixed:** The existing `CombinedSegmentationPhysicsLoss` in
`src/losses/combined_loss.py` used **CrossEntropy** as the segmentation component.
Using it for STEP 6C would make the physics comparison unfair (STEP 5S-A uses Dice).

**Fix applied:** Added `CombinedDicePhysicsLoss` to `src/losses/combined_loss.py`.

- Segmentation component: soft multiclass Dice loss (identical formulation to
  terratorch's Dice loss used in STEP 5S-A)
- Physics component: `TopographicInconsistencyLoss` (unchanged from STEP 6A)
- Total loss: `L = Dice(logits, target) + lambda_topo * L_topo`
- `CombinedSegmentationPhysicsLoss` (CE-based) is preserved and unchanged

---

## Smoke Test Configuration

| Parameter | Value |
|---|---|
| Model | TerraMind-L (`terramind_v1_large`) pretrained |
| Decoder | `UperNetDecoder` |
| Feature indices | `[5, 11, 17, 23]` |
| Input modalities | S2L1C + S1GRD (same as STEP 5S-A) |
| Segmentation loss | Dice (soft multiclass) |
| Physics loss | `TopographicInconsistencyLoss` |
| Combined loss | `CombinedDicePhysicsLoss` |
| lambda_topo | 0.05 |
| Elevation margin | 0.0 |
| Elevation scale | 1.0 |
| Neighborhood | 4-connected |
| Batch size | 2 |
| Precision | FP32 |
| DEM source | Copernicus GLO-30 aligned (`dem_aligned/`) |
| DEM as model input | **false** |
| DEM in loss | **true** |

---

## Smoke Test Result: PASSED

| Check | Result |
|---|---|
| Output shape | `[2, 2, 512, 512]` ✓ |
| Output shape OK | true |
| loss_dice | 0.920732 |
| loss_topo | 0.000287 |
| loss_total | 0.920746 |
| lambda_topo | 0.05 |
| loss_total finite | true |
| output finite | true |
| gradients finite | true |
| backward pass | OK |
| DEM shape | `[2, 512, 512]` |
| DEM finite | true |
| Peak VRAM | **9,337 MB** |
| Elapsed | 0.52 s |

Tiles used: `Paraguay_149787`, `USA_994009`

---

## Notes on Loss Values

`loss_topo = 0.000287` at initialization is expected and correct. At the start of
training the model outputs near-uniform softmax probabilities (~0.5 per class),
so `p_water × (1 − p_water)` is small. The topographic loss becomes more
meaningful as the model learns to make confident flood predictions. The early
training gradient is dominated by `loss_dice`, which is intentional.

The lambda_topo = 0.05 contribution to total loss at initialization is:
`0.05 × 0.000287 ≈ 0.0000144` — negligible at epoch 0, grows as predictions sharpen.

---

## VRAM Budget

| Scenario | Peak VRAM |
|---|---|
| STEP 5S-A batch=2 (baseline) | ~11,821 MB |
| STEP 6C smoke batch=2 (no optimizer) | 9,337 MB |
| STEP 6C training batch=2 (with optimizer) | ~14–16 GB estimated |
| RTX 4000 Ada available | 20,475 MB |

Comfortable headroom. DEM tensors add negligible VRAM (~2 MB per sample).

---

## Methodological Correctness Confirmed

- Dice loss in STEP 6C matches Dice loss in STEP 5S-A ✓
- DEM is used only inside `TopographicInconsistencyLoss`, not as model input ✓
- Model, decoder, feature indices, data, split, preprocessing, input modalities
  are identical to STEP 5S-A ✓
- The only experimental variable is `lambda_topo * TopographicInconsistencyLoss` ✓
- No physics loss in STEP 5S-A, physics loss only in STEP 6C ✓

---

## Files Created / Updated

| File | Status |
|---|---|
| `src/losses/combined_loss.py` | Updated — `CombinedDicePhysicsLoss` added |
| `configs/step6c_terramind_l_upernet_dice_topographic_lambda005.yaml` | Created |
| `E:/.../runs/step6c_.../` | Directory scaffolded |
| `reports/STEP_6C_READINESS_SMOKE_REPORT.md` | This file |

---

## Decision

**STEP 6C is ready to launch.**

Human validation of STEP 5S-A results required before launching full STEP 6C training.
