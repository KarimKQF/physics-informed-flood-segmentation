# STEP 5S-A — Final Report: TerraMind-L + UPerNet Corrected Indices Classical Baseline

Generated: 2026-06-22

## Purpose

STEP 5R identified that STEP 5O/5P used incorrect UPerNet feature indices `[2, 5, 8, 11]`
instead of the correct Large-backbone indices `[5, 11, 17, 23]`. STEP 5S-A reruns
TerraMind-L + UPerNet with the corrected indices under otherwise identical conditions.

This run is the **definitive TerraMind-L classical baseline** for the physics-informed
comparison in STEP 6C.

---

## Model Configuration

| Parameter | Value |
|---|---|
| Backbone | `terramind_v1_large` (TerraMind Large) |
| Decoder | `UperNetDecoder` |
| Feature indices | `[5, 11, 17, 23]` (corrected for Large backbone) |
| Previous wrong indices | `[2, 5, 8, 11]` (STEP 5O / 5P) |
| Input modalities | S2L1C (13 bands) + S1GRD (2 bands) |
| Backbone merge method | mean |
| Pretrained checkpoint | `TerraMind_v1_large.pt` |
| Decoder channels | 256 |
| Head dropout | 0.1 |
| Num classes | 2 (Others, Flood) |
| `ignore_index` | -1 |

---

## Training Recipe

| Parameter | Value |
|---|---|
| Loss | Dice |
| Physics loss | None |
| DEM input | None |
| DARN | Not started |
| STURM-Flood training | Not started |
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Weight decay | 1e-4 |
| Scheduler | ReduceLROnPlateau (mode=max, factor=0.5, patience=3) |
| Batch size (physical) | 2 |
| Gradient accumulation steps | 4 |
| Effective batch size | 8 |
| Precision | FP32 (32) |
| Max epochs | 80 |
| Early stopping monitor | validation_miou |
| Early stopping patience | 15 epochs |
| Early stopping min epochs | 30 |
| Train transform | D4 augmentation + ToTensorV2 |
| Val / test transform | None |
| BatchNorm eval policy | BN modules frozen in eval mode (UPerNet PSP requirement) |
| Seed | 42 |
| GPU | NVIDIA RTX 4000 Ada Generation (20 GB) |

---

## Dataset Protocol

| Split | Count |
|---|---|
| Train | 251 |
| Valid | 86 |
| Test | 89 |
| Bolivia (OOD holdout) | 15 |
| **Total** | **441** |

- Official Sen1Floods11 hand-labeled split
- Excluded fully invalid tiles: `Ghana_234935`, `Ghana_26376`, `Ghana_277`, `Ghana_5079`, `Ghana_83483`
- `keep_warning_review: true`
- `keep_no_water: true`
- `ignore_index: -1` for invalid pixels

---

## Training History (key epochs)

| Epoch | val_mIoU | val_IoU_water | val_F1_water | LR | no_improve |
|---|---|---|---|---|---|
| 1 | 0.7499 | 0.5541 | 0.7131 | 2.00e-5 | 0 |
| 5 | 0.8192 | 0.6848 | 0.8129 | 2.00e-5 | 0 |
| 12 | 0.8371 | 0.7128 | 0.8323 | 2.00e-5 | 0 |
| 16 | 0.8536 | 0.7410 | 0.8513 | 2.00e-5 | 0 |
| 19 | 0.8541 | 0.7424 | 0.8522 | 2.00e-5 | 0 |
| 30 | — | — | — | decaying | — |
| **55** | **0.8868** | **0.8027** | **0.8904** | 6.25e-7 | 0 (prev best) |
| **69** | **0.8812** | **0.7915** | **0.8836** | 7.81e-8 | 0 **(BEST)** |
| 80 | 0.8811 | 0.7913 | 0.8835 | 1.95e-8 | 11 |

Best epoch: **69** (max epochs reached, early stopping not triggered)

---

## Final Metrics (best checkpoint, epoch 69)

### Validation

| Metric | Value |
|---|---|
| **mIoU** | **0.8812** |
| IoU water | 0.7915 |
| IoU background | 0.9710 |
| F1 water | 0.8836 |
| Precision water | 0.8675 |
| Recall water | 0.9004 |
| Accuracy | 0.9738 |

### Test

| Metric | Value |
|---|---|
| **mIoU** | **0.8731** |
| IoU water | 0.7802 |
| IoU background | 0.9659 |
| F1 water | 0.8765 |
| Precision water | 0.8904 |
| Recall water | 0.8631 |
| Accuracy | 0.9696 |

### Bolivia (OOD holdout)

| Metric | Value |
|---|---|
| **mIoU** | **0.8440** |
| IoU water | 0.7388 |
| IoU background | 0.9491 |
| F1 water | 0.8498 |
| Precision water | 0.9150 |
| Recall water | 0.7933 |
| Accuracy | 0.9555 |

---

## Comparison with Previous Baselines

| Run | Model | Decoder | Indices | val_mIoU | test_mIoU | Bolivia_mIoU |
|---|---|---|---|---|---|---|
| STEP 5I | TM-base pretrained | UNetDecoder | — | 0.8433 | 0.8642 | **0.8614** |
| STEP 5O | TM-L pretrained | UPerNet | `[2,5,8,11]` (wrong) | 0.8503 | 0.8526 | 0.8457 |
| **STEP 5S-A** | **TM-L pretrained** | **UPerNet** | **`[5,11,17,23]`** | **0.8812** | **0.8731** | 0.8440 |

**Index correction gain (5S-A vs 5O):**
- Valid: +3.09 pp
- Test: +2.05 pp
- Bolivia: −0.17 pp

---

## Interpretation

**STEP 5S-A is now the best in-domain classical baseline** on both validation and test splits.
The corrected UPerNet feature indices `[5, 11, 17, 23]` for the Large backbone produced a
significant improvement over the wrong indices used in STEP 5O/5P.

**STEP 5I (base + UNetDecoder) remains better on Bolivia/OOD** (0.8614 vs 0.8440).
This is scientifically important: the larger TerraMind-L backbone with UPerNet may be
overfitting more to the in-domain training distribution, while the smaller base model
generalizes better to the out-of-distribution Bolivia holdout.

**This makes STEP 6C scientifically interesting** because:
1. We have a strong in-domain baseline (STEP 5S-A, 0.8812 valid, 0.8731 test)
2. OOD generalization is still below base + UNetDecoder
3. The topographic physics loss may improve physical consistency and potentially
   OOD generalization — since topographic constraints are geography-independent
4. A positive result on Bolivia would strongly support the physics-informed approach

---

## Checkpoint

- Path: `E:/flood_research/experiments/terramind_baseline/runs/step5s_a_terramind_l_upernet_corrected_indices_dice_bs2_accum4/checkpoints/best_checkpoint.pt`
- Version: v2 (atomic save, full state)
- Epoch: 69
- `best_validation_miou`: 0.88123251
- Keys: `step`, `ckpt_version`, `epoch`, `best_validation_miou`, `best_epoch`, `no_improve`, `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `config`, `saved_at`
- Integrity: VERIFIED

---

## Guardrails Confirmed

- Physics-loss training started: **false**
- Topographic loss: **false**
- DEM as model input: **false**
- DARN started: **false**
- STURM-Flood training started: **false**
- Raw data modified: **false**
- Official split files modified: **false**

---

## Next Steps

- **STEP 6C**: Add `lambda_topo * TopographicInconsistencyLoss` to STEP 5S-A setup.
  - Same model, decoder, indices, data, preprocessing, input modalities
  - DEM used only inside the loss (not as model input)
  - lambda_topo = 0.05 (initial sweep value)
  - Human validation of STEP 5S-A results required before launch ← **current gate**
- **STEP 6D** (after 6C): lambda sweep: 0.01, 0.05, 0.1
- **Later**: STEP 5S-B loss ablations (CE+Dice, Weighted CE+Dice) if needed
