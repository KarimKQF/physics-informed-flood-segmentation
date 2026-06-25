# STEP 6C Low-Data Pilot Preparation Report

## Scope

Prepared only. No training was launched, no raw data was modified, and no existing runs, logs, metrics, reports, or checkpoints were overwritten.

## Scientific Question

The full-data comparison showed that STEP 6C/v3 slightly reduced valid/test segmentation mIoU but improved Bolivia/OOD mIoU and reduced topographic violations by about 7-10 percent. This pilot tests whether the same physics prior becomes more useful when labeled train samples are scarce.

The core comparison is paired by train subset: STEP 5S-A low-data Dice-only versus STEP 6C/v3 low-data Dice plus topographic loss. Both runs use the same train manifest for a given N, and the valid/test/Bolivia manifests are unchanged from the full-data results.

## Prepared Train Subsets

Source train manifest: `E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/manifests/flood_train_step5e_filtered.txt`

Selection algorithm: `random.Random(42).shuffle(full_train_ids)`, then first 100 samples define N=100 and first 50 of that same order define N=50. Manifests are sorted on disk for stable diffs.

| N | Manifest | Notes |
|---:|---|---|
| 50 | `C:/flood_research/repos/physics-informed-flood-segmentation/manifests/terramind_baseline/low_data_seed42/flood_train_low_data_n50_seed42.txt` | Primary low-data pilot; subset of N=100. |
| 100 | `C:/flood_research/repos/physics-informed-flood-segmentation/manifests/terramind_baseline/low_data_seed42/flood_train_low_data_n100_seed42.txt` | Optional secondary pilot; contains the N=50 subset. |

Selection summary: `C:/flood_research/repos/physics-informed-flood-segmentation/manifests/terramind_baseline/low_data_seed42/selection_summary_seed42.json`

Unchanged evaluation manifests:

- Valid: `E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/manifests/flood_valid_step5e_filtered.txt`
- Test: `E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/manifests/flood_test_step5e_filtered.txt`
- Bolivia/OOD: `E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/manifests/flood_bolivia_step5e_filtered.txt`

## Configs Created

| Experiment | Config | Run directory |
|---|---|---|
| 5S-A low-data N=50 Dice baseline | `configs/low_data/step5s_a_low_data_n50_seed42.yaml` | `E:/flood_research/experiments/terramind_baseline/runs/step5s_a_low_data_n50_seed42_dice` |
| 6C/v3 low-data N=50 physics | `configs/low_data/step6c_v3_low_data_n50_seed42_lambda05_warmup.yaml` | `E:/flood_research/experiments/terramind_baseline/runs/step6c_v3_low_data_n50_seed42_lambda05_warmup` |
| 5S-A low-data N=100 Dice baseline | `configs/low_data/step5s_a_low_data_n100_seed42.yaml` | `E:/flood_research/experiments/terramind_baseline/runs/step5s_a_low_data_n100_seed42_dice` |
| 6C/v3 low-data N=100 physics | `configs/low_data/step6c_v3_low_data_n100_seed42_lambda05_warmup.yaml` | `E:/flood_research/experiments/terramind_baseline/runs/step6c_v3_low_data_n100_seed42_lambda05_warmup` |

## Protocol Details

Baseline runs:

- TerraMind-L + UPerNet, corrected Large-backbone feature indices `[5, 11, 17, 23]`.
- Input modalities: S2L1C + S1GRD.
- Dice loss only.
- No DEM loaded, no physics loss, no DARN, no STURM-Flood, no SegFormer, no Mamba.

Physics runs:

- Same model, optimizer, augmentation, batch size, gradient accumulation, validation split, and test splits as STEP 5S-A.
- DEM is loaded by the dataloader only as `batch["topography"]`.
- DEM is never included in `batch["image"]` and is never passed as model input.
- Loss: `smp DiceLoss + lambda_topo * TopographicInconsistencyLoss`.
- Lambda schedule: epochs 1-5 `lambda_topo = 0`, epochs 6-19 linear warmup, epoch 20+ `lambda_topo = 0.5`.

Training hyperparameters preserved from the validated full-data runs:

- `batch_size = 2`
- `gradient_accumulation_steps = 4`
- `effective_batch_size = 8`
- `max_epochs = 80`
- `early_stopping_min_epochs = 30`
- `early_stopping_patience = 15`
- FP32 precision

## Launch Commands

Do not run these until training is explicitly approved.

```powershell
cd C:/flood_research/repos/physics-informed-flood-segmentation; & "E:/flood_research/venvs/terramind-gpu/Scripts/python.exe" "scripts/step5s_a_low_data_train.py" --config "configs/low_data/step5s_a_low_data_n50_seed42.yaml"
cd C:/flood_research/repos/physics-informed-flood-segmentation; & "E:/flood_research/venvs/terramind-gpu/Scripts/python.exe" "scripts/step6c_v3_train.py" --config "configs/low_data/step6c_v3_low_data_n50_seed42_lambda05_warmup.yaml"
cd C:/flood_research/repos/physics-informed-flood-segmentation; & "E:/flood_research/venvs/terramind-gpu/Scripts/python.exe" "scripts/step5s_a_low_data_train.py" --config "configs/low_data/step5s_a_low_data_n100_seed42.yaml"
cd C:/flood_research/repos/physics-informed-flood-segmentation; & "E:/flood_research/venvs/terramind-gpu/Scripts/python.exe" "scripts/step6c_v3_train.py" --config "configs/low_data/step6c_v3_low_data_n100_seed42_lambda05_warmup.yaml"
```

## Estimated Runtime

Estimates assume the same RTX 4000 Ada 20 GB environment used for full-data STEP 5S-A and STEP 6C/v3. Validation cost is unchanged, so runtime does not scale linearly with train subset size.

| Experiment | Estimated runtime |
|---|---:|
| 5S-A N=50 | about 1.2-1.5 hours for 80 epochs on RTX 4000 Ada 20 GB |
| 6C/v3 N=50 | about 1.3-1.7 hours for 80 epochs on RTX 4000 Ada 20 GB |
| 5S-A N=100 | about 1.7-2.0 hours for 80 epochs on RTX 4000 Ada 20 GB |
| 6C/v3 N=100 | about 1.8-2.2 hours for 80 epochs on RTX 4000 Ada 20 GB |

## Why This Is Useful

The full-data physics result mostly improved physical consistency and Bolivia/OOD generalization, but did not improve valid/test mIoU. A low-data paired pilot directly tests a stronger claim: when labels are scarce, the topographic prior may compensate for missing supervision by penalizing physically implausible flood predictions.

Because the train subset is identical within each baseline/physics pair and valid/test/Bolivia splits are unchanged, any difference between the paired runs is attributable primarily to the physics loss rather than to split drift or added DEM input information.

## Stop Condition

Preparation is complete. Next action requires explicit approval to launch training.
