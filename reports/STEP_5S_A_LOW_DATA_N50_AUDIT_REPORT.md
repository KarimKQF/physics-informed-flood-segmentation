# STEP 5S-A Low-Data N=50 Audit Report

## Scope

No training was launched. This audit checks whether the failed STEP 5S-A low-data N=50 Dice baseline is explained by a config/wrapper issue or by a real low-data collapse.

## 1. N=50 Manifest Audit

Manifest: `C:/flood_research/repos/physics-informed-flood-segmentation/manifests/terramind_baseline/low_data_seed42/flood_train_low_data_n50_seed42.txt`

- Sample count: 50
- Event/location distribution: `{'Ghana': 8, 'India': 6, 'Mekong': 4, 'Nigeria': 1, 'Pakistan': 5, 'Paraguay': 5, 'Somalia': 3, 'Spain': 3, 'Sri-Lanka': 5, 'USA': 10}`
- No-water samples: 8
- Water-positive samples: 42
- Water percentage mean: 5.529152%
- Water percentage median: 1.307297%
- Water percentage min/max: 0.000000% / 67.575073%
- Total valid water pixels: 724,717
- Total valid pixels: 11,129,121
- Aggregate water fraction over valid pixels: 6.511898%
- Label values seen in selected masks: `['-1', '0', '1']`

Assessment: the subset contains enough water-positive samples for a meaningful pilot: 42/50 tiles are water-positive, with 724,717 valid water pixels. It is low-data, but not accidentally all-background.

## 2. Low-Data Baseline Config Audit

Config: `C:/flood_research/repos/physics-informed-flood-segmentation/configs/low_data/step5s_a_low_data_n50_seed42.yaml`

- Pretrained checkpoint: `E:/flood_research/experiments/terramind_baseline/checkpoints/pretrained/TerraMind_v1_large.pt`
- Backbone: `terramind_v1_large`
- Decoder: `UperNetDecoder`
- Feature indices: `[5, 11, 17, 23]`
- Loss: `dice`
- Ignore index: `-1`
- Seed: `42`
- Train split: `C:/flood_research/repos/physics-informed-flood-segmentation/manifests/terramind_baseline/low_data_seed42/flood_train_low_data_n50_seed42.txt`
- Valid split: `E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/manifests/flood_valid_step5e_filtered.txt`
- Test split: `E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/manifests/flood_test_step5e_filtered.txt`
- Bolivia split: `E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/manifests/flood_bolivia_step5e_filtered.txt`
- Batch size / grad accumulation / effective batch: `2 / 4 / 8`
- LR / weight decay: `2e-05 / 0.0001`
- Scheduler: `torch.optim.lr_scheduler.ReduceLROnPlateau`, mode `max`, factor `0.5`, patience `3`
- Early stopping min epochs / patience: `30 / 15`
- Frozen backbone / decoder: `False / False`
- DEM guardrail: `dem_input=False`, `dem_loaded=False`

## 3. Wrapper / Pipeline Audit

Wrapper: `scripts/step5s_a_low_data_train.py`

The wrapper imports `step5s_a_bs2_accum4_train` and redirects only these runner globals: `RUN_DIR`, `CONFIG_PATH`, `SPLIT_FILES`, `EPOCH_CSV`, `SUMMARY_JSON`, `TRAINING_STATE`, `BEST_CKPT`, and `LAST_CKPT`. It then calls `runner.main()`. The model construction, datamodule construction, Dice loss handler, optimizer, scheduler, BatchNorm policy, evaluation code, and checkpoint code are the original STEP 5S-A pipeline.

Evidence from run log:

- CUDA/GPU used: `True` / `True`
- Train dataset reported as 50: `True`
- Val dataset reported as 86: `True`
- TerraMind-L pretrained checkpoint loaded with no missing keys: `True`
- Early stopping line present: `True`

The config sets `freeze_backbone=false` and `freeze_decoder=false`; the original runner trains `params = [p for p in task.parameters() if p.requires_grad]`. BatchNorm modules are deliberately put in eval mode and their affine parameters frozen, matching the full-data STEP 5S-A policy.

Labels are valid for this pipeline: selected masks contain only `['-1', '0', '1']`, and the config uses `ignore_index=-1`.

No DEM is used in the baseline: model modalities are `['S2L1C', 'S1GRD']`, guardrails set `dem_input=false` and `dem_loaded=false`, and the wrapper never imports or calls the STEP 6C topography dataloader.

## 4. Difference From Full-Data STEP 5S-A Config

Functional diff in `trainer`, `data`, `model`, `optimizer`, and `lr_scheduler` blocks:

- `data.init_args.train_split`: full=`E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/manifests/flood_train_step5e_filtered.txt` low=`C:/flood_research/repos/physics-informed-flood-segmentation/manifests/terramind_baseline/low_data_seed42/flood_train_low_data_n50_seed42.txt`

Interpretation: the only functional training difference is `data.init_args.train_split`. The run directory/name and low-data metadata fields differ as intended. Model, loss, pretrained checkpoint path, optimizer, scheduler, augmentation, batch size, grad accumulation, seed, validation split, and test split match STEP 5S-A.

Additional non-functional metadata present only in the low-data config: `run_tag`, `notes`, `low_data_protocol`, `evaluation_splits`, `comparison_full_data_results`, `runtime_estimate`, plus guardrail annotations for DEM/SegFormer/Mamba/preserve-existing-runs and dataset_policy train count/full_train_reference.

## 5. Run Output Audit

Final predicted water fractions from the best checkpoint:

| Split | Pred water pixels | Valid pixels | Pred water fraction | mIoU | Water IoU |
|---|---:|---:|---:|---:|---:|
| valid | 224 | 20,294,725 | 0.001104% | 0.444924 | 0.000095 |
| test | 2,676 | 20,517,367 | 0.013043% | 0.438036 | 0.001031 |
| bolivia | 611 | 2,867,815 | 0.021305% | 0.421426 | 0.001297 |

Training/validation trace:

- Epochs completed: 30
- Train loss epoch 1 -> epoch 30: 0.604057 -> 0.518688
- Minimum train loss: 0.457393 at epoch 7
- Validation water IoU epoch 1: 0.00009474
- Validation water IoU epochs 2-30 unique values: `[0.0]`
- Validation F1 water epoch 1: `0.0001894693473004416`
- Validation F1 water epochs 2-30 unique values: `['nan']`

The model made a tiny number of water predictions at epoch 1, then validation water IoU stayed exactly zero from epoch 2 onward. The final best checkpoint is epoch 1 because later epochs are all-background on validation.

## Conclusion

Conclusion: **real low-data collapse**.

I found no evidence of a config or wrapper issue. The N=50 manifest is not all-background, the pretrained TerraMind-L checkpoint was loaded, the backbone/decoder were not frozen, labels are compatible with `ignore_index=-1`, and the only functional config difference from full-data STEP 5S-A is the train manifest.

## Recommendation

Launch the paired N=50 physics run next, but interpret it as a stress test against a collapsed Dice-only baseline.

Before interpreting a physics improvement as a general win, compare against this failure mode explicitly: if the physics run predicts water and improves mIoU/topographic metrics, that supports the low-data-prior hypothesis; if it also collapses, the N=50 regime may be too sparse for this recipe without additional balancing or initialization changes.
