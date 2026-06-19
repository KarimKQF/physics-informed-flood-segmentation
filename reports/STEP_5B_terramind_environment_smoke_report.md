# STEP 5B - TerraMind environment setup / smoke tests

## Summary
- Status: `done`
- Generated at: `2026-06-19T17:38:26`
- TerraMind environment created: `true`
- TerraMind imports OK: `true`
- TerraMind config smoke OK: `true`
- TerraMind dataloader smoke OK: `true`
- Model initialization smoke OK: `true`
- Model forward pass performed: `false`
- Full training started: `false`
- DARN/STURM-Flood/physics loss started: `false`
- Raw Sen1Floods11 data modified: `false`
- Next step allowed: `false`

STEP 5B stayed limited to environment setup, dependency verification, official config
inspection, non-destructive data mapping, manifest generation, and minimal smoke tests.

## Installation Status
- Virtual environment: `D:/flood_research/venvs/terramind`
- Python executable: `D:/flood_research/venvs/terramind/Scripts/python.exe`
- Install log: `D:/flood_research/experiments/terramind_baseline/logs/STEP_5B_install.log`
- Installation status: `done`

Note: the first install wrapper exceeded the 30-minute command timeout while pip was in
the `Installing collected packages` phase. The pip process continued and completed; the
venv was then verified with package metadata and import smoke checks.

## Package Versions
| Package | Version |
|---|---:|
| Python | `3.12.10` |
| terratorch | `1.2.8` |
| torch | `2.12.1+cpu` |
| torchvision | `0.27.1` |
| rasterio | `1.5.0` |
| lightning | `2.6.5` |
| pytorch-lightning | `2.6.5` |
| timm | `1.0.27` |
| huggingface_hub | `1.20.1` |
| gdown | `6.1.0` |
| tensorboard | `2.20.0` |
| setuptools | `80.10.2` |
| torchgeo | `0.9.0` |
| segmentation_models_pytorch | `0.5.0` |
| numpy | `2.4.6` |

## CUDA/GPU Status
- CUDA available: `false`
- CUDA device count: `0`
- CUDA device name: not available
- Smoke config was adapted to CPU with batch size `1`.

## Import Checks
| Import | Result | Notes |
|---|---|---|
| torch | `OK` | CPU build, CUDA unavailable |
| rasterio | `OK` | Import took about `2.50 s` |
| lightning | `OK` | Import took about `55.61 s` |
| timm | `OK` | Import took about `9.27 s` |
| huggingface_hub | `OK` | Immediate |
| torchvision | `OK` | Immediate after torch/timm stack loaded |
| torchgeo | `OK` | Immediate after stack loaded |
| segmentation_models_pytorch | `OK` | Import took about `2.23 s` |
| terratorch | `OK` | Import took about `119.25 s` |

TerraTorch import is valid but slow on this CPU-only environment.

## Official TerraMind Files Fetched
Fetched into `D:/flood_research/external/terramind/` without modification.

| File | Local path | Source |
|---|---|---|
| Tiny Sen1Floods11 config | `D:/flood_research/external/terramind/configs/terramind_v1_tiny_sen1floods11.yaml` | https://github.com/IBM/terramind |
| Base Sen1Floods11 config | `D:/flood_research/external/terramind/configs/terramind_v1_base_sen1floods11.yaml` | https://github.com/IBM/terramind |
| Base TiM/LULC Sen1Floods11 config | `D:/flood_research/external/terramind/configs/terramind_v1_base_tim_lulc_sen1floods11.yaml` | https://github.com/IBM/terramind |
| Small Sen1Floods11 notebook | `D:/flood_research/external/terramind/notebooks/terramind_v1_small_sen1floods11.ipynb` | https://github.com/IBM/terramind |

## Experiment Directory Structure
Created or verified:

```text
D:/flood_research/experiments/terramind_baseline/configs/
D:/flood_research/experiments/terramind_baseline/checkpoints/
D:/flood_research/experiments/terramind_baseline/predictions/
D:/flood_research/experiments/terramind_baseline/metrics/
D:/flood_research/experiments/terramind_baseline/logs/
D:/flood_research/experiments/terramind_baseline/reports/
D:/flood_research/experiments/terramind_baseline/manifests/
D:/flood_research/experiments/terramind_baseline/scratch/
D:/flood_research/experiments/terramind_baseline/staging/
```

## Staging / Mapping Method
- Requested staging root: `D:/flood_research/experiments/terramind_baseline/staging/sen1floods11_v1.1/`
- Symlink/junction attempt: `failed`
- Windows error observed: `Fonction incorrecte`
- Likely cause: `D:/` is exFAT, and directory junction/reparse-point support is unavailable for this target.
- Method used: `manifest_direct_raw_path_mapping`
- Raw data modified: `false`
- Large imagery copied: `false`

The adapted TerraMind smoke config points directly to the existing raw directories:

```text
D:/flood_research/data/raw/sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand/
D:/flood_research/data/raw/sen1floods11/v1.1/data/flood_events/HandLabeled/S2Hand/
D:/flood_research/data/raw/sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand/
```

Mapping note:

```text
D:/flood_research/experiments/terramind_baseline/staging/mapping_method.txt
```

## Generated Manifests
Generated from:

```text
C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/sen1floods11_handlabeled_index.csv
```

Policy applied:
- Excluded the five fully invalid `LabelHand` samples.
- Kept `warning_review` samples.
- Kept `no_water` samples.
- Kept Bolivia separate as holdout/OOD.
- Filtering is manifest-based only.

Generated files:

| File | Count / Notes |
|---|---:|
| `D:/flood_research/experiments/terramind_baseline/manifests/flood_train_data.txt` | `251` samples |
| `D:/flood_research/experiments/terramind_baseline/manifests/flood_valid_data.txt` | `86` samples |
| `D:/flood_research/experiments/terramind_baseline/manifests/flood_test_data.txt` | `89` samples |
| `D:/flood_research/experiments/terramind_baseline/manifests/flood_bolivia_data.txt` | `15` samples |
| `D:/flood_research/experiments/terramind_baseline/manifests/terramind_training_manifest.csv` | `446` rows, `5` excluded |

The same `.txt` split files were also mirrored into:

```text
D:/flood_research/experiments/terramind_baseline/staging/sen1floods11_v1.1/splits/
```

## Adapted Smoke Config
Created:

```text
D:/flood_research/experiments/terramind_baseline/configs/terramind_v1_tiny_sen1floods11_smoke.yaml
```

Adaptations:
- Based on official `terramind_v1_tiny_sen1floods11.yaml`.
- Uses `terramind_v1_tiny`.
- Uses `UNetDecoder`, matching the public TerraMind Sen1Floods11 example path.
- Uses absolute local raw data paths.
- Uses generated manifest split files.
- Uses `ignore_index: -1`.
- Uses `batch_size: 1`.
- Uses `num_workers: 0`.
- Uses CPU trainer settings.
- Disables train/val/test transforms for smoke loading.
- Sets `backbone_pretrained: false` to avoid downloading model weights in STEP 5B.

## Dataloader Smoke Test
Smoke summary:

```text
D:/flood_research/experiments/terramind_baseline/reports/STEP_5B_smoke_summary.json
```

Results:
- Config load: `OK`
- `GenericMultiModalDataModule` import: `OK`
- Datamodule init: `OK`
- `setup('fit')`: `OK`
- Train dataset length: `251`
- Validation dataset length: `86`
- One train sample loaded: `OK`

First train sample:

```text
S1GRD: D:/flood_research/data/raw/sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand/Ghana_103272_S1Hand.tif
S2L1C: D:/flood_research/data/raw/sen1floods11/v1.1/data/flood_events/HandLabeled/S2Hand/Ghana_103272_S2Hand.tif
mask:  D:/flood_research/data/raw/sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand/Ghana_103272_LabelHand.tif
```

Loaded tensor checks:

| Item | Shape | Dtype | Check |
|---|---:|---|---|
| S1GRD | `[2, 512, 512]` | `torch.float32` | finite |
| S2L1C | `[13, 512, 512]` | `torch.float32` | finite |
| mask | `[512, 512]` | `torch.int64` | unique values `[-1, 0, 1]` |

Label value validation: `passed`.

## Model Initialization Smoke Test
Smoke summary:

```text
D:/flood_research/experiments/terramind_baseline/reports/STEP_5B_model_init_summary.json
```

Results:
- Model init attempted: `true`
- Model init OK: `true`
- Task class: `SemanticSegmentationTask`
- Pretrained weights enabled: `false`
- Forward pass attempted: `false`
- Training started: `false`
- Elapsed time: about `185.024 s`

No TerraMind checkpoint download was performed.

## Blockers / Risks
- No hard blocker remains for STEP 5B.
- TerraTorch import and model initialization are slow on CPU-only Windows.
- CUDA is not available in this environment.
- `D:/` exFAT did not support symlink/junction staging; manifest/direct-path mapping is currently the non-destructive solution.
- Full TerraMind training should not be attempted on this CPU-only setup without explicit validation.
- Pretrained tiny/small weight download was intentionally avoided in STEP 5B.
- The exact `TerraMindv1-L + UPerNet` path remains a later controlled adaptation; STEP 5B validates only the public `TerraMind + UNetDecoder` path.

## Generated / Updated Files
- Local report: `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/STEP_5B_terramind_environment_smoke_report.md`
- External report: `D:/flood_research/reports/STEP_5B_terramind_environment_smoke_report.md`
- Install log: `D:/flood_research/experiments/terramind_baseline/logs/STEP_5B_install.log`
- Smoke log: `D:/flood_research/experiments/terramind_baseline/logs/STEP_5B_smoke.log`
- Model init log: `D:/flood_research/experiments/terramind_baseline/logs/STEP_5B_model_init.log`
- Environment summary: `D:/flood_research/experiments/terramind_baseline/reports/STEP_5B_environment_summary.json`
- Dataloader smoke summary: `D:/flood_research/experiments/terramind_baseline/reports/STEP_5B_smoke_summary.json`
- Model init summary: `D:/flood_research/experiments/terramind_baseline/reports/STEP_5B_model_init_summary.json`
- Adapted config: `D:/flood_research/experiments/terramind_baseline/configs/terramind_v1_tiny_sen1floods11_smoke.yaml`
- Training manifest: `D:/flood_research/experiments/terramind_baseline/manifests/terramind_training_manifest.csv`
- Pipeline status: `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/pipeline_status.json`

## Exact Next Steps Requiring Validation
Before STEP 5C, validate whether to:
- download the official tiny TerraMind pretrained weights;
- run a tiny smoke inference on 1-2 samples;
- optionally run a one-batch forward pass;
- optionally run a 2-4 sample tiny overfit test;
- preserve the direct raw path mapping or move experiments to NTFS if symlink-style staging is required.

STOP after STEP 5B. Do not start full training, STEP 5C, UPerNet adaptation, DARN, STURM-Flood, or physics loss until validated.
