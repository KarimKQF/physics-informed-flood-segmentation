# STEP 5 - TerraMind baseline reproduction plan

## Summary
- Status: `done`
- Generated at: `2026-06-19T16:36:22`
- Training started: `false`
- TerraMind inference started: `false`
- Raw data modified: `false`
- Next step allowed: `false`

STEP 5 is a technical reproduction plan only. No package installation, model
download, TerraMind run, DARN work, STURM-Flood work, physics loss work, or
training was started.

## Official resources found
Checked on `2026-06-19`.

| Resource | URL | Notes |
|---|---|---|
| TerraMind official repository | https://github.com/IBM/terramind | Official IBM/ESA TerraMind examples and configs. |
| TerraMind project page | https://ibm.github.io/terramind/ | IBM-ESA overview, model links, paper/code links. |
| TerraMind paper | https://arxiv.org/abs/2504.11171 | Documents TerraMind and PANGAEA benchmark results. |
| TerraTorch active repository | https://github.com/torchgeo/terratorch | Current TerraTorch source; IBM/terratorch links appear to redirect here. |
| TerraTorch paper | https://arxiv.org/abs/2503.20563 | Describes TerraTorch as a PyTorch Lightning fine-tuning toolkit. |
| TerraTorch docs | https://torchgeo.github.io/terratorch/ | Current docs location from repository metadata. |
| TerraMind weights - tiny | https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-tiny | Approx. model repo file size checked: `0.209 GiB`. |
| TerraMind weights - small | https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-small | Approx. model repo file size checked: `0.482 GiB`. |
| TerraMind weights - base | https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base | Approx. model repo file size checked: `1.426 GiB`. |
| TerraMind weights - large | https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-large | Approx. model repo file size checked: `3.539 GiB`; `TerraMind_v1_large.pt` is about `3.53 GiB`. |
| Sen1Floods11 TerraMind notebook | https://github.com/IBM/terramind/blob/main/notebooks/terramind_v1_small_sen1floods11.ipynb | Practical Sen1Floods11 walkthrough; uses TerraTorch and `UNetDecoder`. |
| Sen1Floods11 TerraMind base config | https://github.com/IBM/terramind/blob/main/configs/terramind_v1_base_sen1floods11.yaml | Ready config using `GenericMultiModalDataModule` and `UNetDecoder`. |
| Sen1Floods11 TerraMind tiny config | https://github.com/IBM/terramind/blob/main/configs/terramind_v1_tiny_sen1floods11.yaml | Smaller ready config using `UNetDecoder`. |
| Sen1Floods11 TerraMind TiM config | https://github.com/IBM/terramind/blob/main/configs/terramind_v1_base_tim_lulc_sen1floods11.yaml | Future TiM/LULC variant; not for current baseline implementation. |
| TerraTorch UPerNet decoder source | https://github.com/torchgeo/terratorch/blob/main/terratorch/models/decoders/upernet_decoder.py | `UperNetDecoder` exists in current TerraTorch. |

## Baseline definition clarification

### Target benchmark/reference baseline
`TerraMindv1-L + UPerNet head`

- This remains the desired strongest clean/in-distribution reference baseline for the project.
- It corresponds conceptually to a TerraMind large encoder with a UPerNet-style segmentation head.
- Public TerraMind literature documents PANGAEA-style evaluation using a TerraMind encoder with a trainable UPerNet head.
- To reproduce this exactly, we still need an explicit Sen1Floods11 config that combines:
  - `backbone: terramind_v1_large`
  - large-model index selection, likely `indices: [5, 11, 17, 23]`
  - `decoder: UperNetDecoder`
  - correct UPerNet decoder args for TerraTorch current API
  - identical preprocessing, split policy, loss, freeze/unfreeze policy, and evaluation metrics.

### Most directly reproducible public baseline now
`TerraMind + UNetDecoder`

- This is the ready-to-run public path found in the official TerraMind repository.
- The official Sen1Floods11 configs and notebook use:
  - `GenericMultiModalDataModule`
  - modalities `S2L1C` and `S1GRD`
  - `SemanticSegmentationTask`
  - `EncoderDecoderFactory`
  - `decoder: UNetDecoder`
  - `ignore_index: -1`
- The public config includes comments for large-model indices but still uses `UNetDecoder`.
- Therefore this is the safest first reproduction path after environment setup.

### Future work only
`TerraMind-L + DARN`

- DARN must not be implemented in this run.
- It should remain a future experimental extension after a clean TerraMind baseline is reproduced and evaluated.
- It will require separate architecture, loss, and comparison protocol validation.

## UPerNet availability conclusion
- `UperNetDecoder` is present in the current TerraTorch source and listed in the TerraTorch decoder exports/docs.
- However, no official TerraMind Sen1Floods11 config or notebook was found that uses `TerraMindv1-L + UperNetDecoder` directly.
- The exact UPerNet reproduction is therefore not blocked by total absence of UPerNet, but it requires extra config adaptation and a smoke-test phase before it can be treated as reproduced.
- Recommendation: first validate the official UNetDecoder TerraMind-Sen1Floods11 path, then create a controlled UPerNet config variant.

## Dependency and environment requirements

### Official requirements found
- TerraMind repo setup recommends Python `3.11 or higher`.
- TerraMind repo setup recommends `terratorch>=1.2.5`.
- TerraMind notebook uses `terratorch>=1.2.4`, `gdown`, `tensorboard`, and `setuptools<81`.
- TerraTorch current release checked through GitHub: `v1.2.8`, published `2026-05-29`.
- TerraTorch `pyproject.toml` requires Python `>=3.10` and declares classifiers for Python `3.11`, `3.12`, and `3.13`.
- Major TerraTorch dependencies include `torch>2.0`, `torchvision`, `rioxarray`, `albumentations`, `rasterio`, `torchmetrics`, `geopandas`, `lightning>=2.6.0`, `segmentation-models-pytorch>=0.5.0`, `jsonargparse>=4.40.0`, `torchgeo`, `einops`, `timm>=1.0.15`, `pycocotools`, `huggingface_hub`, `tifffile`, `tqdm`, `tensorboard`, `diffusers`, `scikit-learn`, `scikit-image`, `pyarrow`, `rich`, and `termcolor`.

### Local environment inspected
- Python: `3.12.10`
- Torch: `2.12.0`
- CUDA available through PyTorch: `false`
- CUDA device count: `0`
- Rasterio: `1.5.0`
- NumPy: `2.4.1`
- Installed: `torch`, `rasterio`, `numpy`
- Not installed locally at inspection time: `terratorch`, `torchvision`, `lightning`, `timm`, `huggingface_hub`, `albumentations`, `torchgeo`, `segmentation_models_pytorch`

### Compatibility notes
- Python 3.12 appears supported by current TerraTorch metadata, but the official TerraMind notebook metadata used Python `3.12.8` and the repo setup text also accepts Python 3.11+.
- TerraMind examples may still be easier to reproduce in a clean virtual environment rather than the current project environment because TerraTorch requires many packages and may shift NumPy/Torch dependencies.
- No installation was performed in STEP 5.

### Proposed install commands for later validation only
Do not run until STEP 5B is approved.

```powershell
python -m venv D:/flood_research/venvs/terramind
D:/flood_research/venvs/terramind/Scripts/Activate.ps1
python -m pip install --upgrade pip
python -m pip install "terratorch>=1.2.8" gdown tensorboard "setuptools<81"
```

If CUDA is required, choose the PyTorch build matching the machine/GPU before installing TerraTorch dependencies.

## Disk and checkpoint planning
- External free space inspected: approximately `1811.089 GiB`.
- TerraMind large weights are approximately `3.539 GiB` from the Hugging Face model repo tree.
- Training checkpoints, logs, tensorboard outputs, prediction masks, and copied configs should be budgeted separately.
- Practical planning reserve:
  - tiny/small smoke tests: `5-20 GiB`
  - base experiments: `20-50 GiB`
  - large experiments with checkpoints/logs/predictions: `50-150 GiB`

## Input data mapping plan
Raw dataset root:

```text
D:/flood_research/data/raw/sen1floods11/
```

Current downloaded structure:

```text
D:/flood_research/data/raw/sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand/
D:/flood_research/data/raw/sen1floods11/v1.1/data/flood_events/HandLabeled/S2Hand/
D:/flood_research/data/raw/sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand/
D:/flood_research/data/raw/sen1floods11/v1.1/splits/flood_handlabeled/
```

Official TerraMind example expects a simplified structure:

```text
sen1floods11_v1.1/data/S1GRDHand/
sen1floods11_v1.1/data/S2L1CHand/
sen1floods11_v1.1/data/LabelHand/
sen1floods11_v1.1/splits/flood_train_data.txt
sen1floods11_v1.1/splits/flood_valid_data.txt
sen1floods11_v1.1/splits/flood_test_data.txt
```

Recommended mapping without modifying raw data:
- Build a manifest-based adapter or generated experiment manifest.
- Do not move, rename, or rewrite raw GeoTIFFs.
- Map TerraTorch modality `S1GRD` to local `S1Hand/*_S1Hand.tif`.
- Map TerraTorch modality `S2L1C` to local `S2Hand/*_S2Hand.tif`.
- Map labels to local `LabelHand/*_LabelHand.tif`.
- Use `ignore_index=-1` consistently for labels.
- Replace or mask invalid LabelHand pixels only at load/metric time, never in raw files.
- Use official train/valid/test split memberships from STEP 2 index.
- Keep Bolivia as a separate OOD/holdout split and do not merge into train/valid/test.
- Exclude these fully invalid LabelHand samples from supervised metrics and training manifests:
  - `Ghana_234935`
  - `Ghana_26376`
  - `Ghana_277`
  - `Ghana_5079`
  - `Ghana_83483`
- Keep `warning_review`, `no_water`, and non-fully-invalid high-invalid-ratio samples; ignore `-1` pixels.

## Proposed experiment output structure
Base directory:

```text
D:/flood_research/experiments/terramind_baseline/
```

Recommended subdirectories:

```text
configs/
checkpoints/
predictions/
metrics/
logs/
reports/
manifests/
scratch/
```

Suggested concrete layout:

```text
D:/flood_research/experiments/terramind_baseline/configs/
D:/flood_research/experiments/terramind_baseline/checkpoints/
D:/flood_research/experiments/terramind_baseline/predictions/valid/
D:/flood_research/experiments/terramind_baseline/predictions/test/
D:/flood_research/experiments/terramind_baseline/predictions/bolivia/
D:/flood_research/experiments/terramind_baseline/metrics/
D:/flood_research/experiments/terramind_baseline/logs/
D:/flood_research/experiments/terramind_baseline/reports/
D:/flood_research/experiments/terramind_baseline/manifests/
```

## Reproducibility protocol for future STEP 5B+
Do not run these stages until validated.

1. Environment setup
   - Create an isolated TerraMind virtual environment.
   - Install TerraTorch and dependencies.
   - Verify `terratorch --help`.

2. Data adapter check
   - Create an experiment manifest derived from STEP 2/3 outputs.
   - Confirm S1/S2/LabelHand path mapping for train/valid/test/Bolivia.
   - Confirm the 5 fully invalid LabelHand tiles are excluded from supervised training/metrics only.

3. Dataloader smoke test
   - Load 1-2 batches for train and valid.
   - Check tensor keys, shapes, dtypes, label values, and `ignore_index=-1`.
   - Confirm no raw file is modified.

4. Model initialization smoke test
   - Start with `terramind_v1_tiny` or `terramind_v1_small`.
   - Build the official `UNetDecoder` config first.
   - Verify weights download/cache location and memory usage.

5. Frozen backbone test
   - Run one forward/backward pass on 1-2 samples with frozen backbone.
   - Confirm loss is finite and output shape matches LabelHand masks.

6. Tiny overfit test
   - Use 2-4 samples.
   - Overfit for a very small number of steps.
   - Confirm metrics improve from random baseline.

7. Validation-only/inference sanity check
   - Save prediction masks in a directory compatible with `scripts/05_evaluate_predictions.py`.
   - Confirm metrics CSV/JSON generation.

8. UPerNet config variant
   - Change decoder to `UperNetDecoder`.
   - Confirm required decoder args for TerraTorch current API.
   - Use `LearnedInterpolateToPyramidal` neck for hierarchical features.
   - Validate output shape and memory before training.

9. Full training
   - Only after human validation.
   - Train official UNetDecoder baseline first, then UPerNet variant.

## Evaluation protocol
Predictions must be exported as one mask per tile, where:
- class `0` = background/non-water
- class `1` = water/flood
- predictions can be integer masks, or probability/logit arrays with a documented threshold
- filenames should include the `tile_id`, e.g. `Bolivia_103757.tif` or `Bolivia_103757_pred.tif`

Use:

```powershell
python scripts/05_evaluate_predictions.py `
  --prediction-dir D:/flood_research/experiments/terramind_baseline/predictions/test `
  --manifest-csv reports/sen1floods11_handlabeled_index.csv `
  --audit-csv reports/sen1floods11_handlabeled_audit.csv `
  --output-csv D:/flood_research/experiments/terramind_baseline/metrics/test_per_tile_metrics.csv `
  --output-grouped-csv D:/flood_research/experiments/terramind_baseline/metrics/test_grouped_metrics.csv `
  --output-summary-json D:/flood_research/experiments/terramind_baseline/metrics/test_summary.json `
  --split test
```

Required metrics:
- IoU water
- IoU background
- mIoU
- F1 water
- precision water
- recall water
- accuracy
- global metrics
- grouped metrics by split
- grouped metrics by event/location
- Bolivia reported separately

## Risks and blockers
- Exact `TerraMindv1-L + UPerNet head` is not a ready public Sen1Floods11 config in the checked TerraMind repository.
- Official public Sen1Floods11 notebook/config is currently `TerraMind + UNetDecoder`.
- `UperNetDecoder` exists in TerraTorch, but adapting it to TerraMind-L needs explicit config validation.
- TerraMind-L is larger and may be difficult without a GPU.
- Local PyTorch currently reports no CUDA GPU.
- Python 3.12 appears supported by current metadata, but a clean virtual environment is strongly recommended.
- TerraTorch dependency installation may be heavy and could alter package versions.
- The official TerraMind example expects a simplified dataset folder layout, while our raw download preserves the official bucket layout.
- S1/S2 modality naming differs: local `S1Hand`/`S2Hand` must be mapped to TerraTorch `S1GRD`/`S2L1C`.
- Invalid label handling must remain `ignore_index=-1`.
- Cloud/no-data behavior and S2 optical artifacts need confirmation during dataloader smoke tests.
- No-water samples must be kept to avoid overestimating false-positive control.

## Exact next steps requiring validation
Before STEP 5B, human validation is required for:
- whether to create a separate TerraMind virtual environment under `D:/flood_research/venvs/terramind`;
- whether to install `terratorch>=1.2.8` and its dependencies;
- whether to start with `terramind_v1_tiny`, `terramind_v1_small`, or directly `terramind_v1_base`;
- whether to adapt the dataset via manifest-only loader code or a non-destructive symlink/copy staging area;
- whether the first smoke test should target the official `UNetDecoder` config before UPerNet;
- whether TerraMind-L + UPerNet should be treated as an exact-reproduction goal after the public UNetDecoder baseline passes.

## Generated files
- Local report: `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/STEP_5_terramind_reproduction_plan.md`
- External report: `D:/flood_research/reports/STEP_5_terramind_reproduction_plan.md`
- Pipeline status: `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/pipeline_status.json`
