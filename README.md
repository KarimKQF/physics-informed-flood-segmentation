# Physics-Informed Flood Segmentation

This repository contains a research pipeline for flood segmentation on
Sen1Floods11 using TerraMind/TerraTorch baselines, robust evaluation, and future
physics-informed loss extensions.

The codebase is intentionally lightweight: it contains source code, scripts,
configs, tests, manifests, and reproducibility reports. It does not contain raw
imagery, virtual environments, checkpoints, model weights, or large experiment
outputs.

## Project Overview

The project studies flood segmentation with a staged and auditable workflow:

- download and verify Sen1Floods11;
- index and audit the hand-labeled flood segmentation subset;
- generate dataset statistics and visual inspection panels;
- implement segmentation metrics with `ignore_index=-1` support;
- prepare a TerraMind/TerraTorch baseline reproduction path;
- later extend the baseline with physics-informed losses.

The long-term target is to compare strong foundation-model baselines with
physics-aware segmentation methods while keeping evaluation rules explicit and
reproducible.

## Current Status

Completed locally:

- STEP 0R: storage validation
- STEP 1: Sen1Floods11 download
- STEP 2: indexing and audit
- STEP 3: statistics and visualizations
- STEP 4: segmentation metrics implementation
- STEP 5: TerraMind reproduction plan
- STEP 5B: TerraMind/TerraTorch environment setup and smoke tests

The latest pipeline state is tracked in `pipeline_status.json`.

## Dataset

The raw dataset is not included in this repository.

Sen1Floods11 must be downloaded separately from the official bucket:

```text
gs://sen1floods11
```

During development, the validated local dataset path was:

```text
D:/flood_research/data/raw/sen1floods11/
```

The pipeline preserves raw data and applies all filtering through manifests and
reports. The five fully invalid `LabelHand` samples are excluded from supervised
metrics/training candidates:

```text
Ghana_234935
Ghana_26376
Ghana_277
Ghana_5079
Ghana_83483
```

Bolivia is kept as a separate holdout/OOD split.

## Steps Completed

| Step | Output |
|---|---|
| STEP 0R | Environment and storage refresh report |
| STEP 1 | Full Sen1Floods11 download report |
| STEP 2 | File inventory, hand-labeled index, GeoTIFF/mask audit |
| STEP 3 | Statistics summary and visualization report |
| STEP 4 | Binary segmentation metrics and prediction evaluation CLI |
| STEP 5 | TerraMind baseline reproduction plan |
| STEP 5B | TerraMind/TerraTorch venv, configs, manifests, dataloader smoke test |

Key reports live in `reports/`.

## Repository Structure

```text
.
|-- configs/                 # Dataset and experiment configs
|-- manifests/               # Lightweight generated manifests for TerraMind smoke tests
|-- reports/                 # Markdown/CSV/JSON reports and selected figures
|-- scripts/                 # Pipeline, download, audit, statistics, and evaluation scripts
|-- src/
|   |-- metrics/             # STEP 4 segmentation metrics
|   `-- urban_runoff/        # Existing topographic loss and utility code
|-- tests/                   # Unit and smoke tests
|-- README_datasets.md       # Dataset notes
|-- pipeline_status.json     # Latest validated pipeline state
|-- pyproject.toml
|-- requirements.txt
`-- requirements-terramind.txt
```

## Installation

Base development environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

TerraMind/TerraTorch smoke-test environment, used during STEP 5B:

```powershell
python -m venv D:/flood_research/venvs/terramind
D:/flood_research/venvs/terramind/Scripts/Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-terramind.txt
```

## Data Setup

Download Sen1Floods11 outside the repository, for example:

```powershell
gsutil -m rsync -r gs://sen1floods11 D:/flood_research/data/raw/sen1floods11/
```

Then update local config paths as needed in `configs/`.

Do not place raw `.tif` imagery or downloaded data inside this Git repository.

## Metrics

STEP 4 added binary segmentation metrics in `src/metrics/`:

- `compute_confusion_matrix`
- accuracy
- precision
- recall
- F1 score
- IoU background
- IoU water
- mean IoU
- support per class

All metrics support `ignore_index=-1` and are designed not to inflate scores when
a class is absent.

Saved prediction masks can be evaluated with:

```powershell
python scripts/05_evaluate_predictions.py `
  --prediction-dir <prediction_dir> `
  --manifest-csv reports/sen1floods11_handlabeled_index.csv `
  --audit-csv reports/sen1floods11_handlabeled_audit.csv `
  --output-csv reports/example_per_tile_metrics.csv `
  --output-grouped-csv reports/example_grouped_metrics.csv `
  --output-summary-json reports/example_metrics_summary.json `
  --split all
```

## TerraMind Baseline

STEP 5 identified the directly reproducible public baseline as:

```text
TerraMind + UNetDecoder
```

The stronger benchmark reference remains:

```text
TerraMindv1-L + UPerNet head
```

However, the exact UPerNet reproduction requires a controlled config adaptation.
STEP 5B validated the TerraTorch environment, fetched official TerraMind configs,
created lightweight manifests, adapted a tiny smoke config, loaded one sample,
and initialized a tiny TerraMind segmentation task without training.

## What Is Not Included

This repository intentionally excludes:

- raw Sen1Floods11 data;
- processed dataset copies;
- virtual environments;
- model weights and checkpoints;
- `.pt`, `.pth`, `.ckpt`, `.safetensors`, `.onnx`;
- large prediction arrays;
- heavy logs and cache folders;
- credentials, tokens, API keys, and local secrets.

## Next Steps

Recommended next step after cloning on a stronger machine:

1. Configure dataset paths outside the repository.
2. Recreate the TerraMind environment.
3. Run STEP 5C: TerraMind baseline smoke inference / tiny overfit.
4. Only after validation, proceed toward full TerraMind baseline training.
5. Treat TerraMind-L + UPerNet and physics-informed loss extensions as later,
   controlled stages.
