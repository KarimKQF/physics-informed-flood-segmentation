# Smoke Tests

## Baseline and topographic smoke tests

These scripts validate the technical training pipeline before integrating real
datasets. They use small synthetic GeoTIFF rasters with matching image, mask and
DEM shapes. The baseline and topographic scripts write to separate synthetic
data folders so they can be rerun independently during development.

Run the binary baseline smoke test:

```bash
python experiments/smoke_tests/smoke_test_baseline_training.py
```

Run the DEM-only topographic smoke test:

```bash
python experiments/smoke_tests/smoke_test_topographic_loss.py
```

Run a GeoTIFF manifest smoke test after creating a real or subset manifest:

```powershell
python experiments\smoke_tests\smoke_test_geotiff_training.py `
  --manifest D:\urban_runoff_data\manifests\sen1floods11_subset_manifest.csv `
  --max-samples 8
```

## DEM alignment for Sen1Floods11

The raw DEM is not consumed directly by the smoke test. It must first be aligned
to each Sen1Floods11 reference image and written into a separate manifest with a
filled `dem_path` column.

The current 30-sample subset is geographically dispersed, so a single global DEM
would require too many SRTM tiles. Prefer a compact subset and per-sample DEM
alignment for smoke testing.

Inspect subset bounds:

```powershell
python scripts\inspect_sen1floods11_bounds.py `
  --manifest D:\urban_runoff_data\manifests\sen1floods11_subset_manifest.csv `
  --output D:\urban_runoff_data\logs\sen1floods11_subset_bounds.json
```

Download a small raw DEM covering the subset:

```powershell
python scripts\download_dem_for_sen1floods11_subset.py `
  --manifest D:\urban_runoff_data\manifests\sen1floods11_subset_manifest.csv `
  --config configs\local_paths.yaml `
  --source srtm `
  --overwrite
```

Align the raw DEM to each sample:

```powershell
python scripts\align_dem_to_sen1floods11_manifest.py `
  --manifest D:\urban_runoff_data\manifests\sen1floods11_subset_manifest.csv `
  --dem D:\urban_runoff_data\raw\DEM\sen1floods11_subset_dem_raw.tif `
  --output-dir D:\urban_runoff_data\processed\aligned_dem\Sen1Floods11 `
  --output-manifest D:\urban_runoff_data\manifests\sen1floods11_subset_with_dem_manifest.csv `
  --overwrite
```

Validate DEM alignment:

```powershell
python scripts\validate_dem_alignment_manifest.py `
  --manifest D:\urban_runoff_data\manifests\sen1floods11_subset_with_dem_manifest.csv `
  --max-samples 30
```

Run the real GeoTIFF smoke test with DEM:

```powershell
python experiments\smoke_tests\smoke_test_geotiff_training.py `
  --manifest D:\urban_runoff_data\manifests\sen1floods11_subset_with_dem_manifest.csv `
  --use-dem `
  --max-samples 8
```

This validates only technical execution: real GeoTIFF loading, `valid_mask`,
DEM loading, Masked BCE, Topographic Loss, backward pass and optimizer step. It
is not a model performance evaluation.

## Per-sample DEM workflow

SRTM tiles are cached outside the repository:

```text
D:/urban_runoff_data/raw/DEM/srtm_tiles
```

Aligned DEM files are written outside the repository:

```text
D:/urban_runoff_data/processed/aligned_dem/Sen1Floods11
```

Create a compact subset manifest:

```powershell
python scripts\create_compact_sen1floods11_subset_manifest.py `
  --manifest D:\urban_runoff_data\manifests\sen1floods11_subset_manifest.csv `
  --bounds-json D:\urban_runoff_data\logs\sen1floods11_subset_bounds.json `
  --output-manifest D:\urban_runoff_data\manifests\sen1floods11_compact_dem_subset_manifest.csv `
  --max-samples 8 `
  --max-unique-srtm-tiles 16
```

Download and align DEMs per sample:

```powershell
python scripts\download_and_align_srtm_per_sample.py `
  --manifest D:\urban_runoff_data\manifests\sen1floods11_compact_dem_subset_manifest.csv `
  --output-manifest D:\urban_runoff_data\manifests\sen1floods11_compact_with_dem_manifest.csv `
  --tile-cache-dir D:\urban_runoff_data\raw\DEM\srtm_tiles `
  --aligned-dem-dir D:\urban_runoff_data\processed\aligned_dem\Sen1Floods11 `
  --max-samples 8 `
  --max-tiles-total 16 `
  --overwrite
```

Validate and smoke test:

```powershell
python scripts\validate_dem_alignment_manifest.py `
  --manifest D:\urban_runoff_data\manifests\sen1floods11_compact_with_dem_manifest.csv `
  --max-samples 8

python experiments\smoke_tests\smoke_test_geotiff_training.py `
  --manifest D:\urban_runoff_data\manifests\sen1floods11_compact_with_dem_manifest.csv `
  --use-dem `
  --max-samples 8
```

These smoke tests are not meant to evaluate model performance. They only
validate that the training pipeline, classical loss, DEM loading and
Topographic Loss are technically executable.

The synthetic rasters are not Sentinel, DEM products, STURM-Flood, or any other
real dataset. They are development-only fixtures for checking that code runs.
