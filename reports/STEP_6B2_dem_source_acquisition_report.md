# STEP 6B2 - DEM Source Acquisition Report

## Status

Result: BLOCKED

STEP 6B2 computed the required DEM coverage and prepared a safe dry-run plan,
but no DEM source is currently available locally and no credentials/access method
is configured for automatic download.

No model training, physics-loss training, TerraMind training, DARN training,
STURM-Flood training, raw Sen1Floods11 modification, official split-file
modification, or DEM download was started.

## Inputs Read

- `E:/flood_research/experiments/terramind_baseline/runs/step6b_topographic_alignment_validation/inventory/sen1floods11_geospatial_inventory.csv`
- `E:/flood_research/experiments/terramind_baseline/runs/step6b_topographic_alignment_validation/inventory/dem_source_availability.json`
- `E:/flood_research/experiments/terramind_baseline/runs/step6b_topographic_alignment_validation/reports/STEP_6B_topographic_alignment_validation_report.md`
- `E:/flood_research/experiments/terramind_baseline/runs/step6b_topographic_alignment_validation/manifests/topography_manifest_schema.json`

## Required Spatial Coverage

Generated:

`E:/flood_research/experiments/terramind_baseline/runs/step6b2_dem_source_acquisition/inventory/step6b2_required_dem_coverage.json`

Summary:

- Samples: 441
- Event/location names: Bolivia, Ghana, India, Mekong, Nigeria, Pakistan, Paraguay, Somalia, Spain, Sri-Lanka, USA
- CRS: `EPSG:4326`
- Longitude range: `-95.666984498` to `106.107564056`
- Latitude range: `-25.250564658` to `40.290518471`
- Required 1-degree DEM cells: 53

Required-cell manifests:

- `E:/flood_research/experiments/terramind_baseline/runs/step6b2_dem_source_acquisition/manifests/required_dem_cells.csv`
- `E:/flood_research/experiments/terramind_baseline/runs/step6b2_dem_source_acquisition/manifests/required_dem_cells.json`

## Post-6B DEM Rescan

Generated:

`E:/flood_research/experiments/terramind_baseline/runs/step6b2_dem_source_acquisition/inventory/post_6b_dem_rescan.json`

Result:

- Data root scanned: `E:/flood_research/data`
- Files scanned: 79,682
- Raster extensions checked: `.dem`, `.hgt`, `.img`, `.tif`, `.tiff`, `.vrt`
- Keywords checked: Copernicus, GLO-30, SRTM, NASADEM, elevation, DEM, HAND, DTM, DSM
- Candidate DEM rasters found: 0
- DEM source available: False

`HandLabeled`, `LabelHand`, `S1Hand`, and `S2Hand` paths were intentionally
excluded because they are labels/modalities, not HAND topography.

Empty verification inventory written:

- `E:/flood_research/experiments/terramind_baseline/runs/step6b2_dem_source_acquisition/inventory/verified_dem_source_inventory.csv`
- `E:/flood_research/experiments/terramind_baseline/runs/step6b2_dem_source_acquisition/inventory/verified_dem_source_inventory.json`

## Source Selected

Selected planning source:

`copernicus_glo30`

Preferred storage:

`E:/flood_research/data/raw/dem/copernicus_glo30/`

Fallback storage:

`E:/flood_research/data/raw/dem/srtm_1arcsec/`

Source plan:

`reports/STEP_6B2_dem_source_acquisition_plan.md`

## Helper Script

Implemented:

`scripts/physics/step6b2_prepare_dem_source.py`

Copied to run directory:

`E:/flood_research/experiments/terramind_baseline/runs/step6b2_dem_source_acquisition/scripts/step6b2_prepare_dem_source.py`

Capabilities:

- reads `required_dem_cells.csv`;
- supports `--source copernicus_glo30` and `--source srtm_1arcsec`;
- supports `--output-root`;
- defaults to safe dry-run behavior;
- supports `--download` but exits cleanly unless a supported access method is configured;
- avoids assuming credentials;
- writes planned CSV/JSON manifests.

## Dry-Run

Log:

`E:/flood_research/experiments/terramind_baseline/runs/step6b2_dem_source_acquisition/logs/step6b2_dem_source_dry_run.log`

Dry-run summary:

`E:/flood_research/experiments/terramind_baseline/runs/step6b2_dem_source_acquisition/metadata/step6b2_dry_run_summary.json`

Result:

```text
source=copernicus_glo30
source_label=Copernicus DEM GLO-30
required_cells=53
existing_cells=0
missing_cells=53
automatic_download_supported=False
```

Planned download manifests:

- `E:/flood_research/experiments/terramind_baseline/runs/step6b2_dem_source_acquisition/manifests/planned_dem_download_manifest.csv`
- `E:/flood_research/experiments/terramind_baseline/runs/step6b2_dem_source_acquisition/manifests/planned_dem_download_manifest.json`

## Credential/API Status

Current environment:

- `COPERNICUS_DEM_ACCESS_CONFIG`: not configured
- `EARTHDATA_USERNAME`: not configured
- `EARTHDATA_PASSWORD`: not configured

Automatic download was not started. Human confirmation and credentials/access
configuration are required before any download is attempted.

## Can STEP 6B Be Rerun?

Not yet.

STEP 6B can be rerun for sample alignment only after DEM/HAND/elevation files
exist locally and pass source verification.

## Guardrails

- DEM download started: False
- DEM download completed: False
- DEM source verified: False
- Model training started: False
- Physics-loss training started: False
- TerraMind training started: False
- DARN started: False
- STURM-Flood training started: False
- Raw Sen1Floods11 data modified: False
- Official split files altered: False
- Full topographic alignment run: False

## Next Step

Manually provide DEM/HAND source files for the 53 required cells or configure
credentials/access for a small-cell download. After that, verify the files and
continue with STEP 6B3 sample DEM alignment and QC.
