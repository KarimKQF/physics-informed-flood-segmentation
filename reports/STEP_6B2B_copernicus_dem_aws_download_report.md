# STEP 6B2b - Copernicus DEM GLO-30 AWS Download

Generated: 2026-06-21

## Scope

STEP 6B2b downloaded the required Copernicus DEM GLO-30 public AWS tiles for the Sen1Floods11 topographic input pipeline. No model training, physics-loss training, TerraMind training, DARN, or STURM-Flood training was started. Raw Sen1Floods11 data and official split files were not modified.

## Inputs

- Required cells manifest: `E:/flood_research/experiments/terramind_baseline/runs/step6b2_dem_source_acquisition/manifests/required_dem_cells.csv`
- Required 1-degree DEM cells: 53
- Event/location coverage: Bolivia, Ghana, India, Mekong, Nigeria, Pakistan, Paraguay, Somalia, Spain, Sri-Lanka, USA
- Split coverage: bolivia, test, train, valid
- Required coverage bounds: latitude floors -26 to 40, north edge 41; longitude floors -96 to 106, east edge 107

## Tile Resolution

The resolver uses the Copernicus DEM GLO-30 AWS Open Data bucket:

- S3 bucket: `s3://copernicus-dem-30m/`
- Public HTTP endpoint: `https://copernicus-dem-30m.s3.amazonaws.com`
- Access mode: no-sign-request
- Tile list: `https://copernicus-dem-30m.s3.amazonaws.com/tileList.txt`

Tile naming rule:

`Copernicus_DSM_COG_10_<NORTHING>_<EASTING>_DEM/Copernicus_DSM_COG_10_<NORTHING>_<EASTING>_DEM.tif`

Example:

`Copernicus_DSM_COG_10_N40_00_W096_00_DEM/Copernicus_DSM_COG_10_N40_00_W096_00_DEM.tif`

## Dry-Run Result

- Required cells: 53
- Available Copernicus tiles: 53
- Missing Copernicus tiles: 0
- Public tileList folder count: 26,450
- Dry-run log: `E:/flood_research/experiments/terramind_baseline/runs/step6b2b_copernicus_dem_aws_download/logs/step6b2b_copernicus_dem_dry_run.log`

Generated manifests:

- `E:/flood_research/experiments/terramind_baseline/runs/step6b2b_copernicus_dem_aws_download/manifests/copernicus_required_tiles_planned.csv`
- `E:/flood_research/experiments/terramind_baseline/runs/step6b2b_copernicus_dem_aws_download/manifests/copernicus_required_tiles_available.csv`
- `E:/flood_research/experiments/terramind_baseline/runs/step6b2b_copernicus_dem_aws_download/manifests/copernicus_required_tiles_missing.csv`

## Download Status

- Download launched in background via `run_step6b2b_download_copernicus_dem.ps1`
- Parent PID recorded: 24728
- Parent and Python child processes were no longer running at verification time because the download completed quickly
- Download completed: yes
- Downloaded tiles: 53
- Skipped existing tiles: 0
- Failed tiles: 0
- Partial files remaining: 0
- Download log: `E:/flood_research/experiments/terramind_baseline/runs/step6b2b_copernicus_dem_aws_download/logs/step6b2b_copernicus_dem_download.log`
- Target folder: `E:/flood_research/data/raw/dem/copernicus_glo30/`

## Verification

Raster verification was completed with rasterio on all downloaded GeoTIFFs.

- Verified file count: 53 / 53
- Missing file count: 0
- Corrupted file count: 0
- CRS values: `EPSG:4326`
- Dtype values: `float32`
- Resolution: `0.0002777777777777778` degrees x `0.0002777777777777778` degrees
- Raster size: 3600 x 3600 pixels per tile
- Nodata: none reported by the verified tiles
- Finite ratio: 1.0 for all verified tiles
- Global min elevation: -21.44077491760254
- Global max elevation: 7029.01953125

Verification inventory:

- CSV: `E:/flood_research/experiments/terramind_baseline/runs/step6b2b_copernicus_dem_aws_download/inventory/copernicus_dem_verified_inventory.csv`
- JSON: `E:/flood_research/experiments/terramind_baseline/runs/step6b2b_copernicus_dem_aws_download/inventory/copernicus_dem_verified_inventory.json`

## Fallback Requirement

No SRTM fallback is required at this stage. All 53 required Copernicus DEM GLO-30 tiles were present in the public bucket, downloaded, and verified.

## Scripts

- Download script: `scripts/physics/step6b2b_download_copernicus_dem_aws.py`
- Launcher: `run_step6b2b_download_copernicus_dem.ps1`
- Run directory script copies:
  - `E:/flood_research/experiments/terramind_baseline/runs/step6b2b_copernicus_dem_aws_download/scripts/step6b2b_download_copernicus_dem_aws.py`
  - `E:/flood_research/experiments/terramind_baseline/runs/step6b2b_copernicus_dem_aws_download/scripts/run_step6b2b_download_copernicus_dem.ps1`

## Next Step

Human validation is required before STEP 6B3. The next technical step should be sample DEM alignment and QC against Sen1Floods11 geospatial rasters. Do not start physics-loss training until aligned topographic inputs have been validated.
