# STEP 6B3 - Sample Copernicus DEM Alignment and QC

Generated: 2026-06-21

## Scope

STEP 6B3 ran a controlled sample alignment of Copernicus DEM GLO-30 to Sen1Floods11 LabelHand grids. This was sample-only validation, not full 441-sample alignment. No model training, physics-loss training, TerraMind training, DARN, or STURM-Flood training was started. Raw Sen1Floods11 data, raw DEM files, and official split files were not modified.

## Inputs

- Sen1Floods11 geospatial inventory: `E:/flood_research/experiments/terramind_baseline/runs/step6b_topographic_alignment_validation/inventory/sen1floods11_geospatial_inventory.csv`
- DEM verification inventory: `E:/flood_research/experiments/terramind_baseline/runs/step6b2b_copernicus_dem_aws_download/inventory/copernicus_dem_verified_inventory.csv`
- DEM source folder: `E:/flood_research/data/raw/dem/copernicus_glo30/`
- Derived output folder: `E:/flood_research/data/derived/sen1floods11_topography/dem_aligned_sample/`
- Run directory: `E:/flood_research/experiments/terramind_baseline/runs/step6b3_sample_dem_alignment_qc/`

## Sample Selection

Deterministic seed: `20260621`

Selected samples:

| Split | Count | Tile IDs |
|---|---:|---|
| train | 3 | `Paraguay_36015`, `USA_770353`, `USA_831672` |
| valid | 2 | `Paraguay_657443`, `Sri-Lanka_85652` |
| test | 2 | `Paraguay_271769`, `Somalia_685158` |
| bolivia | 2 | `Bolivia_195474`, `Bolivia_290290` |

Excluded fully invalid Ghana tile IDs were kept out of the selection policy.

Selection artifacts:

- `E:/flood_research/experiments/terramind_baseline/runs/step6b3_sample_dem_alignment_qc/manifests/step6b3_selected_samples.csv`
- `E:/flood_research/experiments/terramind_baseline/runs/step6b3_sample_dem_alignment_qc/manifests/step6b3_selected_samples.json`

## Alignment Method

For each selected LabelHand raster, the script selected the intersecting Copernicus DEM GLO-30 tile from the verified DEM inventory, reprojected/resampled the DEM to the exact LabelHand CRS, transform, and `512x512` grid, and wrote a derived aligned GeoTIFF plus a compressed `.npz` copy.

Implementation:

- `scripts/physics/step6b3_sample_dem_alignment_qc.py`
- Resampling: bilinear
- Output topography type: `dem_copernicus_glo30`
- Output CRS: LabelHand CRS, `EPSG:4326`
- Output shape: `512x512`

Main outputs:

- Manifest CSV: `E:/flood_research/experiments/terramind_baseline/runs/step6b3_sample_dem_alignment_qc/manifests/topography_sample_manifest.csv`
- Manifest JSON: `E:/flood_research/experiments/terramind_baseline/runs/step6b3_sample_dem_alignment_qc/manifests/topography_sample_manifest.json`
- QC CSV: `E:/flood_research/experiments/terramind_baseline/runs/step6b3_sample_dem_alignment_qc/metrics/sample_alignment_qc.csv`
- QC JSON: `E:/flood_research/experiments/terramind_baseline/runs/step6b3_sample_dem_alignment_qc/metrics/sample_alignment_qc.json`

## QC Metrics

Sample alignment passed.

- Selected samples: 9
- Aligned samples: 9
- Shape matches: 9 / 9
- CRS matches: 9 / 9
- Transform matches: 9 / 9
- Minimum finite ratio: 1.0
- Finite-ratio threshold: > 0.95
- Plausible elevation checks: 9 / 9
- Corrupted rasters: 0

Geospatial validation artifacts:

- Summary: `E:/flood_research/experiments/terramind_baseline/runs/step6b3_sample_dem_alignment_qc/metrics/step6b3_geospatial_validation_summary.json`
- Details: `E:/flood_research/experiments/terramind_baseline/runs/step6b3_sample_dem_alignment_qc/metrics/step6b3_geospatial_validation_details.csv`

## QC Figures

Each panel includes LabelHand, aligned DEM, DEM finite mask, and S1 VV preview.

- `figures/step6b3_qc_train_Paraguay_36015.png`
- `figures/step6b3_qc_train_USA_770353.png`
- `figures/step6b3_qc_train_USA_831672.png`
- `figures/step6b3_qc_valid_Paraguay_657443.png`
- `figures/step6b3_qc_valid_Sri-Lanka_85652.png`
- `figures/step6b3_qc_test_Paraguay_271769.png`
- `figures/step6b3_qc_test_Somalia_685158.png`
- `figures/step6b3_qc_bolivia_Bolivia_195474.png`
- `figures/step6b3_qc_bolivia_Bolivia_290290.png`

Figure directory:

`E:/flood_research/experiments/terramind_baseline/runs/step6b3_sample_dem_alignment_qc/figures/`

## Loss Compatibility Smoke

The real aligned DEM smoke test passed.

- Script: `scripts/physics/step6b3_topography_loss_compatibility_smoke.py`
- Sample used: `Paraguay_36015`
- Logits shape: `[1, 2, 512, 512]`
- Topographic loss: `0.06805865466594696`
- Combined total loss: `0.9050909876823425`
- Combined segmentation loss: `0.9016894102096558`
- Combined topographic loss: `0.06803122907876968`
- Topographic gradient L1: `0.028229335322976112`
- Combined gradient L1: `0.9997469186782837`
- Status: passed

Smoke artifact:

`E:/flood_research/experiments/terramind_baseline/runs/step6b3_sample_dem_alignment_qc/metrics/step6b3_loss_compatibility_smoke_summary.json`

## Limitations

- Copernicus DEM GLO-30 is DSM-like elevation, not HAND and not guaranteed bare-earth DTM.
- Buildings and vegetation may affect local monotonic assumptions.
- DEM resolution is about 30 m, while Sen1Floods11 chips are `512x512` rasters in `EPSG:4326`.
- Bilinear resampling can smooth local relief and introduce interpolation artifacts.
- This sample QC does not prove full-dataset validity.
- Full 441-sample alignment is still required before physics-loss training.

## Result

STEP 6B3 passed as sample-only DEM alignment and QC. Full DEM alignment can proceed after human validation, but `topographic_alignment_validated` remains false until the full Sen1Floods11 topographic alignment is completed and reviewed.

## Next Step

Human validation is required before STEP 6B4: full DEM alignment for all Sen1Floods11 samples.
