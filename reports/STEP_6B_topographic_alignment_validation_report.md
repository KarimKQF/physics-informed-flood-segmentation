# STEP 6B - Topographic Alignment Validation

## Status

Result: BLOCKED

STEP 6B validated the available Sen1Floods11 geospatial metadata and prepared a
safe alignment workflow, but full topographic alignment is blocked because no
local DEM/SRTM/HAND/elevation raster source exists under `E:/flood_research/data`.

No model training, physics-loss training, TerraMind training, DARN training,
STURM-Flood training, raw Sen1Floods11 modification, raw-data overwrite, or
official split-file modification was started.

## Run Directory

`E:/flood_research/experiments/terramind_baseline/runs/step6b_topographic_alignment_validation`

Created subfolders:

- `reports/`
- `logs/`
- `metrics/`
- `figures/`
- `inventory/`
- `scripts/`
- `sample_outputs/`
- `manifests/`

Derived topography location prepared without raw-data modification:

`E:/flood_research/data/derived/sen1floods11_topography/`

Prepared derived subfolders:

- `dem_aligned/`
- `hand_aligned/`
- `qc_figures/`
- `manifests/`
- `metadata/`

## Existing Topographic Inventory Review

Source reviewed:

`E:/flood_research/experiments/terramind_baseline/runs/step5n_baseline_freeze_and_physics_loss_prep/data_availability/topographic_inputs_inventory.json`

Additional STEP 6A context reviewed:

- `reports/STEP_6A_physics_topographic_loss_implementation_report.md`
- `configs/physics_loss/terramind_l_upernet_topographic_loss_stub.yaml`
- `configs/physics_loss/terramind_base_unetdecoder_topographic_loss_control_stub.yaml`

Review output:

`E:/flood_research/experiments/terramind_baseline/runs/step6b_topographic_alignment_validation/inventory/existing_topographic_inventory_review.json`

Conclusion:

- DEM available: False
- SRTM available: False
- HAND / Height Above Nearest Drainage available: False
- Slope available: False
- Flow direction available: False
- Elevation raster available: False
- Permanent-water masks available: True, but they are not topography and cannot drive `TopographicInconsistencyLoss`
- Existing aligned topography for Sen1Floods11: False

Primary blocker:

No local DEM/SRTM/HAND/elevation source is available. Sen1Floods11 GeoTIFF
metadata can support alignment once a DEM/HAND source is provided.

## Sen1Floods11 Geospatial Inventory

Generated:

- `inventory/sen1floods11_geospatial_inventory.csv`
- `inventory/sen1floods11_geospatial_inventory.json`

Inventory basis:

- STEP 5E filtered split manifests
- train/valid/test official split policy
- Bolivia kept separate
- excluded fully invalid samples:
  `Ghana_234935`, `Ghana_26376`, `Ghana_277`, `Ghana_5079`, `Ghana_83483`

Inventory results:

| Split | Samples |
| --- | ---: |
| train | 251 |
| valid | 86 |
| test | 89 |
| bolivia | 15 |
| total | 441 |

Geospatial metadata result:

- Rows inspected: 441
- Rows OK: 441
- Problem rows: 0
- S1 matches LabelHand grid: True for all rows
- S2 matches LabelHand grid: True for all rows
- Label CRS: `EPSG:4326`
- Label shape: `512x512`

This validates that Sen1Floods11 hand-labeled S1/S2/LabelHand rasters provide a
consistent reference grid for future topographic alignment.

## DEM Source Availability

Generated:

`E:/flood_research/experiments/terramind_baseline/runs/step6b_topographic_alignment_validation/inventory/dem_source_availability.json`

Search result:

- Data root scanned: `E:/flood_research/data`
- Files scanned: 79,682
- DEM/SRTM/HAND/elevation candidate rasters found: 0
- DEM source available: False

Permanent-water and `HandLabeled` files were not counted as HAND topography:
`HandLabeled`, `LabelHand`, `S1Hand`, and `S2Hand` are Sen1Floods11 labels or
modalities, not Height Above Nearest Drainage.

## Alignment Utility

Implemented:

`scripts/physics/step6b_align_topography_to_sen1floods11.py`

Copied to run scripts:

`E:/flood_research/experiments/terramind_baseline/runs/step6b_topographic_alignment_validation/scripts/step6b_align_topography_to_sen1floods11.py`

Capabilities:

- accepts a source DEM file or folder;
- reads the STEP 6B geospatial inventory;
- selects matching tiles by split or tile id;
- reprojects/resamples DEM data onto each `LabelHand` grid;
- writes aligned GeoTIFF or compressed NPZ outputs;
- preserves `tile_id` in filenames;
- writes CSV/JSON topography manifests;
- handles nodata/non-finite values;
- computes finite ratio, nodata ratio, min, max, mean, and std;
- validates output shape, CRS, and transform for GeoTIFF outputs.

The utility was implemented and syntax-checked, but no real alignment was run
because no DEM/HAND source is available.

## Sample Alignment

Requested sample plan:

- 3 train samples
- 2 valid samples
- 2 test samples
- 2 Bolivia samples

Result:

Sample alignment was skipped.

Reason:

`No local DEM/SRTM/HAND/elevation raster source was found, so sample alignment was not run.`

QC file:

`E:/flood_research/experiments/terramind_baseline/runs/step6b_topographic_alignment_validation/metrics/sample_alignment_qc.json`

No aligned sample outputs or QC figures were created.

## Topography Manifest Schema

Created:

`E:/flood_research/experiments/terramind_baseline/runs/step6b_topographic_alignment_validation/manifests/topography_manifest_schema.json`

Expected future fields:

- `tile_id`
- `split`
- `event_location`
- `s1_path`
- `s2_path`
- `label_path`
- `topography_path`
- `topography_type`
- `crs`
- `height`
- `width`
- `finite_ratio`
- `nodata_ratio`
- `status`
- `notes`

No `topography_sample_manifest.csv` or `.json` was created because no sample
alignment could be run.

## Loss Compatibility Smoke

Implemented:

`scripts/physics/step6b_topography_loss_compatibility_smoke.py`

Copied to run scripts:

`E:/flood_research/experiments/terramind_baseline/runs/step6b_topographic_alignment_validation/scripts/step6b_topography_loss_compatibility_smoke.py`

Executed result:

```text
status=blocked_no_aligned_topography
```

Summary:

`E:/flood_research/experiments/terramind_baseline/runs/step6b_topographic_alignment_validation/metrics/step6b_loss_compatibility_smoke_summary.json`

The smoke exits cleanly and records the blocker because no aligned topography
sample manifest exists yet.

## Blockers

STEP 6B is blocked by missing topographic source data:

- no DEM source;
- no SRTM source;
- no HAND product;
- no elevation raster;
- no aligned derived topography;
- no topography sample manifest for STEP 6A loss compatibility.

## Guardrails

- Model training started: False
- Physics-loss training started: False
- TerraMind training started: False
- DARN started: False
- STURM-Flood training started: False
- Raw Sen1Floods11 data modified: False
- Official split files altered: False
- Large DEM/HAND download started: False

## Decision

STEP 6B is `blocked`, not `done`, because topographic alignment cannot be
validated without a DEM/HAND/elevation source. The Sen1Floods11 internal raster
grids are validated and the alignment workflow is ready for a future source.

## Exact Next Step

STEP 6B2 should obtain or generate a DEM/HAND topographic input source for
Sen1Floods11, then rerun a small sample alignment before any physics-informed
training is considered.
