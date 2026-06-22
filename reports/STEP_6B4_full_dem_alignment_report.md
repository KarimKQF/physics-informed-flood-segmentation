# STEP 6B4 - Full Copernicus DEM Alignment

Generated: 2026-06-21

## Scope

STEP 6B4 aligned Copernicus DEM GLO-30 to all valid Sen1Floods11 hand-labeled `LabelHand` grids. No model training, physics-loss training, TerraMind training, DARN, or STURM-Flood training was started. Raw Sen1Floods11 data, raw DEM files, and official split files were not modified.

## DEM Source

- Source: Copernicus DEM GLO-30
- Local source folder: `E:/flood_research/data/raw/dem/copernicus_glo30/`
- Verified DEM tiles from STEP 6B2b: 53 / 53
- Output folder: `E:/flood_research/data/derived/sen1floods11_topography/dem_aligned/`

## Alignment Method

Each verified Sen1Floods11 sample was matched to an intersecting Copernicus DEM tile. The DEM was bilinearly reprojected/resampled to the exact `LabelHand` CRS, transform, and `512x512` grid. Existing valid outputs are reused unless `--overwrite` is requested.

## Full QC Summary

- Expected samples: 441
- Input selected samples: 441
- Aligned samples: 441
- Valid outputs: 441
- Missing outputs: 0
- Failed outputs: 0
- Shape pass count: 441
- CRS pass count: 441
- Transform pass count: 441
- Finite ratio min: 1.0
- Finite ratio mean: 1.0
- Corrupted count: 0

## Split Counts

- train: selected 251, passed 251, expected 251
- valid: selected 86, passed 86, expected 86
- test: selected 89, passed 89, expected 89
- bolivia: selected 15, passed 15, expected 15

## Event Counts

- Bolivia: selected 15, passed 15
- Ghana: selected 48, passed 48
- India: selected 68, passed 68
- Mekong: selected 30, passed 30
- Nigeria: selected 18, passed 18
- Pakistan: selected 28, passed 28
- Paraguay: selected 67, passed 67
- Somalia: selected 26, passed 26
- Spain: selected 30, passed 30
- Sri-Lanka: selected 42, passed 42
- USA: selected 69, passed 69

## Manifests And Metrics

- Full manifest CSV: `E:/flood_research/experiments/terramind_baseline/runs/step6b4_full_dem_alignment/manifests/topography_full_manifest.csv`
- Full manifest JSON: `E:/flood_research/experiments/terramind_baseline/runs/step6b4_full_dem_alignment/manifests/topography_full_manifest.json`
- Full QC CSV: `E:/flood_research/experiments/terramind_baseline/runs/step6b4_full_dem_alignment/metrics/step6b4_full_alignment_qc.csv`
- Full QC JSON: `E:/flood_research/experiments/terramind_baseline/runs/step6b4_full_dem_alignment/metrics/step6b4_full_alignment_qc.json`
- Full summary JSON: `E:/flood_research/experiments/terramind_baseline/runs/step6b4_full_dem_alignment/metrics/step6b4_full_alignment_summary.json`

## QC Figures

- `figures/step6b4_qc_bolivia_Bolivia_103757.png`
- `figures/step6b4_qc_bolivia_Bolivia_129334.png`
- `figures/step6b4_qc_bolivia_Bolivia_195474.png`
- `figures/step6b4_qc_bolivia_Bolivia_290290.png`
- `figures/step6b4_qc_test_India_772630.png`
- `figures/step6b4_qc_test_India_80221.png`
- `figures/step6b4_qc_test_Pakistan_849790.png`
- `figures/step6b4_qc_test_Paraguay_232281.png`
- `figures/step6b4_qc_test_Paraguay_271769.png`
- `figures/step6b4_qc_test_Paraguay_80102.png`
- `figures/step6b4_qc_test_Somalia_685158.png`
- `figures/step6b4_qc_test_Somalia_699062.png`
- `figures/step6b4_qc_test_Sri-Lanka_117737.png`
- `figures/step6b4_qc_test_Sri-Lanka_534068.png`
- `figures/step6b4_qc_train_Ghana_8090.png`
- `figures/step6b4_qc_train_India_136196.png`
- `figures/step6b4_qc_train_India_25540.png`
- `figures/step6b4_qc_train_India_265762.png`
- `figures/step6b4_qc_train_India_373039.png`
- `figures/step6b4_qc_train_Mekong_1282475.png`
- `figures/step6b4_qc_train_Nigeria_78061.png`
- `figures/step6b4_qc_train_Nigeria_952958.png`
- `figures/step6b4_qc_train_Pakistan_474121.png`
- `figures/step6b4_qc_train_Pakistan_909806.png`
- `figures/step6b4_qc_train_Paraguay_36015.png`
- `figures/step6b4_qc_train_Paraguay_605682.png`
- `figures/step6b4_qc_train_Paraguay_791364.png`
- `figures/step6b4_qc_train_Somalia_195014.png`
- `figures/step6b4_qc_train_Somalia_32375.png`
- `figures/step6b4_qc_train_Somalia_989553.png`
- `figures/step6b4_qc_train_Spain_7856615.png`
- `figures/step6b4_qc_train_Sri-Lanka_120804.png`
- `figures/step6b4_qc_train_USA_354981.png`
- `figures/step6b4_qc_train_USA_831672.png`
- `figures/step6b4_qc_valid_Ghana_124834.png`
- `figures/step6b4_qc_valid_Paraguay_305760.png`
- `figures/step6b4_qc_valid_Paraguay_648632.png`
- `figures/step6b4_qc_valid_Paraguay_657443.png`
- `figures/step6b4_qc_valid_Paraguay_934240.png`
- `figures/step6b4_qc_valid_Spain_8372658.png`
- `figures/step6b4_qc_valid_Sri-Lanka_85652.png`
- `figures/step6b4_qc_valid_USA_761032.png`

## Loss Compatibility Smoke

- Status: passed
- Tested samples: 12
- Passed samples: 12
- Failed samples: 0
- Uses real aligned topography: true
- Uses synthetic logits: true
- Summary: `E:/flood_research/experiments/terramind_baseline/runs/step6b4_full_dem_alignment/metrics/step6b4_loss_compatibility_smoke_summary.json`

## Training-Ready Manifest Stubs

- Main target config stub: `C:/Users/Karim/Desktop/flood-segmentation-training/physics-informed-flood-segmentation/configs/physics_loss/terramind_l_upernet_topographic_loss_ready_manifest_stub.yaml`
- Control config stub: `C:/Users/Karim/Desktop/flood-segmentation-training/physics-informed-flood-segmentation/configs/physics_loss/terramind_base_unetdecoder_topographic_loss_control_ready_manifest_stub.yaml`

## Limitations

- Copernicus DEM GLO-30 is DSM-like elevation, not HAND and not guaranteed bare-earth DTM.
- Buildings and vegetation may affect local monotonic assumptions.
- DEM is about 30 m resolution while Sen1Floods11 chips are `512x512` in `EPSG:4326`.
- Bilinear resampling can smooth local relief.
- This validates topographic raster alignment, not hydraulic correctness.

## Result

Full topographic alignment passed: True

Physics-informed training can be considered next only after human validation of this report, the QC figures, and the manifest stubs.

## Next Step

Human validation is required before STEP 6C: first physics-informed training on TerraMind-L + UPerNet.
