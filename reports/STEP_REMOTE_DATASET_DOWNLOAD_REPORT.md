# STEP REMOTE DATASET DOWNLOAD REPORT

## Machine And Storage

- Machine name: DESKTOP-1OMNEU2
- Selected data root: E:/flood_research/data/raw
- Sen1Floods11 path: E:/flood_research/data/raw/sen1floods11
- STURM-Flood path: E:/flood_research/data/raw/sturm_flood

Drive summary after dataset downloads:

| Drive | Filesystem | Total GB | Free GB | Volume |
| --- | --- | ---: | ---: | --- |
| C:/ | NTFS | 445.08 | 55.49 |  |
| E:/ | NTFS | 919.53 | 227.58 | Windows |
| F:/ | NTFS | 10.75 | 1.12 | RECOVERY |
| G:/ | NTFS | 838.35 | 234.40 | .SSD SAMSUNG 2 To |

Initial pre-download free-space check selected E:/ because it exceeded the 100 GB requirement and had the most free space among suitable drives.

## Sen1Floods11

- Official source: gs://sen1floods11
- Official repository: https://github.com/cloudtostreet/Sen1Floods11
- Download command used: gsutil rsync from the official GCS bucket
- Local tool note: Google Cloud SDK 573.0.0 / gsutil 5.37 was staged in C:/tmp. A temporary embedded Python runtime in C:/tmp was used to run gsutil without installing a system Python.
- Certificate note: Windows certificate revocation checking failed for storage.googleapis.com. The gsutil run used `-o Boto:https_validate_certificates=False` after a public HTTPS probe confirmed the official object was reachable.
- Download status: complete
- Sync exit code: 0
- File count: 31,074
- Size: 34.333 GiB / 36.864 GB

Required path verification:

| Required path | Exists |
| --- | --- |
| v1.1/data/flood_events/HandLabeled/S1Hand | yes |
| v1.1/data/flood_events/HandLabeled/S2Hand | yes |
| v1.1/data/flood_events/HandLabeled/LabelHand | yes |
| v1.1/data/flood_events/WeaklyLabeled | yes |
| v1.1/data/perm_water | yes |
| v1.1/splits/flood_handlabeled | yes |
| v1.1/Sen1Floods11_Metadata.geojson | yes |
| v1.1/catalog | yes |
| v1.1/checkpoints | yes |

Sen1Floods11 verification status: ready.

## STURM-Flood

- Official dataset source used: https://zenodo.org/records/12748983
- Dataset DOI: https://doi.org/10.5281/zenodo.12748983
- Official code repository: https://github.com/STURM-WEO/STURM-Flood
- Related paper DOI: https://doi.org/10.1080/20964471.2025.2458714
- Downloaded file: E:/flood_research/data/raw/sturm_flood/Dataset.zip
- Download status: complete
- Download exit code: 0
- Zenodo MD5: 0e4172e74a4cf2e4c608c14d1588d1d9
- Local MD5: 0e4172e74a4cf2e4c608c14d1588d1d9
- Extracted dataset path: E:/flood_research/data/raw/sturm_flood/Dataset
- Official code path: E:/flood_research/data/raw/sturm_flood/official_code/STURM-Flood

STURM-Flood size and count:

| Item | File count | Size GiB | Size GB |
| --- | ---: | ---: | ---: |
| STURM-Flood root, including ZIP and official code | 48,608 | 9.811 | 10.534 |
| Extracted Dataset/ only | 48,556 | 6.093 | 6.543 |
| Dataset.zip | 1 | 3.712 | 3.985 |
| Official code clone | 51 | 0.006 | 0.006 |

Component verification:

| Component | Path | File count | Status |
| --- | --- | ---: | --- |
| Sentinel-1 tiles | Dataset/Sentinel1/S1 | 21,602 | present |
| Sentinel-1 water/flood masks | Dataset/Sentinel1/Floodmaps | 21,602 | present |
| Sentinel-2 tiles | Dataset/Sentinel2/S2 | 2,675 | present |
| Sentinel-2 water/flood masks | Dataset/Sentinel2/Floodmaps | 2,675 | present |
| Sentinel-1 metadata | Dataset/Sentinel1_metadata.csv | 1 | present |
| Sentinel-2 metadata | Dataset/Sentinel2_metadata.csv | 1 | present |
| Official inference notebook | official_code/STURM-Flood/inference.ipynb | 1 | present |
| Official architecture files | official_code/STURM-Flood/arch/ | present | present |
| Official utility files | official_code/STURM-Flood/utils/ | present | present |
| Official requirements file | official_code/STURM-Flood/requirements.txt | 1 | present |

STURM-Flood metadata CSV fields:

- Sentinel-1: `ems_code`, `aoi_code`, `floodmap_id`, `event_type`, `country`, `tile_id`, `epsg_code`, `floodmap_date`, `sentinel_date`
- Sentinel-2: `ems_code`, `aoi_code`, `floodmap_id`, `event_type`, `country`, `tile_id`, `epsg_code`, `floodmap_date`, `sentinel_timestamp`

Missing or not provided in the downloaded dataset archive:

- Separate train/validation/test split files were not found.
- No notebooks or Python scripts are included inside Dataset.zip itself.
- Pretrained U-Net model archives exist as separate Zenodo model records, but they were not downloaded because this step was limited to dataset download/verification and the project instructions prohibit creating model checkpoints.

STURM-Flood verification status: ready as a raw verified dataset; split policy still needs human validation before training use.

## Local Repository Files

- Created/updated configs/local_paths.yaml with selected local paths.
- Updated .gitignore to include E:/flood_research/data/ and missing raw raster/archive patterns.
- Raw data is stored outside the Git-tracked repository.
- No raw dataset files were committed or pushed.

## Training Readiness

- Sen1Floods11 is downloaded and verified.
- STURM-Flood is downloaded and verified from the official Zenodo record.
- The raw datasets are ready for human validation and subsequent preprocessing/configuration.
- Direct training is not started and should still wait for human validation, especially because STURM-Flood does not provide separate train/validation/test split files in Dataset.zip.
- No training, TerraMind run, DARN run, physics loss implementation, model execution, dependency installation, or checkpoint creation was performed.

## Next Required Human Validation

Human validation is required before proceeding:

- Confirm whether to derive STURM-Flood train/validation/test splits from metadata or follow a paper-specific split protocol.
- Confirm whether separate official U-Net model Zenodo records should be downloaded later.
- Confirm before setting up any GPU environment or running training.
