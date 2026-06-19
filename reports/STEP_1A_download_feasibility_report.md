# STEP 1A - Sen1Floods11 download feasibility report

## Summary
- Status: `blocked`
- Generated at: `2026-06-19T10:56:53`
- Download performed: `false`
- Next step allowed: `false`

## Official source checked
- Official repository: https://github.com/cloudtostreet/Sen1Floods11
- Paper page: https://openaccess.thecvf.com/content_CVPRW_2020/html/w11/Bonafilia_Sen1Floods11_A_Georeferenced_Dataset_to_Train_and_Test_Deep_Learning_CVPRW_2020_paper.html
- Public Google Cloud Storage bucket: `gs://sen1floods11`
- HTTPS bucket endpoint: https://storage.googleapis.com/sen1floods11/
- Metadata inspection method: `gsutil metadata listing via C:\Program Files (x86)\Google\Cloud SDK\google-cloud-sdk\bin\gsutil.CMD`
- Official recommended full-download command: `gsutil -m rsync -r gs://sen1floods11 <local_directory>`

## Storage target
- Local repo root: `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow`
- Config file: `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/configs/sen1floods11.yaml`
- External project root: `D:/flood_research`
- Raw target directory: `D:/flood_research/data/raw/sen1floods11`
- Reports directory: `D:/flood_research/reports`
- Logs directory: `D:/flood_research/logs`
- Filesystem label: `ESD-USB`
- Filesystem type: `FAT32`
- Current free space: `29.122 GB` (`27.122 GiB`)

## Remote size analysis
- Remote object count: `31074`
- Estimated total size: `36.864 GB` (`34.333 GiB`)
- Total size exceeds 25 GB threshold: `true`
- Largest individual object: `0.132 GB` (`0.123 GiB`)
- Largest individual object URI: `gs://sen1floods11/v1.1/checkpoints/5e4_flood_0_89_0.5418742895126343.cp`
- Largest GeoTIFF object: `0.007 GB` (`0.006 GiB`)
- Largest GeoTIFF URI: `gs://sen1floods11/v1.1/data/flood_events/WeaklyLabeled/S2Weak/Bolivia_7437667_S2Weak.tif`
- Objects larger than FAT32 4 GiB limit: `0`
- FAT32 compatibility by individual file size: `true`
- Estimated free space after complete download: `-7.742 GB` (`-7.21 GiB`)
- Disk space sufficient for full download and 5 GB reserve: `false`

## Size by main source category
| Category | Objects | Size GB |
|---|---:|---:|
| `v1.1/data/flood_events/WeaklyLabeled` | 17537 | 33.066 |
| `v1.1/data/flood_events/HandLabeled` | 2230 | 1.753 |
| `v1.1/data/perm_water/S1Perm` | 814 | 1.479 |
| `v1.1/checkpoints` | 4 | 0.527 |
| `v1.1/catalog` | 9665 | 0.023 |
| `v1.1/catalog.zip` | 1 | 0.01 |
| `v1.1/data/perm_water/JRCPerm` | 814 | 0.006 |
| `v1.1/splits/perm_water` | 4 | 0.0 |
| `v1.1/splits/flood_handlabeled` | 4 | 0.0 |
| `v1.1/Sen1Floods11_Metadata.geojson` | 1 | 0.0 |

## Size by extension
| Extension | Objects | Size GB |
|---|---:|---:|
| `.tif` | 21392 | 36.302 |
| `.cp` | 4 | 0.527 |
| `.json` | 9665 | 0.023 |
| `.zip` | 1 | 0.01 |
| `.473803173211687` | 1 | 0.002 |
| `.csv` | 8 | 0.0 |
| `.geojson` | 1 | 0.0 |
| `.ds_store` | 1 | 0.0 |
| `.107615658056197` | 1 | 0.0 |

## Format and download options
- The bucket exposes many individual files, mainly Cloud-Optimized GeoTIFF-like `.tif` objects, plus STAC catalog JSON files, split CSV files, and small checkpoints.
- No single full-dataset archive was identified. `catalog.zip` is a small metadata archive, not the full imagery dataset.
- A file-by-file or collection-by-collection download is preferable for FAT32 because the largest object is far below 4 GiB.
- For a constrained disk, the most realistic first option is to download metadata, splits, and the hand-labeled subset before considering the weakly labeled subset.

## Problems detected
- Estimated total dataset size exceeds the 25 GB validation threshold.
- Free space after a complete download would be below 5 GB.
- External disk is FAT32; individual files are compatible, but exFAT/NTFS is safer.

## Recommendation
- Do not launch the full Sen1Floods11 download now.
- The full bucket is larger than the configured validation threshold and larger than the currently available free space with a 5 GB reserve.
- Recommended options: use a larger disk formatted as exFAT or NTFS, free substantial space, or validate a smaller scoped download first.
- FAT32 is not blocked by individual file size for this source, but exFAT/NTFS is preferable for reliability and future large artifacts.

## Generated files
- Local report: `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/STEP_1A_download_feasibility_report.md`
- External report: `D:/flood_research/reports/STEP_1A_download_feasibility_report.md`
- Pipeline status: `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/pipeline_status.json`

## Proposed next step
Human validation is required before STEP 1B. STEP 1B should either prepare a smaller scoped download or move to a larger exFAT/NTFS storage target.
