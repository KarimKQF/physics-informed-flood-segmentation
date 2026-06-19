# STEP 1 - Full Sen1Floods11 download report

## Summary
- Status: `done`
- Generated at: `2026-06-19T11:51:37`
- Full download requested: `true`
- STEP 2 started: `false`
- Next step allowed: `false`

## Source
- Official repository: https://github.com/cloudtostreet/Sen1Floods11
- GCS bucket: `gs://sen1floods11`
- Download command: `C:\Program Files (x86)\Google\Cloud SDK\google-cloud-sdk\bin\gsutil.CMD -m rsync -r gs://sen1floods11 D:\flood_research\data\raw\sen1floods11`
- gsutil executable: `C:\Program Files (x86)\Google\Cloud SDK\google-cloud-sdk\bin\gsutil.CMD`
- gsutil return code: `0`

## Storage
- Local repo root: `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow`
- Config file: `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/configs/sen1floods11.yaml`
- Download target: `D:/flood_research/data/raw/sen1floods11`
- Reports directory: `D:/flood_research/reports`
- Logs directory: `D:/flood_research/logs`
- Download log: `D:/flood_research/logs/01_download_sen1floods11_full.log`
- Disk total space: `2000.38 GB` (`1862.999 GiB`)
- Remaining free space: `1944.705 GB` (`1811.148 GiB`)
- Remaining free space after download: `1944.705 GB`

## Local dataset verification
- Local file count: `31074`
- Expected remote object count from STEP 1A: `31074`
- File count delta: `0`
- Local total size: `36.864 GB` (`34.333 GiB`)
- Expected remote size from STEP 1A: `36.864 GB` (`34.333 GiB`)
- Size delta: `0.0 GB`
- Key prefixes present: `true`

## Key prefixes
| Key | Present | File count | Local path |
|---|---:|---:|---|
| `HandLabeled` | `true` | 2230 | `D:/flood_research/data/raw/sen1floods11/v1.1/data/flood_events/HandLabeled` |
| `WeaklyLabeled` | `true` | 17537 | `D:/flood_research/data/raw/sen1floods11/v1.1/data/flood_events/WeaklyLabeled` |
| `perm_water` | `true` | 1628 | `D:/flood_research/data/raw/sen1floods11/v1.1/data/perm_water` |
| `S1Perm` | `true` | 814 | `D:/flood_research/data/raw/sen1floods11/v1.1/data/perm_water/S1Perm` |
| `JRCPerm` | `true` | 814 | `D:/flood_research/data/raw/sen1floods11/v1.1/data/perm_water/JRCPerm` |
| `splits` | `true` | 8 | `D:/flood_research/data/raw/sen1floods11/v1.1/splits` |
| `flood_handlabeled_splits` | `true` | 4 | `D:/flood_research/data/raw/sen1floods11/v1.1/splits/flood_handlabeled` |
| `perm_water_splits` | `true` | 4 | `D:/flood_research/data/raw/sen1floods11/v1.1/splits/perm_water` |
| `catalog` | `true` | 9665 | `D:/flood_research/data/raw/sen1floods11/v1.1/catalog` |
| `catalog_zip` | `true` | 1 | `D:/flood_research/data/raw/sen1floods11/v1.1/catalog.zip` |
| `metadata` | `true` | 1 | `D:/flood_research/data/raw/sen1floods11/v1.1/Sen1Floods11_Metadata.geojson` |
| `checkpoints` | `true` | 4 | `D:/flood_research/data/raw/sen1floods11/v1.1/checkpoints` |

## Failed or skipped objects
- gsutil return code: `0`
- Log-based hint: `0` failure/error keyword hits, `0` skip keyword hits
- See the full log for object-level transfer details.

## Problems detected
- None

## Generated files
- Local report: `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/STEP_1_full_download_report.md`
- External report: `D:/flood_research/reports/STEP_1_full_download_report.md`
- Pipeline status: `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/pipeline_status.json`

## Decision
- Stop after STEP 1.
- Do not start STEP 2 until human validation.
- Do not index, audit, clean, or train models yet.
