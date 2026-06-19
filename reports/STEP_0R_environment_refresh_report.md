# STEP 0R - Environment refresh report

## Summary
- Status: `done`
- Generated at: `2026-06-19T11:49:31`
- Download launched: `false`
- Next step allowed: `false`
- Local repo root: `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow`

## Environment
- Python: `3.12.10 (C:\Users\ELEVES\AppData\Local\Programs\Python\Python312\python.exe)`
- pip: `pip 25.0.1 from C:\Users\ELEVES\AppData\Local\Programs\Python\Python312\Lib\site-packages\pip (python 3.12)`
- gsutil: `gsutil version: 5.37`

## Required storage rules
- Legacy disk label: `ESD-USB`
- Label requirement: `disabled by explicit human validation`
- Validated external root: `D:/`
- Acceptable filesystems: `exFAT`, `NTFS`
- Minimum expected total capacity: `1500.0 GB`
- Estimated Sen1Floods11 full bucket size: `36.864 GB`
- Required safety reserve: `50.0 GB`

## Detected volumes
| Root | Label | Filesystem | Total GB | Free GB | Drive type |
|---|---|---|---:|---:|---|
| `C:/` | `Windows` | `NTFS` | 127.368 | 6.142 | `fixed` |
| `D:/` | `Nouveau nom` | `exFAT` | 2000.38 | 2000.377 | `fixed` |

## Selected validated external volume
- Disk path: `D:/`
- Current label: `Nouveau nom`
- Filesystem: `exFAT`
- Total space: `2000.38 GB`
- Free space: `2000.377 GB`
- Free space after estimated download: `1963.513 GB`
- Old 33 GB partition limitation cleared: `true`
- Around 2 TB external capacity confirmed: `true`

## Directories
- `D:/flood_research/data/raw/sen1floods11`
- `D:/flood_research/data/processed/sen1floods11`
- `D:/flood_research/reports`
- `D:/flood_research/logs`
- `D:/flood_research/checkpoints`

## Generated files
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/STEP_0R_environment_refresh_report.md`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/pipeline_status.json`
- `D:/flood_research/reports/STEP_0R_environment_refresh_report.md`
- `D:/flood_research/pipeline_status.json`

## Blocking reasons
- None

## Decision
- Storage validation passed.
- STEP 1 full download may proceed automatically in this run.

## Proposed next step
If this step is valid, continue to STEP 1 full download. If blocked, fix the listed storage issue and rerun STEP 0R.
