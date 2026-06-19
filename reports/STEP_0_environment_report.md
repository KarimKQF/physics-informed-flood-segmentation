# STEP 0 - Environment report

## Summary
- Status: `done`
- Generated at: `2026-06-19T10:44:36`
- Dry run: `false`
- Local repo root: `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow`
- Next step allowed: `false`

## Environment
- Python executable: `C:\Users\ELEVES\AppData\Local\Programs\Python\Python312\python.exe`
- Python version: `3.12.10`
- pip: `pip 25.0.1 from C:\Users\ELEVES\AppData\Local\Programs\Python\Python312\Lib\site-packages\pip (python 3.12)`
- conda: `command not found`

## External disk
- Detected label: `ESD-USB`
- Disk path: `D:/`
- Filesystem: `FAT32`
- Project storage root: `D:/flood_research`
- Total space: `34.35 GB`
- Free space: `29.12 GB` (`27.12 GiB`)

## Directories
- `D:/flood_research`
- `D:/flood_research/data/raw/sen1floods11`
- `D:/flood_research/data/processed/sen1floods11`
- `D:/flood_research/reports`
- `D:/flood_research/logs`
- `D:/flood_research/checkpoints`

## Generated files
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/STEP_0_environment_report.md`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/pipeline_status.json`
- `D:/flood_research/reports/STEP_0_environment_report.md`
- `D:/flood_research/pipeline_status.json`
- `D:/flood_research/logs/00_check_environment.log`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/configs/sen1floods11.yaml`

## Problems detected
- conda was not found. This is not blocking because pip is available.
- Detected filesystem is FAT32, which has a 4 GB per-file size limit.
- Free space is below 50 GB; validate capacity before downloading full datasets.

## Decisions to validate
- Confirm that the detected external disk is the intended storage target.
- Confirm that the available free space is sufficient before Step 1.
- Confirm whether FAT32 is acceptable if large files are expected.

## Proposed next step
Step 1 can only start after human validation. It will document and download Sen1Floods11 into the external raw data directory.
