# STEP 2 - Sen1Floods11 indexing and audit report

## Summary
- Status: `done`
- Generated at: `2026-06-19T15:44:35`
- Raw data modified: `false`
- STEP 3 started: `false`
- Next step allowed: `false`
- Dataset root: `D:/flood_research/data/raw/sen1floods11`

## Downloaded dataset structure
| Category | Files | Size GB |
|---|---:|---:|
| `WeaklyLabeled` | 17537 | 33.066 |
| `catalog` | 9666 | 0.033 |
| `HandLabeled` | 2230 | 1.753 |
| `perm_water` | 1628 | 1.485 |
| `splits` | 8 | 0.0 |
| `checkpoints` | 4 | 0.527 |
| `metadata` | 1 | 0.0 |

## Inventory totals
- Total files: `31074`
- Total size: `36.864 GB`
- Extensions: `{'.json': 9665, '.zip': 1, '.cp': 4, '.tif': 21394, '': 1, '.geojson': 1, '.csv': 8}`

## Hand-labeled index totals
- Samples indexed: `446`
- Status counts: `{'warning': 69, 'ok': 372, 'error': 5}`
- Cleaning recommendation counts: `{'warning_review': 69, 'keep': 372, 'exclude_candidate': 5}`

## Split totals
- Split files found: `8`
- Train samples: `252`
- Validation samples: `89`
- Test samples: `90`
- Bolivia holdout samples: `15`
- Samples in split files not matched to indexed files: `0`
- Hand-labeled samples not used by train/valid/test splits: `15`
- Hand-labeled samples not used by any flood_handlabeled split: `0`
- Split file names:
  - `D:/flood_research/data/raw/sen1floods11/v1.1/splits/flood_handlabeled/flood_bolivia_data.csv`
  - `D:/flood_research/data/raw/sen1floods11/v1.1/splits/flood_handlabeled/flood_test_data.csv`
  - `D:/flood_research/data/raw/sen1floods11/v1.1/splits/flood_handlabeled/flood_train_data.csv`
  - `D:/flood_research/data/raw/sen1floods11/v1.1/splits/flood_handlabeled/flood_valid_data.csv`
  - `D:/flood_research/data/raw/sen1floods11/v1.1/splits/perm_water/permanent_water_data.csv`
  - `D:/flood_research/data/raw/sen1floods11/v1.1/splits/perm_water/permanent_water_test_data.csv`
  - `D:/flood_research/data/raw/sen1floods11/v1.1/splits/perm_water/permanent_water_train_data.csv`
  - `D:/flood_research/data/raw/sen1floods11/v1.1/splits/perm_water/permanent_water_validation_data.csv`

## Mask label distribution
| Label | Count | Percent | Interpretation |
|---:|---:|---:|---|
| `-1` | 16061210 | 4.579122 | invalid/no-data |
| `0` | 306128814 | 87.278681 | non-water |
| `1` | 28558648 | 8.142197 | water |

## Image/mask consistency findings
- Dimension mismatches S1/mask: `0`
- Dimension mismatches S2/mask: `0`
- CRS mismatches S1/mask: `0`
- CRS mismatches S2/mask: `0`
- Transform/resolution mismatches S1/mask: `0`
- Transform/resolution mismatches S2/mask: `0`

## Anomaly summary
| Anomaly | Samples |
|---|---:|
| `no_water` | 52 |
| `high_invalid_ratio` | 38 |
| `only_invalid_LabelHand` | 5 |

## Cleaning recommendations
- `keep`: no anomaly detected by this audit.
- `warning_review`: sample is readable but should be reviewed before training.
- `exclude_candidate`: sample has missing critical files, label problems, or alignment problems.
- `corrupted_or_unreadable`: at least one file could not be opened by rasterio.
- No cleaning was applied in STEP 2.

## Open questions
- Confirm whether the Bolivia holdout should remain separate from train/validation/test in future experiments.
- Confirm whether `S1OtsuLabelHand` and `JRCWaterHand` should be treated only as QC/reference layers or included in downstream audits.
- Confirm the policy for no-water samples before STEP 3/cleaning decisions.

## Generated files
- Local inventory: `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/sen1floods11_file_inventory.csv`
- Local hand index: `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/sen1floods11_handlabeled_index.csv`
- Local hand audit: `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/sen1floods11_handlabeled_audit.csv`
- Local report: `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/STEP_2_sen1floods11_audit_report.md`
- External inventory: `D:/flood_research/reports/sen1floods11_file_inventory.csv`
- External hand index: `D:/flood_research/reports/sen1floods11_handlabeled_index.csv`
- External hand audit: `D:/flood_research/reports/sen1floods11_handlabeled_audit.csv`
- External report: `D:/flood_research/reports/STEP_2_sen1floods11_audit_report.md`

## Problems detected
- None

## Decision required before STEP 3
- Validate the audit findings and cleaning recommendation categories.
- Do not start statistics/visualizations until this report is reviewed.
