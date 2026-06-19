# STEP 4 - Segmentation metrics implementation report

## Summary
- Status: `done`
- Generated at: `2026-06-19T16:21:15`
- Raw data modified: `false`
- Training/model preparation started: `false`
- Next step allowed: `false`

## Inputs referenced
- Hand-labeled index: `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/sen1floods11_handlabeled_index.csv`
- Hand-labeled audit: `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/sen1floods11_handlabeled_audit.csv`
- STEP 3 summary: `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/sen1floods11_step3_stats_summary.json`

## Implemented files
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/src/metrics/__init__.py`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/src/metrics/confusion.py`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/src/metrics/segmentation_metrics.py`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/scripts/05_evaluate_predictions.py`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/tests/test_segmentation_metrics.py`

## Confusion matrix behavior
- `compute_confusion_matrix(y_true, y_pred, num_classes=2, ignore_index=-1)` supports numpy arrays and torch tensors.
- Shapes are validated before metric computation.
- Pixels where `y_true == -1` are removed before label validation and counting.
- For binary segmentation, rows are ground truth and columns are predictions:

```text
[[TN, FP],
 [FN, TP]]
```

## Metric definitions
- Class `0`: non-water/background.
- Class `1`: water/flood.
- Accuracy: `(TP + TN) / valid_pixel_count`.
- Precision for water: `TP / (TP + FP)`.
- Recall for water: `TP / (TP + FN)`.
- F1 score for water: `2TP / (2TP + FP + FN)`.
- IoU per class: `TP_class / (TP_class + FP_class + FN_class)`.
- Water IoU: IoU for class `1`.
- Background IoU: IoU for class `0`.
- Mean IoU: mean of finite per-class IoUs.
- Support per class: number of valid ground-truth pixels for each class.

## Undefined and absent-class behavior
- Default `zero_division`: `nan`.
- All-ignored tiles return a zero confusion matrix and NaN metrics where denominators are undefined.
- If the water class is absent and there are no predicted water pixels, water precision/recall/F1/IoU are NaN by default.
- If the water class is absent but false positives exist, water IoU and precision are penalized as `0.0`.
- This avoids silently inflating metrics when a class is absent.

## Evaluation script
`scripts/05_evaluate_predictions.py` evaluates saved prediction masks against `LabelHand` masks without modifying raw data.

Example:

```powershell
python scripts/05_evaluate_predictions.py `
  --prediction-dir D:/flood_research/predictions/example_model `
  --manifest-csv reports/sen1floods11_handlabeled_index.csv `
  --audit-csv reports/sen1floods11_handlabeled_audit.csv `
  --output-csv reports/example_model_per_tile_metrics.csv `
  --output-grouped-csv reports/example_model_grouped_metrics.csv `
  --output-summary-json reports/example_model_metrics_summary.json `
  --split all
```

Supported filters:
- `--split train|valid|test|bolivia|all`
- repeated `--event-location <name>`
- `--threshold <value>` for probability/logit arrays
- `--zero-division nan|0|1`
- `--include-exclude-candidates`, while the five fully invalid LabelHand tiles remain excluded

Outputs:
- per-tile metrics CSV
- grouped global/split/event metrics CSV
- machine-readable JSON summary

## Validated cleaning/evaluation policy
- Keep `keep` samples.
- Keep `warning_review` samples, always ignoring pixels with label `-1`.
- Exclude `exclude_candidate` samples by default from supervised metrics.
- Filtering is manifest-based only.
- Keep `no_water` samples for false-positive control.
- Keep `high_invalid_ratio` samples if they are not 100% invalid; ignore `-1` pixels.
- Keep Bolivia as a separate holdout/OOD split.
- Always exclude the five fully invalid `LabelHand` samples:
  - `Ghana_234935`
  - `Ghana_26376`
  - `Ghana_277`
  - `Ghana_5079`
  - `Ghana_83483`

## Test results
- `python -m pytest tests/test_segmentation_metrics.py`
  - Result: `15 passed`
- `python -m py_compile scripts/05_evaluate_predictions.py src/metrics/confusion.py src/metrics/segmentation_metrics.py`
  - Result: `passed`
- `python scripts/05_evaluate_predictions.py --help`
  - Result: `passed`
- `python -m pytest`
  - Result: `66 passed, 2 warnings`
  - Warnings: rasterio `NotGeoreferencedWarning` in existing synthetic raster tests.

## Notes
- No prediction evaluation was run because no model prediction directory was provided in STEP 4.
- No raw Sen1Floods11 files were modified.
- No cleaning, indexing, visualization, model preparation, or training was started.

## Generated reports
- Local report: `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/STEP_4_segmentation_metrics_report.md`
- External report: `D:/flood_research/reports/STEP_4_segmentation_metrics_report.md`
- Pipeline status: `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/pipeline_status.json`

## Next recommended step
Human validation is required before STEP 5. The next step should define a TerraMind baseline reproduction plan, including exact inputs, split policy, evaluation protocol, and expected artifacts before any model work starts.
