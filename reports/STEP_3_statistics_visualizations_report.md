# STEP 3 - Sen1Floods11 statistics and visualizations report

## Summary
- Status: `done`
- Generated at: `2026-06-19T16:03:00`
- Raw data modified: `false`
- STEP 4 started: `false`
- Next step allowed: `false`

## Global statistics
- Hand-labeled samples: `446`
- Analysis training candidates (`keep` + `warning_review`): `441`
- Samples by split: `{'bolivia': 15, 'train': 252, 'valid': 89, 'test': 90}`
- Samples by status: `{'warning': 69, 'ok': 372, 'error': 5}`
- Samples by cleaning recommendation: `{'warning_review': 69, 'keep': 372, 'exclude_candidate': 5}`
- No-water samples: `52`
- High-invalid-ratio samples: `38`
- Only-invalid LabelHand samples: `5`
- Largest event/location groups: `[('USA', 69), ('India', 68), ('Paraguay', 67), ('Ghana', 53), ('Sri-Lanka', 42)]`

## Label distribution interpretation
- Total mask pixels: `116916224`
- Label distribution: `{'-1': {'count': 15932910, 'percent': 13.62763}, '0': {'count': 90277709, 'percent': 77.215724}, '1': {'count': 10705605, 'percent': 9.156646}}`
- Water percentage per tile summary: `{'min': 0.0, 'p25': 0.461292, 'median': 2.187729, 'mean': 9.156646, 'p75': 9.843922, 'max': 98.139954}`
- Invalid percentage per tile summary: `{'min': 0.0, 'p25': 0.002384, 'median': 0.948524, 'mean': 13.62763, 'p75': 19.217682, 'max': 100.0}`
- The hand-labeled subset is dominated by non-water pixels, so future metrics and thresholding should account for class imbalance.

## Image statistics interpretation
- Sentinel-1 band counts: `{'2': 446}`
- Sentinel-1 dtypes: `{'float32,float32': 446}`
- Sentinel-1 NaN ratio: `{'min': 0.0, 'p25': 0.0, 'median': 0.0, 'mean': 0.021483, 'p75': 0.0, 'max': 1.0}`
- Sentinel-1 Inf ratio: `{'min': 0.0, 'p25': 0.0, 'median': 0.0, 'mean': 0.0, 'p75': 0.0, 'max': 0.0}`
- Sentinel-1 extremes: min `{'tile_id': 'Bolivia_233925', 'band': 2, 'value': -103.87602996826172}`, max `{'tile_id': 'Spain_8372658', 'band': 1, 'value': 36.83202362060547}`
- Sentinel-2 band counts: `{'13': 446}`
- Sentinel-2 dtypes: `{'int16,int16,int16,int16,int16,int16,int16,int16,int16,int16,int16,int16,int16': 446}`
- Sentinel-2 NaN ratio: `{'min': 0.0, 'p25': 0.0, 'median': 0.0, 'mean': 0.0, 'p75': 0.0, 'max': 0.0}`
- Sentinel-2 Inf ratio: `{'min': 0.0, 'p25': 0.0, 'median': 0.0, 'mean': 0.0, 'p75': 0.0, 'max': 0.0}`
- Sentinel-2 extremes: min `{'tile_id': 'Bolivia_103757', 'band': 1, 'value': 0.0}`, max `{'tile_id': 'USA_504150', 'band': 8, 'value': 19199.0}`

## Split/event imbalance comments
- The official train/valid/test split is supplemented by a separate Bolivia holdout.
- Event/location counts are uneven; validation should preserve event-aware interpretation.
- The 5 fully invalid LabelHand masks are marked as `exclude_candidate` for supervised training candidates only.

## Generated figures
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/samples_per_split.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/samples_per_event_location.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/status_counts.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/cleaning_recommendation_counts.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/water_percentage_histogram.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/invalid_percentage_histogram.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/water_percentage_by_split.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/invalid_percentage_by_split.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/water_percentage_by_event_location.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/invalid_percentage_by_event_location.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/s1_minmax_distributions.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/s2_minmax_distributions.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_random_keep_USA_347609.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_random_keep_India_383430.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_random_keep_Bolivia_76104.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_random_keep_Nigeria_812045.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_random_warning_review_India_979278.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_random_warning_review_India_136196.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_random_warning_review_Ghana_194723.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_random_warning_review_Ghana_161233.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_no_water_Paraguay_36146.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_no_water_Somalia_61368.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_no_water_Pakistan_740461.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_no_water_Ghana_142312.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_high_invalid_ratio_USA_231124.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_high_invalid_ratio_Pakistan_528249.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_high_invalid_ratio_Ghana_132163.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_high_invalid_ratio_Bolivia_294583.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_exclude_candidate_Ghana_234935.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_exclude_candidate_Ghana_26376.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_exclude_candidate_Ghana_277.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_exclude_candidate_Ghana_5079.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_exclude_candidate_Ghana_83483.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_split_train_Ghana_103272.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_split_valid_Ghana_1033830.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_split_test_Ghana_1078550.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_split_bolivia_Bolivia_129334.png`

## Examples generated
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_random_keep_USA_347609.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_random_keep_India_383430.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_random_keep_Bolivia_76104.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_random_keep_Nigeria_812045.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_random_warning_review_India_979278.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_random_warning_review_India_136196.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_random_warning_review_Ghana_194723.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_random_warning_review_Ghana_161233.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_no_water_Paraguay_36146.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_no_water_Somalia_61368.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_no_water_Pakistan_740461.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_no_water_Ghana_142312.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_high_invalid_ratio_USA_231124.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_high_invalid_ratio_Pakistan_528249.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_high_invalid_ratio_Ghana_132163.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_high_invalid_ratio_Bolivia_294583.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_exclude_candidate_Ghana_234935.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_exclude_candidate_Ghana_26376.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_exclude_candidate_Ghana_277.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_exclude_candidate_Ghana_5079.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_exclude_candidate_Ghana_83483.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_split_train_Ghana_103272.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_split_valid_Ghana_1033830.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_split_test_Ghana_1078550.png`
- `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/figures/panel_split_bolivia_Bolivia_129334.png`

## Recommended cleaning policy to validate later
- Keep `keep` samples.
- Keep `warning_review` samples for exploratory statistics and visual inspection.
- Treat the 5 `exclude_candidate` samples as invalid for supervised training candidates.
- Do not delete or move any raw data; any future filtering should be manifest-based.

## Open questions before metrics/modeling
- Should Bolivia be used only as holdout, or also for robustness reporting?
- Should no-water samples remain in training to help background precision, or be balanced separately?
- Should high-invalid-ratio samples be weighted, masked, or excluded in training manifests?

## Generated files
- Local summary JSON: `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/sen1floods11_step3_stats_summary.json`
- External summary JSON: `D:/flood_research/reports/sen1floods11_step3_stats_summary.json`
- Local report: `C:/Users/ELEVES/Desktop/STAGE DE RECHERCHE/aiflow/reports/STEP_3_statistics_visualizations_report.md`
- External report: `D:/flood_research/reports/STEP_3_statistics_visualizations_report.md`

## Problems detected
- None

## Decision required before STEP 4
- Validate the cleaning policy and figure set.
- Do not implement segmentation metrics until this report is reviewed.
