# STEP 5F - Tiny Baseline Sanity And Reproducibility Check

## Status

STEP 5F is complete. This was a quality-control gate only. TerraMind small/base, UPerNet, DARN, STURM-Flood training, physics loss, and raw-data modification were not started.

Result: PASS

## STEP 5E Artifact Inspection

- Config: `E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/configs/terramind_v1_tiny_unetdecoder_train.yaml`
- Best checkpoint: `E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/checkpoints/best_checkpoint.pt` (97.046 MB)
- Last checkpoint: `E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/checkpoints/last_checkpoint.pt` (97.046 MB)
- Valid predictions: 86 / expected 86
- Test predictions: 89 / expected 89
- Bolivia predictions: 15 / expected 15
- Existing STEP 5E qualitative panels: 9

## Integrity Checks

- Split integrity OK: True
- Prediction/label matching OK: True
- Metric recomputation OK: True
- Fully invalid LabelHand samples excluded: `Ghana_234935, Ghana_26376, Ghana_277, Ghana_5079, Ghana_83483`
- Ignore index: -1

## Recomputed Metrics

| Split | Tiles | Accuracy | Precision water | Recall water | F1 water | IoU background | IoU water | mIoU | Max diff vs 5E |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| valid | 86 | 0.9466263770511796 | 0.733756540473992 | 0.8097144044637011 | 0.7698664391965319 | 0.9413943285102817 | 0.6258397167001033 | 0.7836170226051925 | 0.0 |
| test | 89 | 0.9487587271797595 | 0.7684119279148122 | 0.8449554401794785 | 0.8048679446855852 | 0.9427037560263088 | 0.6734552396168814 | 0.808079497821595 | 0.0 |
| bolivia | 15 | 0.9063391467022803 | 0.6476666603243463 | 0.8980112736993915 | 0.7525655245092802 | 0.8907829817894379 | 0.603290625115386 | 0.747036803452412 | 0.0 |

## FP32 Stability Check

- Run directory: `E:/flood_research/experiments/terramind_baseline/runs/step5f_tiny_unetdecoder_fp32_check`
- Config: `E:/flood_research/experiments/terramind_baseline/runs/step5f_tiny_unetdecoder_fp32_check/configs/terramind_v1_tiny_unetdecoder_fp32_check.yaml`
- Precision: 32
- Batch size: 2
- Epochs: 2
- OOM fallback used: False
- Finite losses: True
- Loss reasonable/decreased gate: True
- Validation inference clean: True
- FP32 check OK: True
- Best checkpoint: `E:/flood_research/experiments/terramind_baseline/runs/step5f_tiny_unetdecoder_fp32_check/checkpoints/best_checkpoint.pt` (97.046 MB)
- Last checkpoint: `E:/flood_research/experiments/terramind_baseline/runs/step5f_tiny_unetdecoder_fp32_check/checkpoints/last_checkpoint.pt` (97.046 MB)

| Epoch | Train loss | Validation loss | Validation mIoU | Validation IoU water |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.4398538485274719 | 0.3009642170612608 | 0.7776089881356405 | 0.6069940696183664 |
| 2 | 0.32151580644398775 | 0.248659137347661 | 0.7999346690466349 | 0.6501703862863584 |

## Pretrained Tiny Availability

- Repo: `ibm-esa-geospatial/TerraMind-1.0-tiny`
- File: `TerraMind_v1_tiny.pt`
- Available without downloading: True
- Local cached: False
- Remote metadata checked: True
- Expected size MB: 202.058
- Source: `https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-tiny/resolve/main/TerraMind_v1_tiny.pt`
- SHA256: `e56ea9ebcd4451078b9ca4893d5cd8a89bbee376ae16829c3e7fbbbc76de0eba`
- Note: Official tiny checkpoint is listed by the Hugging Face model API with blob metadata; metadata checked only and file was not downloaded.

## Additional Qualitative Panels

- `E:/flood_research/experiments/terramind_baseline/runs/step5f_sanity_recompute/reports/figures/valid/valid_Ghana_895194_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5f_sanity_recompute/reports/figures/valid/valid_Ghana_868803_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5f_sanity_recompute/reports/figures/valid/valid_Ghana_142312_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5f_sanity_recompute/reports/figures/valid/valid_Ghana_132163_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5f_sanity_recompute/reports/figures/valid/valid_Ghana_495107_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5f_sanity_recompute/reports/figures/test/test_Ghana_313799_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5f_sanity_recompute/reports/figures/test/test_Ghana_1078550_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5f_sanity_recompute/reports/figures/test/test_Ghana_97059_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5f_sanity_recompute/reports/figures/test/test_Ghana_359826_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5f_sanity_recompute/reports/figures/test/test_Ghana_319168_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5f_sanity_recompute/reports/figures/bolivia/bolivia_Bolivia_103757_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5f_sanity_recompute/reports/figures/bolivia/bolivia_Bolivia_129334_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5f_sanity_recompute/reports/figures/bolivia/bolivia_Bolivia_195474_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5f_sanity_recompute/reports/figures/bolivia/bolivia_Bolivia_23014_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5f_sanity_recompute/reports/figures/bolivia/bolivia_Bolivia_233925_panel.png`

## Warnings

- STEP 5E remains a non-pretrained baseline unless official TerraMind tiny weights are explicitly downloaded and a new pretrained run is authorized.
- STEP 5E had AMP instability; STEP 5F therefore used a separate 2-epoch precision-32 stability check.
- Repository pytest collection is still known to fail because legacy tests import missing module urban_runoff.data.

## Trust And Gate

- STEP 5E metrics trusted: True
- Scaling to TerraMind small/base approved by automation: False
- Recommended next step: human validation of this sanity report before any scaling or longer stable training.

Human validation required before starting STEP 5G â€” scale TerraMind baseline to small/base or longer stable training.
