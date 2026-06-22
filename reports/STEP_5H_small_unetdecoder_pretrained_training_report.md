# STEP 5H - TerraMind Small Pretrained UNetDecoder Baseline Training

## Status

Result: PASS

STEP 5H trained only the controlled TerraMind small + UNetDecoder pretrained baseline on Sen1Floods11. TerraMind base, UPerNet, DARN, STURM-Flood training, physics loss, and raw-data modification were not started.

## Official Checkpoint

- Repo: `ibm-esa-geospatial/TerraMind-1.0-small`
- File: `TerraMind_v1_small.pt`
- URL: `https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-small/resolve/main/TerraMind_v1_small.pt`
- Local path: `E:/flood_research/experiments/terramind_baseline/checkpoints/pretrained/TerraMind_v1_small.pt`
- Size bytes: 504231662
- Size MB: 480.873
- SHA256: `755e9cce9483fd61334ef66c79f805406db5151a8b44a685c8fbbe023c684701`
- Download used: True
- Verified: True
- Pretrained smoke forward succeeded: True

## Smoke Forward

- Input shapes: `{'S2L1C': [1, 13, 512, 512], 'S1GRD': [1, 2, 512, 512]}`
- Mask shape: `[1, 512, 512]`
- Output shape: `[1, 2, 512, 512]`
- Loss: 0.8060178160667419
- GPU allocated/reserved MB: 133.48 / 238.0
- Elapsed seconds: 13.303

## Configuration

- Run directory: `E:/flood_research/experiments/terramind_baseline/runs/step5h_small_unetdecoder_pretrained`
- Config path: `E:/flood_research/experiments/terramind_baseline/runs/step5h_small_unetdecoder_pretrained/configs/terramind_v1_small_unetdecoder_pretrained_train.yaml`
- Model/backbone: `terramind_v1_small`
- Decoder: `UNetDecoder`
- Backbone checkpoint path: `E:/flood_research/experiments/terramind_baseline/checkpoints/pretrained/TerraMind_v1_small.pt`
- Dataset: Sen1Floods11 hand-labeled split
- Input modalities: S1GRD 2 channels, S2L1C 13 channels
- Classes: 0 background/non-water, 1 water/flood
- Ignore index: -1

## Dataset Splits

- Train tiles: 251
- Validation tiles: 86
- Test tiles: 89
- Bolivia holdout tiles: 15
- Excluded fully invalid samples: `Ghana_234935, Ghana_26376, Ghana_277, Ghana_5079, Ghana_83483`

## Hyperparameters

- Epochs: 5
- Batch size: 1
- Precision requested/used: 32 / 32
- Optimizer: AdamW
- Learning rate: 0.0001
- Weight decay: 0.0001
- Seed: 42
- Trainable parameters: 26642850
- Total parameters: 26642850

## Training Curve

| Epoch | Train loss | Validation loss | Validation mIoU | Validation IoU water | Precision |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0.372727 | 0.3496393694333667 | 0.743877 | 0.570566 | 32 |
| 2 | 0.249258 | 0.17406998204616658 | 0.825901 | 0.693156 | 32 |
| 3 | 0.197778 | 0.23363319743565844 | 0.822293 | 0.691639 | 32 |
| 4 | 0.177567 | 0.17715830732333646 | 0.769311 | 0.609611 | 32 |
| 5 | 0.159950 | 0.11693238563746053 | 0.829888 | 0.699929 | 32 |

## Evaluation Metrics

| Split | Evaluated tiles | Accuracy | Precision water | Recall water | F1 water | IoU background | IoU water | mIoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 86 | 0.9632855335561334 | 0.8762244034207924 | 0.7767260083884332 | 0.8234805817031585 | 0.959846888881596 | 0.699929443489551 | 0.8298881661855735 |
| Test | 89 | 0.9655369034437996 | 0.869927157138485 | 0.8518133152202505 | 0.8607749514154017 | 0.9614275471199127 | 0.7555793760722257 | 0.8585034615960692 |
| Bolivia | 15 | 0.9527643868241152 | 0.8335383567909408 | 0.8774139083330402 | 0.8549135616293105 | 0.9451280445544253 | 0.7465930369813325 | 0.845860540767879 |

## STEP 5G Tiny vs STEP 5H Small

Positive differences mean STEP 5H small pretrained is higher than STEP 5G tiny pretrained.

| Split | 5G mIoU | 5H mIoU | Delta mIoU | 5G IoU water | 5H IoU water | Delta IoU water | 5G F1 water | 5H F1 water | Delta F1 water |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| valid | 0.8270253444539271 | 0.8298881661855735 | 0.0028628217316464255 | 0.6943352476607539 | 0.699929443489551 | 0.005594195828797055 | 0.8195960611919895 | 0.8234805817031585 | 0.003884520511169054 |
| test | 0.8526102568963974 | 0.8585034615960692 | 0.005893204699671806 | 0.7448119052825767 | 0.7555793760722257 | 0.010767470789649014 | 0.8537446392102105 | 0.8607749514154017 | 0.007030312205191236 |
| bolivia | 0.8390577176497471 | 0.845860540767879 | 0.006802823118131851 | 0.7350870143233763 | 0.7465930369813325 | 0.011506022657956194 | 0.8473200574439602 | 0.8549135616293105 | 0.007593504185350319 |

- Comparison JSON: `E:/flood_research/experiments/terramind_baseline/runs/step5h_small_unetdecoder_pretrained/metrics/step5g_vs_step5h_comparison.json`

## Runtime And GPU

- Machine OS: Windows-11-10.0.26200-SP0
- CPU: AMD64 Family 25 Model 33 Stepping 2, AuthenticAMD
- RAM GB: 31.92
- Device: NVIDIA GeForce RTX 4070
- Training elapsed seconds: 306.165
- Total elapsed seconds: 375.227
- GPU peak allocated/reserved MB: 1068.07 / 1208.0

## Artifacts

- Training step metrics: `E:/flood_research/experiments/terramind_baseline/runs/step5h_small_unetdecoder_pretrained/metrics/training_step_metrics.csv`
- Training epoch metrics: `E:/flood_research/experiments/terramind_baseline/runs/step5h_small_unetdecoder_pretrained/metrics/training_epoch_metrics.csv`
- Validation predictions: `E:/flood_research/experiments/terramind_baseline/runs/step5h_small_unetdecoder_pretrained/predictions/valid`
- Test predictions: `E:/flood_research/experiments/terramind_baseline/runs/step5h_small_unetdecoder_pretrained/predictions/test`
- Bolivia predictions: `E:/flood_research/experiments/terramind_baseline/runs/step5h_small_unetdecoder_pretrained/predictions/bolivia`
- Summary JSON: `E:/flood_research/experiments/terramind_baseline/runs/step5h_small_unetdecoder_pretrained/metrics/step5h_summary.json`
- Log: `E:/flood_research/experiments/terramind_baseline/runs/step5h_small_unetdecoder_pretrained/logs/STEP_5H_small_unetdecoder_pretrained_training.log`

## Checkpoints

- best: `E:/flood_research/experiments/terramind_baseline/runs/step5h_small_unetdecoder_pretrained/checkpoints/best_checkpoint.pt` (305.744 MB)
- last: `E:/flood_research/experiments/terramind_baseline/runs/step5h_small_unetdecoder_pretrained/checkpoints/last_checkpoint.pt` (305.744 MB)

## Qualitative Panels

- `E:/flood_research/experiments/terramind_baseline/runs/step5h_small_unetdecoder_pretrained/reports/figures/valid/valid_Ghana_895194_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5h_small_unetdecoder_pretrained/reports/figures/valid/valid_Ghana_868803_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5h_small_unetdecoder_pretrained/reports/figures/valid/valid_Ghana_142312_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5h_small_unetdecoder_pretrained/reports/figures/valid/valid_Ghana_132163_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5h_small_unetdecoder_pretrained/reports/figures/valid/valid_Ghana_495107_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5h_small_unetdecoder_pretrained/reports/figures/test/test_Ghana_313799_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5h_small_unetdecoder_pretrained/reports/figures/test/test_Ghana_1078550_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5h_small_unetdecoder_pretrained/reports/figures/test/test_Ghana_97059_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5h_small_unetdecoder_pretrained/reports/figures/test/test_Ghana_359826_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5h_small_unetdecoder_pretrained/reports/figures/test/test_Ghana_319168_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5h_small_unetdecoder_pretrained/reports/figures/bolivia/bolivia_Bolivia_103757_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5h_small_unetdecoder_pretrained/reports/figures/bolivia/bolivia_Bolivia_129334_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5h_small_unetdecoder_pretrained/reports/figures/bolivia/bolivia_Bolivia_195474_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5h_small_unetdecoder_pretrained/reports/figures/bolivia/bolivia_Bolivia_23014_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5h_small_unetdecoder_pretrained/reports/figures/bolivia/bolivia_Bolivia_233925_panel.png`

## Problems And Warnings

- Rasterio may emit CPLE_IllegalArg BLOCKXSIZE warnings while writing prediction GeoTIFFs; metrics are trusted only when missing/error counts remain zero.

## Gate

Recommended next step: human validation of STEP 5H small pretrained metrics, checkpoints, comparison JSON, and qualitative panels.

Human validation required before starting STEP 5I — scale TerraMind baseline to base or longer stable small training.
