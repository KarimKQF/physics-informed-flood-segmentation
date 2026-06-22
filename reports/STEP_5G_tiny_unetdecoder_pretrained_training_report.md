# STEP 5G - TerraMind Tiny Pretrained UNetDecoder Baseline Training

## Status

Result: PASS

STEP 5G compared the validated non-pretrained TerraMind tiny + UNetDecoder baseline against the official TerraMind tiny checkpoint only. TerraMind small/base, UPerNet, DARN, STURM-Flood training, physics loss, and raw-data modification were not started.

## Official Checkpoint

- Repo: `ibm-esa-geospatial/TerraMind-1.0-tiny`
- File: `TerraMind_v1_tiny.pt`
- URL: `https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-tiny/resolve/main/TerraMind_v1_tiny.pt`
- Local path: `E:/flood_research/experiments/terramind_baseline/checkpoints/pretrained/TerraMind_v1_tiny.pt`
- Size bytes: 211873402
- SHA256: `e56ea9ebcd4451078b9ca4893d5cd8a89bbee376ae16829c3e7fbbbc76de0eba`
- Download used: True
- Verified: True
- Load smoke test: True

## Configuration

- Run directory: `E:/flood_research/experiments/terramind_baseline/runs/step5g_tiny_unetdecoder_pretrained`
- Config path: `E:/flood_research/experiments/terramind_baseline/runs/step5g_tiny_unetdecoder_pretrained/configs/terramind_v1_tiny_unetdecoder_pretrained_train.yaml`
- Model: `terramind_v1_tiny`
- Decoder: `UNetDecoder`
- Backbone checkpoint path: `E:/flood_research/experiments/terramind_baseline/checkpoints/pretrained/TerraMind_v1_tiny.pt`
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
- Batch size: 2
- Precision requested/used: 32 / 32
- Optimizer: AdamW
- Learning rate: 0.0001
- Weight decay: 0.0001
- Seed: 42
- Trainable parameters: 8433330
- Total parameters: 8433330

## Training Curve

| Epoch | Train loss | Validation loss | Validation mIoU | Validation IoU water | Precision |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0.593315 | 0.5048264357751558 | 0.680087 | 0.477918 | 32 |
| 2 | 0.423984 | 0.27256976836630414 | 0.813331 | 0.670982 | 32 |
| 3 | 0.346167 | 0.3706990099635061 | 0.755547 | 0.587698 | 32 |
| 4 | 0.284154 | 0.1937709626741943 | 0.813649 | 0.669793 | 32 |
| 5 | 0.241735 | 0.1742810790807804 | 0.827025 | 0.694335 | 32 |

## Evaluation Metrics

| Split | Evaluated tiles | Accuracy | Precision water | Recall water | F1 water | IoU background | IoU water | mIoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 86 | 0.9630927741075576 | 0.8888038568445775 | 0.7603875572319512 | 0.8195960611919895 | 0.9597154412471002 | 0.6943352476607539 | 0.8270253444539271 |
| Test | 89 | 0.9645096761197477 | 0.8808961790036901 | 0.8282168160957032 | 0.8537446392102105 | 0.9604086085102179 | 0.7448119052825767 | 0.8526102568963974 |
| Bolivia | 15 | 0.9508054041142822 | 0.8344043805297474 | 0.8606418621829822 | 0.8473200574439602 | 0.9430284209761179 | 0.7350870143233763 | 0.8390577176497471 |

## STEP 5E vs STEP 5G

Positive differences mean STEP 5G is higher than STEP 5E.

| Split | 5E mIoU | 5G mIoU | Delta mIoU | 5E IoU water | 5G IoU water | Delta IoU water | 5E F1 water | 5G F1 water | Delta F1 water |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| valid | 0.7836170226051925 | 0.8270253444539271 | 0.043408321848734555 | 0.6258397167001033 | 0.6943352476607539 | 0.06849553096065064 | 0.7698664391965319 | 0.8195960611919895 | 0.04972962199545761 |
| test | 0.808079497821595 | 0.8526102568963974 | 0.044530759074802306 | 0.6734552396168814 | 0.7448119052825767 | 0.07135666566569532 | 0.8048679446855852 | 0.8537446392102105 | 0.048876694524625286 |
| bolivia | 0.747036803452412 | 0.8390577176497471 | 0.09202091419733516 | 0.603290625115386 | 0.7350870143233763 | 0.1317963892079903 | 0.7525655245092802 | 0.8473200574439602 | 0.09475453293468006 |

- Comparison JSON: `E:/flood_research/experiments/terramind_baseline/runs/step5g_tiny_unetdecoder_pretrained/metrics/step5e_vs_step5g_comparison.json`

## Runtime And GPU

- Machine OS: Windows-11-10.0.26200-SP0
- CPU: AMD64 Family 25 Model 33 Stepping 2, AuthenticAMD
- RAM GB: 31.92
- Device: NVIDIA GeForce RTX 4070
- Training elapsed seconds: 244.54
- Total elapsed seconds: 338.425
- GPU peak allocated/reserved MB: 969.52 / 1176.0

## Artifacts

- Training step metrics: `E:/flood_research/experiments/terramind_baseline/runs/step5g_tiny_unetdecoder_pretrained/metrics/training_step_metrics.csv`
- Training epoch metrics: `E:/flood_research/experiments/terramind_baseline/runs/step5g_tiny_unetdecoder_pretrained/metrics/training_epoch_metrics.csv`
- Validation predictions: `E:/flood_research/experiments/terramind_baseline/runs/step5g_tiny_unetdecoder_pretrained/predictions/valid`
- Test predictions: `E:/flood_research/experiments/terramind_baseline/runs/step5g_tiny_unetdecoder_pretrained/predictions/test`
- Bolivia predictions: `E:/flood_research/experiments/terramind_baseline/runs/step5g_tiny_unetdecoder_pretrained/predictions/bolivia`
- Summary JSON: `E:/flood_research/experiments/terramind_baseline/runs/step5g_tiny_unetdecoder_pretrained/metrics/step5g_summary.json`
- Log: `E:/flood_research/experiments/terramind_baseline/runs/step5g_tiny_unetdecoder_pretrained/logs/STEP_5G_tiny_unetdecoder_pretrained_training.log`

## Checkpoints

- best: `E:/flood_research/experiments/terramind_baseline/runs/step5g_tiny_unetdecoder_pretrained/checkpoints/best_checkpoint.pt` (97.046 MB)
- last: `E:/flood_research/experiments/terramind_baseline/runs/step5g_tiny_unetdecoder_pretrained/checkpoints/last_checkpoint.pt` (97.046 MB)

## Qualitative Panels

- `E:/flood_research/experiments/terramind_baseline/runs/step5g_tiny_unetdecoder_pretrained/reports/figures/valid/valid_Ghana_895194_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5g_tiny_unetdecoder_pretrained/reports/figures/valid/valid_Ghana_868803_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5g_tiny_unetdecoder_pretrained/reports/figures/valid/valid_Ghana_142312_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5g_tiny_unetdecoder_pretrained/reports/figures/valid/valid_Ghana_132163_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5g_tiny_unetdecoder_pretrained/reports/figures/valid/valid_Ghana_495107_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5g_tiny_unetdecoder_pretrained/reports/figures/test/test_Ghana_313799_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5g_tiny_unetdecoder_pretrained/reports/figures/test/test_Ghana_1078550_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5g_tiny_unetdecoder_pretrained/reports/figures/test/test_Ghana_97059_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5g_tiny_unetdecoder_pretrained/reports/figures/test/test_Ghana_359826_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5g_tiny_unetdecoder_pretrained/reports/figures/test/test_Ghana_319168_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5g_tiny_unetdecoder_pretrained/reports/figures/bolivia/bolivia_Bolivia_103757_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5g_tiny_unetdecoder_pretrained/reports/figures/bolivia/bolivia_Bolivia_129334_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5g_tiny_unetdecoder_pretrained/reports/figures/bolivia/bolivia_Bolivia_195474_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5g_tiny_unetdecoder_pretrained/reports/figures/bolivia/bolivia_Bolivia_23014_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5g_tiny_unetdecoder_pretrained/reports/figures/bolivia/bolivia_Bolivia_233925_panel.png`

## Problems And Warnings

- Rasterio may emit CPLE_IllegalArg BLOCKXSIZE warnings while writing prediction GeoTIFFs; metrics are trusted only when missing/error counts remain zero.

## Gate

Recommended next step: human validation of STEP 5G pretrained tiny metrics, checkpoints, comparison JSON, and qualitative panels.

Human validation required before starting STEP 5H — scale TerraMind baseline to small/base.
