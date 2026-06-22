# STEP 5I - TerraMind Base Pretrained UNetDecoder Baseline Training

## Status

Result: PASS

STEP 5I trained only the controlled TerraMind base + UNetDecoder pretrained baseline on Sen1Floods11. TerraMind-L, UPerNet, DARN, STURM-Flood training, physics loss, and raw-data modification were not started.

## Official Checkpoint

- Repo: `ibm-esa-geospatial/TerraMind-1.0-base`
- File: `TerraMind_v1_base.pt`
- URL: `https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base/resolve/main/TerraMind_v1_base.pt`
- Local path: `E:/flood_research/experiments/terramind_baseline/checkpoints/pretrained/TerraMind_v1_base.pt`
- Size bytes: 1518713958
- Size MB: 1448.358
- SHA256: `83c3a0938067c83867a46e564443c2fa38383bf4f966d931b11cb025b847d7ec`
- Download used: True
- Verified: True
- Pretrained smoke forward succeeded: True

## Smoke Forward

- Input shapes: `{'S2L1C': [1, 13, 512, 512], 'S1GRD': [1, 2, 512, 512]}`
- Mask shape: `[1, 512, 512]`
- Output shape: `[1, 2, 512, 512]`
- Loss: 0.955159604549408
- GPU allocated/reserved MB: 396.6 / 642.0
- GPU peak allocated/reserved MB: 550.6 / 642.0
- Elapsed seconds: 13.368

## Configuration

- Run directory: `E:/flood_research/experiments/terramind_baseline/runs/step5i_base_unetdecoder_pretrained`
- Config path: `E:/flood_research/experiments/terramind_baseline/runs/step5i_base_unetdecoder_pretrained/configs/terramind_v1_base_unetdecoder_pretrained_train.yaml`
- Model/backbone: `terramind_v1_base`
- Decoder: `UNetDecoder`
- Backbone checkpoint path: `E:/flood_research/experiments/terramind_baseline/checkpoints/pretrained/TerraMind_v1_base.pt`
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
- Trainable parameters: 95824770
- Total parameters: 95824770

## Training Curve

| Epoch | Train loss | Validation loss | Validation mIoU | Validation IoU water | Precision |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0.452735 | 0.24346565770509698 | 0.788665 | 0.627923 | 32 |
| 2 | 0.293508 | 0.2183825664976426 | 0.767276 | 0.592330 | 32 |
| 3 | 0.230807 | 0.16212232688179012 | 0.827344 | 0.697181 | 32 |
| 4 | 0.192912 | 0.19697325511152303 | 0.835632 | 0.713103 | 32 |
| 5 | 0.171529 | 0.13692580638049856 | 0.843346 | 0.722938 | 32 |

## Evaluation Metrics

| Split | Evaluated tiles | Accuracy | Precision water | Recall water | F1 water | IoU background | IoU water | mIoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 86 | 0.9668857301589452 | 0.903166908566125 | 0.7836812127252129 | 0.8391922536055492 | 0.9637538781292817 | 0.7229381921443395 | 0.8433460351368106 |
| Test | 89 | 0.9675648927077242 | 0.8917988935898631 | 0.8429364237806696 | 0.8666795015549114 | 0.9637424066187844 | 0.7647258676993773 | 0.8642341371590809 |
| Bolivia | 15 | 0.9604130670911478 | 0.9143674309743709 | 0.8279529705582328 | 0.8690172370030228 | 0.9544285525622298 | 0.76837354682597 | 0.8614010496940999 |

## STEP 5G Tiny vs STEP 5H Small vs STEP 5I Base

Positive differences mean STEP 5I base pretrained is higher than the comparison run.

| Split | 5G mIoU | 5H mIoU | 5I mIoU | Base-Tiny mIoU | Base-Small mIoU | 5G IoU water | 5H IoU water | 5I IoU water | Base-Tiny IoU water | Base-Small IoU water | 5G F1 water | 5H F1 water | 5I F1 water | Base-Tiny F1 water | Base-Small F1 water |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| valid | 0.8270253444539271 | 0.8298881661855735 | 0.8433460351368106 | 0.016320690682883554 | 0.013457868951237129 | 0.6943352476607539 | 0.699929443489551 | 0.7229381921443395 | 0.02860294448358558 | 0.023008748654788524 | 0.8195960611919895 | 0.8234805817031585 | 0.8391922536055492 | 0.0195961924135597 | 0.015711671902390645 |
| test | 0.8526102568963974 | 0.8585034615960692 | 0.8642341371590809 | 0.011623880262683572 | 0.005730675563011767 | 0.7448119052825767 | 0.7555793760722257 | 0.7647258676993773 | 0.019913962416800657 | 0.009146491627151643 | 0.8537446392102105 | 0.8607749514154017 | 0.8666795015549114 | 0.01293486234470087 | 0.0059045501395096345 |
| bolivia | 0.8390577176497471 | 0.845860540767879 | 0.8614010496940999 | 0.022343332044352793 | 0.015540508926220942 | 0.7350870143233763 | 0.7465930369813325 | 0.76837354682597 | 0.033286532502593724 | 0.02178050984463753 | 0.8473200574439602 | 0.8549135616293105 | 0.8690172370030228 | 0.021697179559062585 | 0.014103675373712266 |

- Comparison JSON: `E:/flood_research/experiments/terramind_baseline/runs/step5i_base_unetdecoder_pretrained/metrics/step5g_step5h_step5i_comparison.json`

## Runtime And GPU

- Machine OS: Windows-11-10.0.26200-SP0
- CPU: AMD64 Family 25 Model 33 Stepping 2, AuthenticAMD
- RAM GB: 31.92
- Device: NVIDIA GeForce RTX 4070
- Training elapsed seconds: 448.304
- Total elapsed seconds: 553.245
- GPU peak allocated/reserved MB: 2999.27 / 3156.0

## Artifacts

- Training step metrics: `E:/flood_research/experiments/terramind_baseline/runs/step5i_base_unetdecoder_pretrained/metrics/training_step_metrics.csv`
- Training epoch metrics: `E:/flood_research/experiments/terramind_baseline/runs/step5i_base_unetdecoder_pretrained/metrics/training_epoch_metrics.csv`
- Validation predictions: `E:/flood_research/experiments/terramind_baseline/runs/step5i_base_unetdecoder_pretrained/predictions/valid`
- Test predictions: `E:/flood_research/experiments/terramind_baseline/runs/step5i_base_unetdecoder_pretrained/predictions/test`
- Bolivia predictions: `E:/flood_research/experiments/terramind_baseline/runs/step5i_base_unetdecoder_pretrained/predictions/bolivia`
- Summary JSON: `E:/flood_research/experiments/terramind_baseline/runs/step5i_base_unetdecoder_pretrained/metrics/step5i_summary.json`
- Log: `E:/flood_research/experiments/terramind_baseline/runs/step5i_base_unetdecoder_pretrained/logs/STEP_5I_base_unetdecoder_pretrained_training.log`

## Checkpoints

- best: `E:/flood_research/experiments/terramind_baseline/runs/step5i_base_unetdecoder_pretrained/checkpoints/best_checkpoint.pt` (1098.04 MB)
- last: `E:/flood_research/experiments/terramind_baseline/runs/step5i_base_unetdecoder_pretrained/checkpoints/last_checkpoint.pt` (1098.04 MB)

## Qualitative Panels

- `E:/flood_research/experiments/terramind_baseline/runs/step5i_base_unetdecoder_pretrained/reports/figures/valid/valid_Ghana_895194_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5i_base_unetdecoder_pretrained/reports/figures/valid/valid_Ghana_868803_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5i_base_unetdecoder_pretrained/reports/figures/valid/valid_Ghana_142312_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5i_base_unetdecoder_pretrained/reports/figures/valid/valid_Ghana_132163_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5i_base_unetdecoder_pretrained/reports/figures/valid/valid_Ghana_495107_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5i_base_unetdecoder_pretrained/reports/figures/test/test_Ghana_313799_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5i_base_unetdecoder_pretrained/reports/figures/test/test_Ghana_1078550_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5i_base_unetdecoder_pretrained/reports/figures/test/test_Ghana_97059_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5i_base_unetdecoder_pretrained/reports/figures/test/test_Ghana_359826_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5i_base_unetdecoder_pretrained/reports/figures/test/test_Ghana_319168_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5i_base_unetdecoder_pretrained/reports/figures/bolivia/bolivia_Bolivia_103757_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5i_base_unetdecoder_pretrained/reports/figures/bolivia/bolivia_Bolivia_129334_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5i_base_unetdecoder_pretrained/reports/figures/bolivia/bolivia_Bolivia_195474_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5i_base_unetdecoder_pretrained/reports/figures/bolivia/bolivia_Bolivia_23014_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5i_base_unetdecoder_pretrained/reports/figures/bolivia/bolivia_Bolivia_233925_panel.png`

## Problems And Warnings

- Rasterio may emit CPLE_IllegalArg BLOCKXSIZE warnings while writing prediction GeoTIFFs; metrics are trusted only when missing/error counts remain zero.

## Gate

Recommended next step: human validation of STEP 5I base pretrained metrics, checkpoints, comparison JSON, and qualitative panels.

Human validation required before starting STEP 5J — decide between longer stable training, UPerNet adaptation, or full-model memory test.
