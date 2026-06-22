# STEP 5M - TerraMind-L Pretrained UPerNet Training

## Status

Result: PASS

STEP 5M trained only the controlled classical-loss SOTA-candidate baseline: TerraMind-L pretrained + UPerNet on Sen1Floods11. Physics-informed loss, DARN, STURM-Flood training, raw-data modification, and longer training were not started.

## Checkpoint

- Source: `ibm-esa-geospatial/TerraMind-1.0-large`
- File: `TerraMind_v1_large.pt`
- Path: `E:/flood_research/experiments/terramind_baseline/checkpoints/pretrained/TerraMind_v1_large.pt`
- Size bytes: 3787890978
- Size MB: 3612.414
- SHA256: `a1c6b567ce6862c7ac07181551add840edce52f652abccfe5f17d23544060f81`
- Size verified: True
- SHA256 verified: True
- Download used in STEP 5M: False

## Configuration

- Config path: `E:/flood_research/experiments/terramind_baseline/runs/step5m_terramind_l_upernet_pretrained/configs/terramind_l_upernet_pretrained_train.yaml`
- Model/backbone: `terramind_v1_large`
- Decoder: `UperNetDecoder`
- Neck path: `SelectIndices([2, 5, 8, 11]) -> ReshapeTokensToImage(remove_cls_token=false) -> LearnedInterpolateToPyramidal`
- Decoder config: `channels=256`, `pool_scales=[1, 2, 3, 6]`, `align_corners=true`, `scale_modules=false`
- Ignore index: -1
- BatchNorm policy: BatchNorm modules are kept in eval mode and their affine parameters are frozen during batch-size-1 UPerNet smoke/training, because the default PSP pool scale 1 creates a 1x1 feature map that BatchNorm cannot train on with a single sample.

## Dataset

- Train tiles: 251
- Validation tiles: 86
- Test tiles: 89
- Bolivia holdout tiles: 15
- Excluded fully invalid samples: `Ghana_234935, Ghana_26376, Ghana_277, Ghana_5079, Ghana_83483`
- Raw data modified: False

## Pre-Training Smoke

- Passed: True
- Input shapes: `{'S2L1C': [1, 13, 512, 512], 'S1GRD': [1, 2, 512, 512]}`
- Output shape: `[1, 2, 512, 512]`
- Loss: 474.5939025878906
- Backward OK: True
- Optimizer step attempted/OK: True / True
- GPU allocated/reserved MB: 2581.53 / 6074.0
- GPU peak allocated/reserved MB: 5322.84 / 6074.0
- Elapsed seconds: 17.491

## Hyperparameters

- Epochs: 5
- Batch size: 1
- Precision requested/used: 32 / 32
- Optimizer: AdamW
- Learning rate: 0.0001
- Weight decay: 0.0001
- Seed: 42
- Trainable parameters: 321000194
- Total parameters: 321007362
- Training duration seconds: 706.45

## Training Curve

| Epoch | Train loss | Validation loss | Validation mIoU | Validation IoU water | Precision |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 14.885288 | 0.6550117171348883 | 0.509590 | 0.287717 | 32 |
| 2 | 0.202064 | 0.277612010630936 | 0.461487 | 0.030364 | 32 |
| 3 | 0.148934 | 0.1335397732206714 | 0.787761 | 0.623909 | 32 |
| 4 | 0.182177 | 0.16510913731006424 | 0.803859 | 0.653858 | 32 |
| 5 | 0.175185 | 0.12531084912370394 | 0.807700 | 0.660359 | 32 |

## Evaluation Metrics

| Split | Evaluated tiles | Accuracy | Precision water | Recall water | F1 water | IoU background | IoU water | mIoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 86 | 0.9586539852104426 | 0.8750471323422535 | 0.7291121533961534 | 0.7954415807174778 | 0.9550396876001299 | 0.6603594877459502 | 0.8076995876730401 |
| Test | 89 | 0.959029586983554 | 0.8692514973391575 | 0.7914680677027132 | 0.8285382040758548 | 0.9545284149736184 | 0.7072686509782726 | 0.8308985329759455 |
| Bolivia | 15 | 0.9455735464107692 | 0.7996618036736225 | 0.8764224031798515 | 0.8362843786022733 | 0.9367862456979331 | 0.7186329402348496 | 0.8277095929663913 |

## STEP 5I vs STEP 5K vs STEP 5M

Positive differences mean STEP 5M TerraMind-L UPerNet is higher than the comparison run.

| Split | 5I mIoU | 5K mIoU | 5M mIoU | 5M-5I mIoU | 5M-5K mIoU | 5I IoU water | 5K IoU water | 5M IoU water | 5M-5I IoU water | 5M-5K IoU water | 5I F1 water | 5K F1 water | 5M F1 water | 5M-5I F1 | 5M-5K F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| valid | 0.8433460351368106 | 0.8059480274105153 | 0.8076995876730401 | -0.03564644746377055 | 0.001751560262524765 | 0.7229381921443395 | 0.656033029568658 | 0.6603594877459502 | -0.0625787043983893 | 0.004326458177292158 | 0.8391922536055492 | 0.7922946195578394 | 0.7954415807174778 | -0.04375067288807133 | 0.0031469611596384173 |
| test | 0.8642341371590809 | 0.823168884398115 | 0.8308985329759455 | -0.033335604183135414 | 0.007729648577830539 | 0.7647258676993773 | 0.6924485201401742 | 0.7072686509782726 | -0.05745721672110471 | 0.014820130838098411 | 0.8666795015549114 | 0.818280156707896 | 0.8285382040758548 | -0.03814129747905659 | 0.010258047367958767 |
| bolivia | 0.8614010496940999 | 0.8309888600687256 | 0.8277095929663913 | -0.03369145672770857 | -0.0032792671023342823 | 0.76837354682597 | 0.7206744685932649 | 0.7186329402348496 | -0.04974060659112045 | -0.002041528358415312 | 0.8690172370030228 | 0.8376650920873503 | 0.8362843786022733 | -0.032732858400749465 | -0.0013807134850769565 |

- Comparison JSON: `E:/flood_research/experiments/terramind_baseline/runs/step5m_terramind_l_upernet_pretrained/metrics/step5i_step5k_step5m_comparison.json`
- SOTA-candidate baseline recommendation: no, STEP 5M does not outperform the current STEP 5I base UNetDecoder baseline across split mIoU values

## Runtime And Artifacts

- Machine OS: Windows-11-10.0.26200-SP0
- CPU: AMD64 Family 25 Model 33 Stepping 2, AuthenticAMD
- RAM GB: None
- Device: NVIDIA GeForce RTX 4070
- Total elapsed seconds: 798.624
- GPU peak allocated/reserved MB: 8227.81 / 9408.0
- Training step metrics: `E:/flood_research/experiments/terramind_baseline/runs/step5m_terramind_l_upernet_pretrained/metrics/training_step_metrics.csv`
- Training epoch metrics: `E:/flood_research/experiments/terramind_baseline/runs/step5m_terramind_l_upernet_pretrained/metrics/training_epoch_metrics.csv`
- Validation predictions: `E:/flood_research/experiments/terramind_baseline/runs/step5m_terramind_l_upernet_pretrained/predictions/valid`
- Test predictions: `E:/flood_research/experiments/terramind_baseline/runs/step5m_terramind_l_upernet_pretrained/predictions/test`
- Bolivia predictions: `E:/flood_research/experiments/terramind_baseline/runs/step5m_terramind_l_upernet_pretrained/predictions/bolivia`
- Summary JSON: `E:/flood_research/experiments/terramind_baseline/runs/step5m_terramind_l_upernet_pretrained/metrics/step5m_summary.json`
- Log: `E:/flood_research/experiments/terramind_baseline/runs/step5m_terramind_l_upernet_pretrained/logs/STEP_5M_terramind_l_upernet_pretrained_training.log`

## Checkpoints

- best: `E:/flood_research/experiments/terramind_baseline/runs/step5m_terramind_l_upernet_pretrained/checkpoints/best_checkpoint.pt` (2523.545 MB)
- last: `E:/flood_research/experiments/terramind_baseline/runs/step5m_terramind_l_upernet_pretrained/checkpoints/last_checkpoint.pt` (2523.545 MB)

## Qualitative Panels

- `E:/flood_research/experiments/terramind_baseline/runs/step5m_terramind_l_upernet_pretrained/reports/figures/valid/valid_Ghana_895194_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5m_terramind_l_upernet_pretrained/reports/figures/valid/valid_Ghana_868803_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5m_terramind_l_upernet_pretrained/reports/figures/valid/valid_Ghana_142312_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5m_terramind_l_upernet_pretrained/reports/figures/valid/valid_Ghana_132163_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5m_terramind_l_upernet_pretrained/reports/figures/valid/valid_Ghana_495107_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5m_terramind_l_upernet_pretrained/reports/figures/test/test_Ghana_313799_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5m_terramind_l_upernet_pretrained/reports/figures/test/test_Ghana_1078550_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5m_terramind_l_upernet_pretrained/reports/figures/test/test_Ghana_97059_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5m_terramind_l_upernet_pretrained/reports/figures/test/test_Ghana_359826_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5m_terramind_l_upernet_pretrained/reports/figures/test/test_Ghana_319168_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5m_terramind_l_upernet_pretrained/reports/figures/bolivia/bolivia_Bolivia_103757_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5m_terramind_l_upernet_pretrained/reports/figures/bolivia/bolivia_Bolivia_129334_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5m_terramind_l_upernet_pretrained/reports/figures/bolivia/bolivia_Bolivia_195474_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5m_terramind_l_upernet_pretrained/reports/figures/bolivia/bolivia_Bolivia_23014_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5m_terramind_l_upernet_pretrained/reports/figures/bolivia/bolivia_Bolivia_233925_panel.png`

## Guardrails

- Physics-informed loss started: False
- DARN started: False
- STURM-Flood training started: False
- Raw data modified: False
- Longer training launched: False

## Problems And Warnings

- BatchNorm modules are kept in eval mode and their affine parameters are frozen during batch-size-1 UPerNet smoke/training, because the default PSP pool scale 1 creates a 1x1 feature map that BatchNorm cannot train on with a single sample.
- Rasterio may emit CPLE_IllegalArg BLOCKXSIZE warnings while writing prediction GeoTIFFs; metrics are trusted only when missing/error counts remain zero.

## Result

- STEP 5M passed: True
- Recommended next step: wait for human validation.

Human validation required before starting STEP 5N — decide whether to freeze the SOTA baseline and prepare physics-informed loss, or run longer classical training.
