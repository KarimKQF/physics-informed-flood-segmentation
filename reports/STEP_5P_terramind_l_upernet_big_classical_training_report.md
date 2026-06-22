# STEP 5P - SOTA-Candidate Big Classical Training

## Status

Result: PASS

STEP 5P gave the SOTA-candidate model a fairer classical-loss budget before deciding which validated baseline should receive the first physics-informed topographic loss. STEP 5M was only a 5-epoch controlled run, so it was not sufficient to conclude definitively that TerraMind-L + UPerNet was inferior.

Physics-informed loss, DARN, STURM-Flood training, architecture experiments beyond TerraMind-L + UPerNet, raw-data modification, and deletion/overwrite of STEP 5N outputs were not started.

## Checkpoint

- Source: `ibm-esa-geospatial/TerraMind-1.0-large`
- File: `TerraMind_v1_large.pt`
- Path: `E:/flood_research/experiments/terramind_baseline/checkpoints/pretrained/TerraMind_v1_large.pt`
- Size bytes: 3787890978
- Size MB: 3612.414
- SHA256: `a1c6b567ce6862c7ac07181551add840edce52f652abccfe5f17d23544060f81`
- Size verified: True
- SHA256 verified: True
- Download used in STEP 5P: False

## Configuration

- Config path: `E:/flood_research/experiments/terramind_baseline/runs/step5p_terramind_l_upernet_big_classical_training/configs/terramind_l_upernet_big_classical_train.yaml`
- Model/backbone: `terramind_v1_large`
- Decoder: `UperNetDecoder`
- Neck path: `SelectIndices([2, 5, 8, 11]) -> ReshapeTokensToImage(remove_cls_token=false) -> LearnedInterpolateToPyramidal`
- Decoder config: `channels=256`, `pool_scales=[1, 2, 3, 6]`, `align_corners=true`, `scale_modules=false`
- Ignore index: -1
- BatchNorm policy: BatchNorm modules are kept in eval mode and their affine parameters are frozen during batch-size-1 UPerNet smoke/training, because the default PSP pool scale 1 creates a 1x1 feature map that BatchNorm cannot train on with a single sample.
- Scheduler: ReduceLROnPlateau(mode=max, factor=0.5, patience=3)
- Early stopping: monitor validation mIoU, patience 15, min epochs 30

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
- Elapsed seconds: 39.859

## Hyperparameters

- Max epochs: 80
- Epochs completed: 30
- Best epoch: 15
- Early stopped: True
- Early stop reason: validation_miou did not improve for 15 epochs after epoch 15
- Batch size: 1
- Precision requested/used: 32 / 32
- Optimizer: AdamW
- Learning rate: 0.0001
- Weight decay: 0.0001
- Seed: 42
- Trainable parameters: 321000194
- Total parameters: 321007362
- Training duration seconds: 3689.272

## Training Curve

| Epoch | Train loss | Validation loss | Validation mIoU | Validation IoU water | Validation F1 water | LR | GPU alloc MB | GPU reserved MB | GPU peak alloc MB | GPU peak reserved MB | Epoch sec | Precision | Improved |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 14.673007 | 0.152993 | 0.780871 | 0.617137 | 0.763246 | 0.00010000 | 4459.81 | 8536.0 | 7867.5 | 8536.0 | 158.461 | 32 | True |
| 2 | 0.186855 | 0.178543 | 0.743966 | 0.543225 | 0.704013 | 0.00010000 | 4459.81 | 8536.0 | 7867.5 | 8536.0 | 118.348 | 32 | False |
| 3 | 0.153745 | 0.146246 | 0.755679 | 0.565209 | 0.722215 | 0.00010000 | 4459.81 | 8536.0 | 7867.5 | 8536.0 | 118.609 | 32 | False |
| 4 | 0.139061 | 0.120343 | 0.812637 | 0.673678 | 0.805027 | 0.00010000 | 4460.06 | 8536.0 | 7867.5 | 8536.0 | 118.473 | 32 | True |
| 5 | 0.166526 | 0.119823 | 0.781754 | 0.612362 | 0.759583 | 0.00010000 | 4459.81 | 8536.0 | 7867.5 | 8536.0 | 119.337 | 32 | False |
| 6 | 0.148652 | 0.114433 | 0.823816 | 0.690674 | 0.817040 | 0.00010000 | 4459.81 | 8536.0 | 7867.5 | 8536.0 | 118.792 | 32 | True |
| 7 | 0.160010 | 0.112451 | 0.801133 | 0.648450 | 0.786739 | 0.00010000 | 4459.81 | 8536.0 | 7867.5 | 8536.0 | 119.499 | 32 | False |
| 8 | 0.217364 | 0.124799 | 0.813417 | 0.674455 | 0.805582 | 0.00010000 | 4460.06 | 8536.0 | 7867.5 | 8536.0 | 119.757 | 32 | False |
| 9 | 0.176498 | 0.111140 | 0.807733 | 0.660235 | 0.795351 | 0.00010000 | 4459.81 | 8536.0 | 7867.5 | 8536.0 | 119.336 | 32 | False |
| 10 | 0.158394 | 0.121500 | 0.761718 | 0.574972 | 0.730136 | 0.00010000 | 4459.81 | 8536.0 | 7867.5 | 8536.0 | 119.126 | 32 | False |
| 11 | 0.143964 | 0.109216 | 0.835874 | 0.712134 | 0.831867 | 0.00005000 | 4459.81 | 8536.0 | 7867.5 | 8536.0 | 118.939 | 32 | True |
| 12 | 0.125131 | 0.119511 | 0.819572 | 0.682820 | 0.811519 | 0.00005000 | 4460.06 | 8536.0 | 7867.5 | 8536.0 | 119.696 | 32 | False |
| 13 | 0.135015 | 0.104819 | 0.825678 | 0.693618 | 0.819096 | 0.00005000 | 4459.81 | 8536.0 | 7867.5 | 8536.0 | 119.061 | 32 | False |
| 14 | 0.119478 | 0.106154 | 0.818271 | 0.678470 | 0.808439 | 0.00005000 | 4459.81 | 8536.0 | 7867.5 | 8536.0 | 119.094 | 32 | False |
| 15 | 0.125804 | 0.096800 | 0.846118 | 0.728556 | 0.842965 | 0.00005000 | 4459.81 | 8536.0 | 7867.5 | 8536.0 | 119.596 | 32 | True |
| 16 | 0.112524 | 0.243584 | 0.779282 | 0.624925 | 0.769174 | 0.00005000 | 4460.06 | 8536.0 | 7867.5 | 8536.0 | 119.653 | 32 | False |
| 17 | 0.128114 | 0.183458 | 0.818281 | 0.678205 | 0.808251 | 0.00005000 | 4459.81 | 8536.0 | 7867.5 | 8536.0 | 119.267 | 32 | False |
| 18 | 0.137363 | 0.110736 | 0.816258 | 0.673971 | 0.805236 | 0.00005000 | 4459.81 | 8536.0 | 7867.5 | 8536.0 | 118.938 | 32 | False |
| 19 | 0.129957 | 0.111216 | 0.808306 | 0.659447 | 0.794779 | 0.00005000 | 4459.81 | 8536.0 | 7867.5 | 8536.0 | 119.339 | 32 | False |
| 20 | 0.112420 | 0.210684 | 0.829785 | 0.705823 | 0.827546 | 0.00002500 | 4460.06 | 8536.0 | 7867.5 | 8536.0 | 119.495 | 32 | False |
| 21 | 0.117175 | 0.097200 | 0.843382 | 0.725421 | 0.840863 | 0.00002500 | 4459.81 | 8536.0 | 7867.5 | 8536.0 | 119.694 | 32 | False |
| 22 | 0.106947 | 0.106644 | 0.826370 | 0.692133 | 0.818060 | 0.00002500 | 4459.81 | 8536.0 | 7867.5 | 8536.0 | 119.698 | 32 | False |
| 23 | 0.117736 | 0.091868 | 0.845998 | 0.727823 | 0.842474 | 0.00002500 | 4459.81 | 8536.0 | 7867.5 | 8536.0 | 119.255 | 32 | False |
| 24 | 0.098055 | 0.089850 | 0.834689 | 0.707315 | 0.828570 | 0.00001250 | 4460.06 | 8536.0 | 7867.5 | 8536.0 | 119.242 | 32 | False |
| 25 | 0.098387 | 0.101772 | 0.795586 | 0.635579 | 0.777191 | 0.00001250 | 4459.81 | 8536.0 | 7867.5 | 8536.0 | 119.624 | 32 | False |
| 26 | 0.097027 | 0.087557 | 0.845955 | 0.727499 | 0.842257 | 0.00001250 | 4459.81 | 8536.0 | 7867.5 | 8536.0 | 119.045 | 32 | False |
| 27 | 0.095590 | 0.087885 | 0.837132 | 0.711496 | 0.831431 | 0.00001250 | 4459.81 | 8536.0 | 7867.5 | 8536.0 | 118.761 | 32 | False |
| 28 | 0.090752 | 0.086327 | 0.839555 | 0.715633 | 0.834249 | 0.00000625 | 4460.06 | 8536.0 | 7867.5 | 8536.0 | 118.697 | 32 | False |
| 29 | 0.091357 | 0.085921 | 0.843145 | 0.722563 | 0.838939 | 0.00000625 | 4459.81 | 8536.0 | 7867.5 | 8536.0 | 118.578 | 32 | False |
| 30 | 0.089120 | 0.088772 | 0.829313 | 0.696862 | 0.821353 | 0.00000625 | 4459.81 | 8536.0 | 7867.5 | 8536.0 | 118.69 | 32 | False |

## Evaluation Metrics

| Split | Evaluated tiles | Accuracy | Precision water | Recall water | F1 water | IoU background | IoU water | mIoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 86 | 0.9669059817267788 | 0.8839366840887906 | 0.8056229763519477 | 0.8429648493815476 | 0.9636798692887579 | 0.7285559552196581 | 0.846117912254208 |
| Test | 89 | 0.9623499935444934 | 0.8778700156573122 | 0.8119232251575444 | 0.8436097827422702 | 0.9580946289687092 | 0.7295199926049281 | 0.8438073107868187 |
| Bolivia | 15 | 0.9506272196776989 | 0.8504427629842091 | 0.8356761581483696 | 0.8429947994633135 | 0.9430830776750857 | 0.7286007003877626 | 0.8358418890314241 |

## STEP 5I vs STEP 5M vs STEP 5P

Positive differences mean STEP 5P TerraMind-L UPerNet big training is higher than the comparison run.

| Split | 5I mIoU | 5M mIoU | 5P mIoU | 5P-5I mIoU | 5P-5M mIoU | 5I IoU water | 5M IoU water | 5P IoU water | 5P-5I IoU water | 5P-5M IoU water | 5I F1 water | 5M F1 water | 5P F1 water | 5P-5I F1 | 5P-5M F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| valid | 0.8433460351368106 | 0.8076995876730401 | 0.846117912254208 | 0.0027718771173973478 | 0.038418324581167895 | 0.7229381921443395 | 0.6603594877459502 | 0.7285559552196581 | 0.0056177630753185825 | 0.06819646747370789 | 0.8391922536055492 | 0.7954415807174778 | 0.8429648493815476 | 0.003772595775998422 | 0.04752326866406975 |
| test | 0.8642341371590809 | 0.8308985329759455 | 0.8438073107868187 | -0.02042682637226223 | 0.012908777810873184 | 0.7647258676993773 | 0.7072686509782726 | 0.7295199926049281 | -0.03520587509444928 | 0.02225134162665543 | 0.8666795015549114 | 0.8285382040758548 | 0.8436097827422702 | -0.023069718812641193 | 0.015071578666415397 |
| bolivia | 0.8614010496940999 | 0.8277095929663913 | 0.8358418890314241 | -0.02555916066267583 | 0.00813229606503274 | 0.76837354682597 | 0.7186329402348496 | 0.7286007003877626 | -0.03977284643820744 | 0.009967760152913008 | 0.8690172370030228 | 0.8362843786022733 | 0.8429947994633135 | -0.026022437539709342 | 0.006710420861040123 |

- Comparison JSON: `E:/flood_research/experiments/terramind_baseline/runs/step5p_terramind_l_upernet_big_classical_training/metrics/step5i_step5m_step5p_comparison.json`
- Comparison CSV: `E:/flood_research/experiments/terramind_baseline/runs/step5p_terramind_l_upernet_big_classical_training/metrics/step5i_step5m_step5p_comparison.csv`

## Baseline Decision

- Decision case: C
- Selected physics baseline: `TerraMind base pretrained + UNetDecoder`
- Recommendation: Results are mixed; selection is based on validation mIoU first, then water IoU, Bolivia generalization, and qualitative consistency.
- Rationale: STEP 5P minus STEP 5I mIoU deltas: {'valid': 0.0027718771173973478, 'test': -0.02042682637226223, 'bolivia': -0.02555916066267583}
- Can physics-informed loss start next: Yes, but only after human validation of STEP 5P.

## Runtime And Artifacts

- Machine OS: Windows-11-10.0.26200-SP0
- CPU: AMD64 Family 25 Model 33 Stepping 2, AuthenticAMD
- RAM GB: None
- Device: NVIDIA GeForce RTX 4070
- Total elapsed seconds: 3904.322
- GPU peak allocated/reserved MB: 8227.81 / 9408.0
- Training step metrics: `E:/flood_research/experiments/terramind_baseline/runs/step5p_terramind_l_upernet_big_classical_training/metrics/training_step_metrics.csv`
- Training epoch metrics: `E:/flood_research/experiments/terramind_baseline/runs/step5p_terramind_l_upernet_big_classical_training/metrics/training_epoch_metrics.csv`
- Validation predictions: `E:/flood_research/experiments/terramind_baseline/runs/step5p_terramind_l_upernet_big_classical_training/predictions/valid`
- Test predictions: `E:/flood_research/experiments/terramind_baseline/runs/step5p_terramind_l_upernet_big_classical_training/predictions/test`
- Bolivia predictions: `E:/flood_research/experiments/terramind_baseline/runs/step5p_terramind_l_upernet_big_classical_training/predictions/bolivia`
- Summary JSON: `E:/flood_research/experiments/terramind_baseline/runs/step5p_terramind_l_upernet_big_classical_training/metrics/step5p_summary.json`
- Log: `E:/flood_research/experiments/terramind_baseline/runs/step5p_terramind_l_upernet_big_classical_training/logs/STEP_5P_terramind_l_upernet_big_classical_training.log`

## Checkpoints

- best: `E:/flood_research/experiments/terramind_baseline/runs/step5p_terramind_l_upernet_big_classical_training/checkpoints/best_checkpoint.pt` (2523.546 MB)
- last: `E:/flood_research/experiments/terramind_baseline/runs/step5p_terramind_l_upernet_big_classical_training/checkpoints/last_checkpoint.pt` (2523.546 MB)

## Qualitative Panels

- `E:/flood_research/experiments/terramind_baseline/runs/step5p_terramind_l_upernet_big_classical_training/reports/figures/valid/valid_Ghana_895194_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5p_terramind_l_upernet_big_classical_training/reports/figures/valid/valid_Ghana_868803_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5p_terramind_l_upernet_big_classical_training/reports/figures/valid/valid_Ghana_142312_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5p_terramind_l_upernet_big_classical_training/reports/figures/valid/valid_Ghana_132163_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5p_terramind_l_upernet_big_classical_training/reports/figures/valid/valid_Ghana_495107_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5p_terramind_l_upernet_big_classical_training/reports/figures/test/test_Ghana_313799_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5p_terramind_l_upernet_big_classical_training/reports/figures/test/test_Ghana_1078550_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5p_terramind_l_upernet_big_classical_training/reports/figures/test/test_Ghana_97059_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5p_terramind_l_upernet_big_classical_training/reports/figures/test/test_Ghana_359826_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5p_terramind_l_upernet_big_classical_training/reports/figures/test/test_Ghana_319168_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5p_terramind_l_upernet_big_classical_training/reports/figures/bolivia/bolivia_Bolivia_103757_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5p_terramind_l_upernet_big_classical_training/reports/figures/bolivia/bolivia_Bolivia_129334_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5p_terramind_l_upernet_big_classical_training/reports/figures/bolivia/bolivia_Bolivia_195474_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5p_terramind_l_upernet_big_classical_training/reports/figures/bolivia/bolivia_Bolivia_23014_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5p_terramind_l_upernet_big_classical_training/reports/figures/bolivia/bolivia_Bolivia_233925_panel.png`

## Guardrails

- Physics-informed loss started: False
- DARN started: False
- STURM-Flood training started: False
- Raw data modified: False
- Architecture experiments beyond STEP 5P started: False

## Problems And Warnings

- BatchNorm modules are kept in eval mode and their affine parameters are frozen during batch-size-1 UPerNet smoke/training, because the default PSP pool scale 1 creates a 1x1 feature map that BatchNorm cannot train on with a single sample.
- Rasterio may emit CPLE_IllegalArg BLOCKXSIZE warnings while writing prediction GeoTIFFs; metrics are trusted only when missing/error counts remain zero.

## Result

- STEP 5P passed: True
- Recommended next step: wait for human validation.

Human validation required before starting STEP 6A — implement the first physics-informed topographic loss on the selected final baseline.
