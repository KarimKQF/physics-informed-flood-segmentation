# STEP 5O - SOTA-Candidate Long Classical Training

## Status

Result: PASS

STEP 5O gave the SOTA-candidate model a fairer classical-loss budget before deciding which validated baseline should receive the first physics-informed topographic loss. STEP 5M was only a 5-epoch controlled run, so it was not sufficient to conclude definitively that TerraMind-L + UPerNet was inferior.

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
- Download used in STEP 5O: False

## Configuration

- Config path: `E:/flood_research/experiments/terramind_baseline/runs/step5o_terramind_l_upernet_long_classical_training/configs/terramind_l_upernet_long_classical_train.yaml`
- Model/backbone: `terramind_v1_large`
- Decoder: `UperNetDecoder`
- Neck path: `SelectIndices([2, 5, 8, 11]) -> ReshapeTokensToImage(remove_cls_token=false) -> LearnedInterpolateToPyramidal`
- Decoder config: `channels=256`, `pool_scales=[1, 2, 3, 6]`, `align_corners=true`, `scale_modules=false`
- Ignore index: -1
- BatchNorm policy: BatchNorm modules are kept in eval mode and their affine parameters are frozen during batch-size-1 UPerNet smoke/training, because the default PSP pool scale 1 creates a 1x1 feature map that BatchNorm cannot train on with a single sample.
- Scheduler: ReduceLROnPlateau(mode=max, factor=0.5, patience=3)
- Early stopping: monitor validation mIoU, patience 7, min epochs 10

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
- Elapsed seconds: 45.509

## Hyperparameters

- Max epochs: 30
- Epochs completed: 30
- Best epoch: 23
- Early stopped: True
- Early stop reason: validation_miou did not improve for 7 epochs after epoch 23
- Batch size: 1
- Precision requested/used: 32 / 32
- Optimizer: AdamW
- Learning rate: 0.0001
- Weight decay: 0.0001
- Seed: 42
- Trainable parameters: 321000194
- Total parameters: 321007362
- Training duration seconds: 4040.622

## Training Curve

| Epoch | Train loss | Validation loss | Validation mIoU | Validation IoU water | Validation F1 water | LR | GPU alloc MB | GPU reserved MB | GPU peak alloc MB | GPU peak reserved MB | Epoch sec | Precision | Improved |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 14.622741 | 0.151350 | 0.720832 | 0.501405 | 0.667914 | 0.00010000 | 4459.81 | 8536.0 | 7867.5 | 8536.0 | 162.018 | 32 | True |
| 2 | 0.163670 | 0.167597 | 0.733873 | 0.524931 | 0.688466 | 0.00010000 | 4459.81 | 8536.0 | 7867.5 | 8536.0 | 128.647 | 32 | True |
| 3 | 0.169996 | 0.143441 | 0.781829 | 0.624797 | 0.769077 | 0.00010000 | 4459.81 | 8536.0 | 7867.5 | 8536.0 | 127.059 | 32 | True |
| 4 | 0.181611 | 0.122197 | 0.783984 | 0.616292 | 0.762600 | 0.00010000 | 4460.06 | 8536.0 | 7867.5 | 8536.0 | 128.893 | 32 | True |
| 5 | 0.140266 | 0.112859 | 0.821221 | 0.685686 | 0.813540 | 0.00010000 | 4459.81 | 8536.0 | 7867.5 | 8536.0 | 129.057 | 32 | True |
| 6 | 0.158840 | 0.157985 | 0.800529 | 0.652257 | 0.789535 | 0.00010000 | 4459.81 | 8536.0 | 7867.5 | 8536.0 | 128.855 | 32 | False |
| 7 | 0.226295 | 0.118889 | 0.811994 | 0.667121 | 0.800327 | 0.00010000 | 4459.81 | 8536.0 | 7867.5 | 8536.0 | 129.161 | 32 | False |
| 8 | 0.145579 | 0.118291 | 0.812695 | 0.669124 | 0.801766 | 0.00010000 | 4460.06 | 8536.0 | 7867.5 | 8536.0 | 128.709 | 32 | False |
| 9 | 0.149019 | 0.124705 | 0.804415 | 0.653619 | 0.790532 | 0.00010000 | 4459.81 | 8536.0 | 7867.5 | 8536.0 | 129.318 | 32 | False |
| 10 | 0.130947 | 0.110093 | 0.810844 | 0.664885 | 0.798716 | 0.00005000 | 4459.81 | 8536.0 | 7867.5 | 8536.0 | 129.1 | 32 | False |
| 11 | 0.126798 | 0.103541 | 0.827738 | 0.695983 | 0.820743 | 0.00005000 | 4459.81 | 8536.0 | 7867.5 | 8536.0 | 131.065 | 32 | True |
| 12 | 0.121760 | 0.114619 | 0.830634 | 0.701631 | 0.824657 | 0.00005000 | 4460.06 | 8536.0 | 7867.5 | 8536.0 | 130.035 | 32 | True |
| 13 | 0.137919 | 0.106969 | 0.822190 | 0.686332 | 0.813994 | 0.00005000 | 4459.81 | 8536.0 | 7867.5 | 8536.0 | 128.989 | 32 | False |
| 14 | 0.126949 | 0.101382 | 0.821352 | 0.683671 | 0.812120 | 0.00005000 | 4459.81 | 8536.0 | 7867.5 | 8536.0 | 129.016 | 32 | False |
| 15 | 0.128154 | 0.104342 | 0.839127 | 0.716239 | 0.834661 | 0.00005000 | 4459.81 | 8536.0 | 7867.5 | 8536.0 | 128.891 | 32 | True |
| 16 | 0.126181 | 0.242581 | 0.465275 | 0.037178 | 0.071691 | 0.00005000 | 4460.06 | 8536.0 | 7867.5 | 8536.0 | 129.515 | 32 | False |
| 17 | 0.143136 | 0.190759 | 0.786770 | 0.620530 | 0.765836 | 0.00005000 | 4459.81 | 8536.0 | 7867.5 | 8536.0 | 128.841 | 32 | False |
| 18 | 0.129231 | 0.101130 | 0.816523 | 0.674894 | 0.805895 | 0.00005000 | 4459.81 | 8536.0 | 7867.5 | 8536.0 | 128.51 | 32 | False |
| 19 | 0.117452 | 0.096735 | 0.831675 | 0.701667 | 0.824682 | 0.00005000 | 4459.81 | 8536.0 | 7867.5 | 8536.0 | 129.51 | 32 | False |
| 20 | 0.104927 | 0.164608 | 0.843449 | 0.727914 | 0.842535 | 0.00002500 | 4460.06 | 8536.0 | 7867.5 | 8536.0 | 130.611 | 32 | True |
| 21 | 0.111404 | 0.111162 | 0.830190 | 0.699079 | 0.822892 | 0.00002500 | 4459.81 | 8536.0 | 7867.5 | 8536.0 | 129.044 | 32 | False |
| 22 | 0.110333 | 0.106978 | 0.832148 | 0.703373 | 0.825859 | 0.00002500 | 4459.81 | 8536.0 | 7867.5 | 8536.0 | 128.852 | 32 | False |
| 23 | 0.111000 | 0.088858 | 0.850327 | 0.735229 | 0.847415 | 0.00002500 | 4459.81 | 8536.0 | 7867.5 | 8536.0 | 129.312 | 32 | True |
| 24 | 0.107927 | 0.092735 | 0.831838 | 0.701927 | 0.824861 | 0.00002500 | 4460.06 | 8536.0 | 7867.5 | 8536.0 | 129.036 | 32 | False |
| 25 | 0.105919 | 0.112719 | 0.799276 | 0.642264 | 0.782169 | 0.00002500 | 4459.81 | 8536.0 | 7867.5 | 8536.0 | 129.665 | 32 | False |
| 26 | 0.106849 | 0.087137 | 0.848234 | 0.731272 | 0.844780 | 0.00002500 | 4459.81 | 8536.0 | 7867.5 | 8536.0 | 129.182 | 32 | False |
| 27 | 0.109024 | 0.090010 | 0.836876 | 0.711084 | 0.831150 | 0.00002500 | 4459.81 | 8536.0 | 7867.5 | 8536.0 | 128.49 | 32 | False |
| 28 | 0.099844 | 0.087437 | 0.838299 | 0.713145 | 0.832556 | 0.00001250 | 4460.06 | 8536.0 | 7867.5 | 8536.0 | 129.014 | 32 | False |
| 29 | 0.099773 | 0.086118 | 0.844885 | 0.725505 | 0.840919 | 0.00001250 | 4459.81 | 8536.0 | 7867.5 | 8536.0 | 128.893 | 32 | False |
| 30 | 0.097298 | 0.094162 | 0.821165 | 0.681810 | 0.810805 | 0.00001250 | 4459.81 | 8536.0 | 7867.5 | 8536.0 | 128.839 | 32 | False |

## Evaluation Metrics

| Split | Evaluated tiles | Accuracy | Precision water | Recall water | F1 water | IoU background | IoU water | mIoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 86 | 0.9684538716341315 | 0.9078688740578763 | 0.7945088610366888 | 0.847414589874575 | 0.9654251504353206 | 0.7352293222090663 | 0.8503272363221934 |
| Test | 89 | 0.9651802787365454 | 0.9024451522938842 | 0.8090570090577105 | 0.8532032178994475 | 0.9612602619833964 | 0.7439881513590065 | 0.8526242066712015 |
| Bolivia | 15 | 0.9544590568080576 | 0.8732242953004684 | 0.8339481691230786 | 0.8531344283900597 | 0.9475170003238922 | 0.7438835461704998 | 0.845700273247196 |

## STEP 5I vs STEP 5M vs STEP 5O

Positive differences mean STEP 5O TerraMind-L UPerNet long training is higher than the comparison run.

| Split | 5I mIoU | 5M mIoU | 5O mIoU | 5O-5I mIoU | 5O-5M mIoU | 5I IoU water | 5M IoU water | 5O IoU water | 5O-5I IoU water | 5O-5M IoU water | 5I F1 water | 5M F1 water | 5O F1 water | 5O-5I F1 | 5O-5M F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| valid | 0.8433460351368106 | 0.8076995876730401 | 0.8503272363221934 | 0.006981201185382813 | 0.04262764864915336 | 0.7229381921443395 | 0.6603594877459502 | 0.7352293222090663 | 0.012291130064726818 | 0.07486983446311613 | 0.8391922536055492 | 0.7954415807174778 | 0.847414589874575 | 0.008222336269025798 | 0.05197300915709713 |
| test | 0.8642341371590809 | 0.8308985329759455 | 0.8526242066712015 | -0.011609930487879394 | 0.02172567369525602 | 0.7647258676993773 | 0.7072686509782726 | 0.7439881513590065 | -0.020737716340370804 | 0.036719500380733905 | 0.8666795015549114 | 0.8285382040758548 | 0.8532032178994475 | -0.013476283655463828 | 0.024665013823592763 |
| bolivia | 0.8614010496940999 | 0.8277095929663913 | 0.845700273247196 | -0.01570077644690393 | 0.01799068028080464 | 0.76837354682597 | 0.7186329402348496 | 0.7438835461704998 | -0.024490000655470245 | 0.025250605935650206 | 0.8690172370030228 | 0.8362843786022733 | 0.8531344283900597 | -0.015882808612963095 | 0.01685004978778637 |

- Comparison JSON: `E:/flood_research/experiments/terramind_baseline/runs/step5o_terramind_l_upernet_long_classical_training/metrics/step5i_step5m_step5o_comparison.json`
- Comparison CSV: `E:/flood_research/experiments/terramind_baseline/runs/step5o_terramind_l_upernet_long_classical_training/metrics/step5i_step5m_step5o_comparison.csv`

## Baseline Decision

- Decision case: C
- Selected physics baseline: `TerraMind base pretrained + UNetDecoder`
- Recommendation: Results are mixed; selection is based on validation mIoU first, then water IoU, Bolivia generalization, and qualitative consistency.
- Rationale: STEP 5O minus STEP 5I mIoU deltas: {'valid': 0.006981201185382813, 'test': -0.011609930487879394, 'bolivia': -0.01570077644690393}
- Can physics-informed loss start next: Yes, but only after human validation of STEP 5O.

## Runtime And Artifacts

- Machine OS: Windows-11-10.0.26200-SP0
- CPU: AMD64 Family 25 Model 33 Stepping 2, AuthenticAMD
- RAM GB: None
- Device: NVIDIA GeForce RTX 4070
- Total elapsed seconds: 4201.661
- GPU peak allocated/reserved MB: 8227.81 / 9408.0
- Training step metrics: `E:/flood_research/experiments/terramind_baseline/runs/step5o_terramind_l_upernet_long_classical_training/metrics/training_step_metrics.csv`
- Training epoch metrics: `E:/flood_research/experiments/terramind_baseline/runs/step5o_terramind_l_upernet_long_classical_training/metrics/training_epoch_metrics.csv`
- Validation predictions: `E:/flood_research/experiments/terramind_baseline/runs/step5o_terramind_l_upernet_long_classical_training/predictions/valid`
- Test predictions: `E:/flood_research/experiments/terramind_baseline/runs/step5o_terramind_l_upernet_long_classical_training/predictions/test`
- Bolivia predictions: `E:/flood_research/experiments/terramind_baseline/runs/step5o_terramind_l_upernet_long_classical_training/predictions/bolivia`
- Summary JSON: `E:/flood_research/experiments/terramind_baseline/runs/step5o_terramind_l_upernet_long_classical_training/metrics/step5o_summary.json`
- Log: `E:/flood_research/experiments/terramind_baseline/runs/step5o_terramind_l_upernet_long_classical_training/logs/STEP_5O_sota_candidate_long_classical_training.log`

## Checkpoints

- best: `E:/flood_research/experiments/terramind_baseline/runs/step5o_terramind_l_upernet_long_classical_training/checkpoints/best_checkpoint.pt` (2523.546 MB)
- last: `E:/flood_research/experiments/terramind_baseline/runs/step5o_terramind_l_upernet_long_classical_training/checkpoints/last_checkpoint.pt` (2523.546 MB)

## Qualitative Panels

- `E:/flood_research/experiments/terramind_baseline/runs/step5o_terramind_l_upernet_long_classical_training/reports/figures/valid/valid_Ghana_895194_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5o_terramind_l_upernet_long_classical_training/reports/figures/valid/valid_Ghana_868803_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5o_terramind_l_upernet_long_classical_training/reports/figures/valid/valid_Ghana_142312_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5o_terramind_l_upernet_long_classical_training/reports/figures/valid/valid_Ghana_132163_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5o_terramind_l_upernet_long_classical_training/reports/figures/valid/valid_Ghana_495107_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5o_terramind_l_upernet_long_classical_training/reports/figures/test/test_Ghana_313799_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5o_terramind_l_upernet_long_classical_training/reports/figures/test/test_Ghana_1078550_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5o_terramind_l_upernet_long_classical_training/reports/figures/test/test_Ghana_97059_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5o_terramind_l_upernet_long_classical_training/reports/figures/test/test_Ghana_359826_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5o_terramind_l_upernet_long_classical_training/reports/figures/test/test_Ghana_319168_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5o_terramind_l_upernet_long_classical_training/reports/figures/bolivia/bolivia_Bolivia_103757_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5o_terramind_l_upernet_long_classical_training/reports/figures/bolivia/bolivia_Bolivia_129334_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5o_terramind_l_upernet_long_classical_training/reports/figures/bolivia/bolivia_Bolivia_195474_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5o_terramind_l_upernet_long_classical_training/reports/figures/bolivia/bolivia_Bolivia_23014_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5o_terramind_l_upernet_long_classical_training/reports/figures/bolivia/bolivia_Bolivia_233925_panel.png`

## Guardrails

- Physics-informed loss started: False
- DARN started: False
- STURM-Flood training started: False
- Raw data modified: False
- Architecture experiments beyond STEP 5O started: False

## Problems And Warnings

- BatchNorm modules are kept in eval mode and their affine parameters are frozen during batch-size-1 UPerNet smoke/training, because the default PSP pool scale 1 creates a 1x1 feature map that BatchNorm cannot train on with a single sample.
- Rasterio may emit CPLE_IllegalArg BLOCKXSIZE warnings while writing prediction GeoTIFFs; metrics are trusted only when missing/error counts remain zero.

## Result

- STEP 5O passed: True
- Recommended next step: wait for human validation.

Human validation required before starting STEP 6A — implement the first physics-informed topographic loss on the selected final baseline.
