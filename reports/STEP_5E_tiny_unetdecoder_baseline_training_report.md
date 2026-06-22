# STEP 5E - TerraMind Tiny UNetDecoder Baseline Training

## Status

STEP 5E is complete. This was the first controlled real TerraMind baseline training run only. UPerNet, DARN, STURM-Flood training, physics loss, TerraMind small/base, and raw-data modification were not started.

Result: PASS

## Configuration

- Config path: `E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/configs/terramind_v1_tiny_unetdecoder_train.yaml`
- Run directory: `E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline`
- Model: `terramind_v1_tiny`
- Decoder: `UNetDecoder`
- Pretrained status: non-pretrained baseline fallback
- Pretrained attempt error: `huggingface_hub.errors.LocalEntryNotFoundError: An error happened while trying to locate the file on the Hub and we cannot find the requested files in the local cache. Please check your connection and try again or make sure your Internet connection is on.`
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
- Precision requested/used: 16-mixed / 32
- Optimizer: AdamW
- Learning rate: 0.0001
- Weight decay: 0.0001
- Seed: 42
- Trainable parameters: 8433330
- Total parameters: 8433330

## Training Curve

| Epoch | Train loss | Validation loss | Validation mIoU | Validation IoU water | Precision |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0.564631 | 0.410765 | 0.783617 | 0.625840 | 16-mixed |
| 2 | 0.439023 | 0.330091 | 0.775630 | 0.606596 | 16-mixed |
| 3 | 0.356852 | 0.328513 | 0.746901 | 0.567265 | 16-mixed |
| 4 | 0.301707 | nan | 0.444872 | 0.000000 | 32 |
| 5 | 0.273055 | nan | 0.444872 | 0.000000 | 32 |

## Evaluation Metrics

| Split | Evaluated tiles | Accuracy | Precision water | Recall water | F1 water | IoU background | IoU water | mIoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 86 | 0.9466263770511796 | 0.733756540473992 | 0.8097144044637011 | 0.7698664391965319 | 0.9413943285102817 | 0.6258397167001033 | 0.7836170226051925 |
| Test | 89 | 0.9487587271797595 | 0.7684119279148122 | 0.8449554401794785 | 0.8048679446855852 | 0.9427037560263088 | 0.6734552396168814 | 0.808079497821595 |
| Bolivia | 15 | 0.9063391467022803 | 0.6476666603243463 | 0.8980112736993915 | 0.7525655245092802 | 0.8907829817894379 | 0.603290625115386 | 0.747036803452412 |

## Runtime And GPU

- Device: NVIDIA GeForce RTX 4070
- Training elapsed seconds: 255.497
- Total elapsed seconds: 371.114
- GPU peak allocated/reserved MB: 1375.25 / 1576.0

## Artifacts

- Training step metrics: `E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/metrics/training_step_metrics.csv`
- Training epoch metrics: `E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/metrics/training_epoch_metrics.csv`
- Validation predictions: `E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/predictions/valid`
- Test predictions: `E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/predictions/test`
- Bolivia predictions: `E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/predictions/bolivia`
- Summary JSON: `E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/metrics/step5e_summary.json`
- Log: `E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/logs/STEP_5E_tiny_unetdecoder_training.log`

## Checkpoints

- best: `E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/checkpoints/best_checkpoint.pt` (97.046 MB)
- last: `E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/checkpoints/last_checkpoint.pt` (97.046 MB)

## Qualitative Panels

- `E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/reports/figures/valid/valid_Ghana_895194_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/reports/figures/valid/valid_Ghana_868803_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/reports/figures/valid/valid_Ghana_142312_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/reports/figures/test/test_Ghana_313799_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/reports/figures/test/test_Ghana_1078550_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/reports/figures/test/test_Ghana_97059_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/reports/figures/bolivia/bolivia_Bolivia_103757_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/reports/figures/bolivia/bolivia_Bolivia_129334_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/reports/figures/bolivia/bolivia_Bolivia_195474_panel.png`

## Problems And Warnings

- Rasterio emitted CPLE_IllegalArg BLOCKXSIZE warnings while writing prediction GeoTIFFs; validation, test, and Bolivia evaluation completed with zero prediction errors and zero missing predictions.
- Epochs 4 and 5 had validation_loss=nan and validation IoU water=0.0 after the precision fallback; the best checkpoint selected for inference was epoch 1.
- 16-mixed precision produced a non-finite loss at global step 457; the step was retried in precision 32 and the run completed with precision_used=32.
- Official TerraMind tiny pretrained weights were not available from local Hugging Face cache; the run used a documented non-pretrained baseline fallback.
- Pretrained attempt error: huggingface_hub.errors.LocalEntryNotFoundError: An error happened while trying to locate the file on the Hub and we cannot find the requested files in the local cache. Please check your connection and try again or make sure your Internet connection is on.
- Repository pytest collection is still known to fail because legacy tests import missing module urban_runoff.data.

## Gate

Recommended next step: human validation of STEP 5E metrics, checkpoints, and qualitative panels before scaling model size or training duration.

Human validation required before starting STEP 5F — scale baseline to TerraMind small/base or longer training.

