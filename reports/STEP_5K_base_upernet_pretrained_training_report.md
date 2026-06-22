# STEP 5K - TerraMind Base Pretrained UPerNet Training

## Status

Result: PASS

STEP 5K trained only the controlled TerraMind base pretrained + UPerNet baseline on Sen1Floods11. TerraMind-L, full-model memory test, DARN, STURM-Flood training, physics loss, raw-data modification, and additional architectures were not started.

## Configuration

- Config path: `E:/flood_research/experiments/terramind_baseline/runs/step5k_base_upernet_pretrained/configs/terramind_v1_base_upernet_pretrained_train.yaml`
- Model/backbone: `terramind_v1_base`
- Decoder: `UperNetDecoder`
- Pretrained checkpoint: `E:/flood_research/experiments/terramind_baseline/checkpoints/pretrained/TerraMind_v1_base.pt`
- Pretrained checkpoint SHA256: `83c3a0938067c83867a46e564443c2fa38383bf4f966d931b11cb025b847d7ec`
- UPerNet class: `terratorch.models.decoders.upernet_decoder.UperNetDecoder`
- UPerNet signature: `(self, embed_dim: list[int], pool_scales: tuple[int] = (1, 2, 3, 6), channels: int = 256, align_corners: bool = True, scale_modules: bool = False)`
- Neck path: `SelectIndices([2, 5, 8, 11]) -> ReshapeTokensToImage(remove_cls_token=false) -> LearnedInterpolateToPyramidal`
- Decoder config: `channels=256`, `pool_scales=[1, 2, 3, 6]`, `align_corners=true`, `scale_modules=false`
- BatchNorm policy: BatchNorm modules are kept in eval mode and their affine parameters are frozen during batch-size-1 UPerNet smoke/training, because the default PSP pool scale 1 creates a 1x1 feature map that BatchNorm cannot train on with a single sample.

## Dataset

- Train tiles: 251
- Validation tiles: 86
- Test tiles: 89
- Bolivia holdout tiles: 15
- Excluded fully invalid samples: `Ghana_234935, Ghana_26376, Ghana_277, Ghana_5079, Ghana_83483`
- Ignore index: -1
- Raw data modified: False

## Smoke Forward

- Passed: True
- Input shapes: `{'S2L1C': [1, 13, 512, 512], 'S1GRD': [1, 2, 512, 512]}`
- Mask shape: `[1, 512, 512]`
- Output shape: `[1, 2, 512, 512]`
- Classes: 2
- Loss: 1.6675368547439575
- Valid pixels: 236164
- BatchNorm modules kept eval/frozen: 13
- GPU allocated/reserved MB: 412.12 / 784.0
- GPU peak allocated/reserved MB: 628.87 / 784.0
- Elapsed seconds: 2.684

## Hyperparameters

- Epochs: 5
- Batch size: 1
- Precision requested/used: 32 / 32
- Optimizer: AdamW
- Learning rate: 0.0001
- Weight decay: 0.0001
- Seed: 42
- Trainable parameters: 99948226
- Total parameters: 99955138
- Training duration seconds: 446.879

## Training Curve

| Epoch | Train loss | Validation loss | Validation mIoU | Validation IoU water | Precision |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 4.150405 | 0.15355087912113324 | 0.754669 | 0.562813 | 32 |
| 2 | 0.176374 | 0.1227107753513693 | 0.805948 | 0.656033 | 32 |
| 3 | 0.128665 | 0.43630154169892454 | 0.684932 | 0.489333 | 32 |
| 4 | 0.145659 | 0.11437617340512338 | 0.788857 | 0.624863 | 32 |
| 5 | 0.158271 | 0.12104736049299325 | 0.779303 | 0.607240 | 32 |

## Evaluation Metrics

| Split | Evaluated tiles | Accuracy | Precision water | Recall water | F1 water | IoU background | IoU water | mIoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 86 | 0.9592900125525229 | 0.9055452594195278 | 0.7042221482343846 | 0.7922946195578394 | 0.9558630252523725 | 0.656033029568658 | 0.8059480274105153 |
| Test | 89 | 0.9582261213146892 | 0.8973621738756397 | 0.7520078126309135 | 0.818280156707896 | 0.9538892486560557 | 0.6924485201401742 | 0.823168884398115 |
| Bolivia | 15 | 0.949023211050922 | 0.8462824269566115 | 0.8292214815857047 | 0.8376650920873503 | 0.9413032515441864 | 0.7206744685932649 | 0.8309888600687256 |

## STEP 5I Base UNetDecoder vs STEP 5K Base UPerNet

Positive differences mean STEP 5K base UPerNet is higher than STEP 5I base UNetDecoder.

| Split | 5I mIoU | 5K mIoU | Delta mIoU | 5I IoU water | 5K IoU water | Delta IoU water | 5I F1 water | 5K F1 water | Delta F1 water |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| valid | 0.8433460351368106 | 0.8059480274105153 | -0.03739800772629531 | 0.7229381921443395 | 0.656033029568658 | -0.06690516257568146 | 0.8391922536055492 | 0.7922946195578394 | -0.046897634047709746 |
| test | 0.8642341371590809 | 0.823168884398115 | -0.04106525276096595 | 0.7647258676993773 | 0.6924485201401742 | -0.07227734755920312 | 0.8666795015549114 | 0.818280156707896 | -0.04839934484701536 |
| bolivia | 0.8614010496940999 | 0.8309888600687256 | -0.030412189625374286 | 0.76837354682597 | 0.7206744685932649 | -0.04769907823270514 | 0.8690172370030228 | 0.8376650920873503 | -0.03135214491567251 |

- Comparison JSON: `E:/flood_research/experiments/terramind_baseline/runs/step5k_base_upernet_pretrained/metrics/step5i_vs_step5k_comparison.json`

## Runtime And Artifacts

- Machine OS: Windows-11-10.0.26200-SP0
- CPU: AMD64 Family 25 Model 33 Stepping 2, AuthenticAMD
- RAM GB: None
- Device: NVIDIA GeForce RTX 4070
- Total elapsed seconds: 530.343
- GPU peak allocated/reserved MB: 3110.58 / 3386.0
- Training step metrics: `E:/flood_research/experiments/terramind_baseline/runs/step5k_base_upernet_pretrained/metrics/training_step_metrics.csv`
- Training epoch metrics: `E:/flood_research/experiments/terramind_baseline/runs/step5k_base_upernet_pretrained/metrics/training_epoch_metrics.csv`
- Validation predictions: `E:/flood_research/experiments/terramind_baseline/runs/step5k_base_upernet_pretrained/predictions/valid`
- Test predictions: `E:/flood_research/experiments/terramind_baseline/runs/step5k_base_upernet_pretrained/predictions/test`
- Bolivia predictions: `E:/flood_research/experiments/terramind_baseline/runs/step5k_base_upernet_pretrained/predictions/bolivia`
- Summary JSON: `E:/flood_research/experiments/terramind_baseline/runs/step5k_base_upernet_pretrained/metrics/step5k_summary.json`
- Log: `E:/flood_research/experiments/terramind_baseline/runs/step5k_base_upernet_pretrained/logs/STEP_5K_base_upernet_pretrained_training.log`

## Checkpoints

- best: `E:/flood_research/experiments/terramind_baseline/runs/step5k_base_upernet_pretrained/checkpoints/best_checkpoint.pt` (1145.268 MB)
- last: `E:/flood_research/experiments/terramind_baseline/runs/step5k_base_upernet_pretrained/checkpoints/last_checkpoint.pt` (1145.268 MB)

## Qualitative Panels

- `E:/flood_research/experiments/terramind_baseline/runs/step5k_base_upernet_pretrained/reports/figures/valid/valid_Ghana_895194_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5k_base_upernet_pretrained/reports/figures/valid/valid_Ghana_868803_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5k_base_upernet_pretrained/reports/figures/valid/valid_Ghana_142312_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5k_base_upernet_pretrained/reports/figures/valid/valid_Ghana_132163_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5k_base_upernet_pretrained/reports/figures/valid/valid_Ghana_495107_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5k_base_upernet_pretrained/reports/figures/test/test_Ghana_313799_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5k_base_upernet_pretrained/reports/figures/test/test_Ghana_1078550_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5k_base_upernet_pretrained/reports/figures/test/test_Ghana_97059_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5k_base_upernet_pretrained/reports/figures/test/test_Ghana_359826_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5k_base_upernet_pretrained/reports/figures/test/test_Ghana_319168_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5k_base_upernet_pretrained/reports/figures/bolivia/bolivia_Bolivia_103757_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5k_base_upernet_pretrained/reports/figures/bolivia/bolivia_Bolivia_129334_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5k_base_upernet_pretrained/reports/figures/bolivia/bolivia_Bolivia_195474_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5k_base_upernet_pretrained/reports/figures/bolivia/bolivia_Bolivia_23014_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5k_base_upernet_pretrained/reports/figures/bolivia/bolivia_Bolivia_233925_panel.png`

## Guardrails

- TerraMind-L started: False
- Full model memory test started: False
- DARN started: False
- STURM-Flood training started: False
- Physics loss started: False
- Raw data modified: False

## Problems And Warnings

- BatchNorm modules are kept in eval mode and their affine parameters are frozen during batch-size-1 UPerNet smoke/training, because the default PSP pool scale 1 creates a 1x1 feature map that BatchNorm cannot train on with a single sample.
- Rasterio may emit CPLE_IllegalArg BLOCKXSIZE warnings while writing prediction GeoTIFFs; metrics are trusted only when missing/error counts remain zero.

## Result

- Base UPerNet passed: True
- Recommended next step: wait for human validation before choosing STEP 5L.

Human validation required before starting STEP 5L — choose between TerraMind-L memory test, longer base training, or physics-informed loss preparation.
