# STEP 5J - Controlled TerraMind UPerNet Adaptation

## Status

Result: PASS

STEP 5J validated only the controlled TerraMind UPerNet decoder path on Sen1Floods11. TerraMind-L, base UPerNet training, DARN, STURM-Flood training, physics loss, raw-data modification, and any full-model training were not started.

## UPerNet Support Inspection

- Decoder class: `UperNetDecoder`
- Import path: `terratorch.models.decoders.upernet_decoder.UperNetDecoder`
- Registry name: `UperNetDecoder`
- Source file: `E:/flood_research/venvs/terramind-gpu/Lib/site-packages/terratorch/models/decoders/upernet_decoder.py`
- Constructor signature: `(self, embed_dim: list[int], pool_scales: tuple[int] = (1, 2, 3, 6), channels: int = 256, align_corners: bool = True, scale_modules: bool = False)`
- Compatibility: EncoderDecoderFactory passes the post-neck channel list as UperNetDecoder embed_dim. The TerraMind ViT token outputs are selected, reshaped to images, then converted into a four-level pyramid before UPerNet PSP/FPN fusion.
- BatchNorm policy: BatchNorm modules are kept in eval mode and their affine parameters are frozen during batch-size-1 UPerNet smoke/training, because the default PSP pool scale 1 creates a 1x1 feature map that BatchNorm cannot train on with a single sample.

Required syntax used in the generated configs:

```yaml
decoder: UperNetDecoder
decoder_channels: 256
decoder_pool_scales: [1, 2, 3, 6]
decoder_align_corners: true
decoder_scale_modules: false
necks:
  - name: SelectIndices
    indices: [2, 5, 8, 11]
  - name: ReshapeTokensToImage
    remove_cls_token: false
  - name: LearnedInterpolateToPyramidal
```

## Tiny UPerNet Smoke

- Config: `E:/flood_research/experiments/terramind_baseline/runs/step5j_upernet_adaptation/configs/terramind_v1_tiny_upernet_smoke.yaml`
- Model/backbone: `terramind_v1_tiny`
- Decoder: `UperNetDecoder`
- Backbone checkpoint: `E:/flood_research/experiments/terramind_baseline/checkpoints/pretrained/TerraMind_v1_tiny.pt`
- Passed: True
- Input shapes: `{'S2L1C': [1, 13, 512, 512], 'S1GRD': [1, 2, 512, 512]}`
- Mask shape: `[1, 512, 512]`
- Output shape: `[1, 2, 512, 512]`
- Classes: 2
- Loss: 22.076749801635742
- Valid pixels: 214619
- BatchNorm modules kept eval/frozen: 13
- GPU allocated/reserved MB: 81.51 / 348.0
- GPU peak allocated/reserved MB: 267.01 / 348.0
- Elapsed seconds: 0.991

## Tiny UPerNet Training

- Run directory: `E:/flood_research/experiments/terramind_baseline/runs/step5j_upernet_adaptation`
- Config path: `E:/flood_research/experiments/terramind_baseline/runs/step5j_upernet_adaptation/configs/terramind_v1_tiny_upernet_smoke.yaml`
- Epochs: 5
- Batch size: 1
- Precision requested/used: 32 / 32
- Optimizer: AdamW
- Learning rate: 0.0001
- Weight decay: 0.0001
- Seed: 42
- BatchNorm eval/frozen modules: 13
- Trainable parameters: 13451314
- Total parameters: 13457650

| Epoch | Train loss | Validation loss | Validation mIoU | Validation IoU water | Precision |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 7.236227 | 0.14598632010739704 | 0.769763 | 0.601453 | 32 |
| 2 | 0.135926 | 0.6360183155112908 | 0.439571 | 0.234430 | 32 |
| 3 | 0.193165 | 0.13190591319074255 | 0.772953 | 0.596386 | 32 |
| 4 | 0.152825 | 0.13901985045744109 | 0.779985 | 0.609268 | 32 |
| 5 | 0.139521 | 0.2813463268328993 | 0.806423 | 0.658938 | 32 |

## Evaluation Metrics

| Split | Evaluated tiles | Accuracy | Precision water | Recall water | F1 water | IoU background | IoU water | mIoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 86 | 0.9576760463618009 | 0.8552534609235932 | 0.741647877976676 | 0.7944096469516511 | 0.9539069857543865 | 0.6589382910563089 | 0.8064226384053477 |
| Test | 89 | 0.956362188189157 | 0.8447984949756854 | 0.7976276070193652 | 0.8205356711544124 | 0.9515265199662988 | 0.695685025046514 | 0.8236057725064063 |
| Bolivia | 15 | 0.9372445572674667 | 0.7640546143013853 | 0.8743448591227267 | 0.8154875962054036 | 0.9271404032294958 | 0.688458469149822 | 0.8077994361896589 |

## Tiny UPerNet vs STEP 5G Tiny UNetDecoder

Positive differences mean STEP 5J tiny UPerNet is higher than STEP 5G tiny UNetDecoder.

| Split | 5G mIoU | 5J mIoU | Delta mIoU | 5G IoU water | 5J IoU water | Delta IoU water | 5G F1 water | 5J F1 water | Delta F1 water |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| valid | 0.8270253444539271 | 0.8064226384053477 | -0.02060270604857939 | 0.6943352476607539 | 0.6589382910563089 | -0.035396956604445085 | 0.8195960611919895 | 0.7944096469516511 | -0.025186414240338406 |
| test | 0.8526102568963974 | 0.8236057725064063 | -0.029004484389991037 | 0.7448119052825767 | 0.695685025046514 | -0.04912688023606271 | 0.8537446392102105 | 0.8205356711544124 | -0.033208968055798094 |
| bolivia | 0.8390577176497471 | 0.8077994361896589 | -0.031258281460088244 | 0.7350870143233763 | 0.688458469149822 | -0.0466285451735543 | 0.8473200574439602 | 0.8154875962054036 | -0.03183246123855665 |

- Comparison JSON: `E:/flood_research/experiments/terramind_baseline/runs/step5j_upernet_adaptation/metrics/tiny_upernet_vs_step5g_tiny_unetdecoder_comparison.json`

## Optional Base UPerNet Smoke

- Config: `E:/flood_research/experiments/terramind_baseline/runs/step5j_upernet_adaptation/configs/terramind_v1_base_upernet_smoke.yaml`
- Model/backbone: `terramind_v1_base`
- Decoder: `UperNetDecoder`
- Backbone checkpoint: `E:/flood_research/experiments/terramind_baseline/checkpoints/pretrained/TerraMind_v1_base.pt`
- Passed: True
- Input shapes: `{'S2L1C': [1, 13, 512, 512], 'S1GRD': [1, 2, 512, 512]}`
- Mask shape: `[1, 512, 512]`
- Output shape: `[1, 2, 512, 512]`
- Loss: 7.27852725982666
- BatchNorm modules kept eval/frozen: 13
- GPU peak allocated/reserved MB: 901.55 / 1060.0
- Elapsed seconds: 2.177
- Error: None
- Base UPerNet training started: False

## Runtime And Artifacts

- Machine OS: Windows-11-10.0.26200-SP0
- CPU: AMD64 Family 25 Model 33 Stepping 2, AuthenticAMD
- RAM GB: None
- Device: NVIDIA GeForce RTX 4070
- Total elapsed seconds: 335.676
- Training elapsed seconds: 278.459
- GPU peak allocated/reserved MB: 901.55 / 1060.0
- Training step metrics: `E:/flood_research/experiments/terramind_baseline/runs/step5j_upernet_adaptation/metrics/training_step_metrics.csv`
- Training epoch metrics: `E:/flood_research/experiments/terramind_baseline/runs/step5j_upernet_adaptation/metrics/training_epoch_metrics.csv`
- Validation predictions: `E:/flood_research/experiments/terramind_baseline/runs/step5j_upernet_adaptation/predictions/valid`
- Test predictions: `E:/flood_research/experiments/terramind_baseline/runs/step5j_upernet_adaptation/predictions/test`
- Bolivia predictions: `E:/flood_research/experiments/terramind_baseline/runs/step5j_upernet_adaptation/predictions/bolivia`
- Summary JSON: `E:/flood_research/experiments/terramind_baseline/runs/step5j_upernet_adaptation/metrics/step5j_summary.json`
- Log: `E:/flood_research/experiments/terramind_baseline/runs/step5j_upernet_adaptation/logs/STEP_5J_upernet_adaptation.log`

## Checkpoints

- best: `E:/flood_research/experiments/terramind_baseline/runs/step5j_upernet_adaptation/checkpoints/best_checkpoint.pt` (154.509 MB)
- last: `E:/flood_research/experiments/terramind_baseline/runs/step5j_upernet_adaptation/checkpoints/last_checkpoint.pt` (154.509 MB)

## Qualitative Panels

- `E:/flood_research/experiments/terramind_baseline/runs/step5j_upernet_adaptation/reports/figures/valid/valid_Ghana_895194_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5j_upernet_adaptation/reports/figures/valid/valid_Ghana_868803_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5j_upernet_adaptation/reports/figures/valid/valid_Ghana_142312_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5j_upernet_adaptation/reports/figures/valid/valid_Ghana_132163_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5j_upernet_adaptation/reports/figures/valid/valid_Ghana_495107_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5j_upernet_adaptation/reports/figures/test/test_Ghana_313799_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5j_upernet_adaptation/reports/figures/test/test_Ghana_1078550_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5j_upernet_adaptation/reports/figures/test/test_Ghana_97059_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5j_upernet_adaptation/reports/figures/test/test_Ghana_359826_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5j_upernet_adaptation/reports/figures/test/test_Ghana_319168_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5j_upernet_adaptation/reports/figures/bolivia/bolivia_Bolivia_103757_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5j_upernet_adaptation/reports/figures/bolivia/bolivia_Bolivia_129334_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5j_upernet_adaptation/reports/figures/bolivia/bolivia_Bolivia_195474_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5j_upernet_adaptation/reports/figures/bolivia/bolivia_Bolivia_23014_panel.png`
- `E:/flood_research/experiments/terramind_baseline/runs/step5j_upernet_adaptation/reports/figures/bolivia/bolivia_Bolivia_233925_panel.png`

## Guardrails

- TerraMind-L started: False
- DARN started: False
- STURM-Flood training started: False
- Physics loss started: False
- Raw data modified: False
- Base UPerNet training started: False

## Problems And Warnings

- BatchNorm modules are kept in eval mode and their affine parameters are frozen during batch-size-1 UPerNet smoke/training, because the default PSP pool scale 1 creates a 1x1 feature map that BatchNorm cannot train on with a single sample.
- Rasterio may emit CPLE_IllegalArg BLOCKXSIZE warnings while writing prediction GeoTIFFs; metrics are trusted only when missing/error counts remain zero.

## Gate

Recommended next step: wait for human validation before setting up STEP 5K.

Human validation required before starting STEP 5K — base UPerNet training or full-model memory test.
