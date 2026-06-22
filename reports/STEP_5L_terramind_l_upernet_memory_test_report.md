# STEP 5L - TerraMind-L UPerNet Memory Test

## Status

Result: DONE

STEP 5L performed only a memory feasibility test for TerraMind-L / TerraMind large pretrained + UPerNet on Sen1Floods11. No full training, epochs, DARN, STURM-Flood training, physics loss, or raw-data modification was started.

## Checkpoint

- Source: `ibm-esa-geospatial/TerraMind-1.0-large`
- API URL: `https://huggingface.co/api/models/ibm-esa-geospatial/TerraMind-1.0-large`
- Download URL: `https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-large/resolve/main/TerraMind_v1_large.pt`
- Checkpoint filename: `TerraMind_v1_large.pt`
- Metadata resolvable: True
- Expected size bytes: 3787890978
- Expected size GB: 3.528
- Local checkpoint path: `E:/flood_research/experiments/terramind_baseline/checkpoints/pretrained/TerraMind_v1_large.pt`
- Downloaded: True
- Local size bytes: 3787890978
- SHA256: `a1c6b567ce6862c7ac07181551add840edce52f652abccfe5f17d23544060f81`

## Configuration

- Config path: `E:/flood_research/experiments/terramind_baseline/runs/step5l_terramind_l_upernet_memory_test/configs/terramind_l_upernet_memory_test.yaml`
- Model/backbone: `terramind_v1_large`
- Decoder: `UperNetDecoder`
- Neck path: `SelectIndices([2, 5, 8, 11]) -> ReshapeTokensToImage(remove_cls_token=false) -> LearnedInterpolateToPyramidal`
- Decoder config: `channels=256`, `pool_scales=[1, 2, 3, 6]`, `align_corners=true`, `scale_modules=false`
- Batch size: 1
- Image size: 512x512
- Ignore index: -1
- BatchNorm policy: BatchNorm modules are kept in eval mode and their affine parameters are frozen during batch-size-1 UPerNet train-mode memory tests, because the default PSP pool scale 1 creates a 1x1 feature map that BatchNorm cannot train on with a single sample.

## Level A - Model Initialization

- ok: `True`
- gpu_memory_allocated_mb: `1228.95`
- gpu_memory_reserved_mb: `1336.0`
- gpu_peak_memory_allocated_mb: `1228.95`
- gpu_peak_memory_reserved_mb: `1336.0`
- elapsed_seconds: `22.312`

## Level B - Inference Forward

- ok: `True`
- input_shapes: `{'S2L1C': [1, 13, 512, 512], 'S1GRD': [1, 2, 512, 512]}`
- output_shape: `[1, 2, 512, 512]`
- loss: `475.4401550292969`
- gpu_memory_allocated_mb: `1257.08`
- gpu_memory_reserved_mb: `2066.0`
- gpu_peak_memory_allocated_mb: `1655.08`
- gpu_peak_memory_reserved_mb: `2066.0`
- elapsed_seconds: `1.481`

## Level C - Training Forward And Backward

- ok: `True`
- precision: `32`
- input_shapes: `{'S2L1C': [1, 13, 512, 512], 'S1GRD': [1, 2, 512, 512]}`
- output_shape: `[1, 2, 512, 512]`
- loss: `4.2405877113342285`
- backward_ok: `True`
- optimizer_step_attempted: `False`
- gpu_memory_allocated_mb: `3142.33`
- gpu_memory_reserved_mb: `7370.0`
- gpu_peak_memory_allocated_mb: `6550.46`
- gpu_peak_memory_reserved_mb: `7370.0`
- elapsed_seconds: `5.786`

## Level C Retry

- not run

## Conclusion

- TerraMind-L UPerNet init OK: True
- TerraMind-L UPerNet inference OK: True
- TerraMind-L UPerNet backward OK: True
- Precision used for passing backward: 32
- Full model training started: False
- Feasibility conclusion: feasible for full training locally at batch size 1 and precision 32 from a memory perspective
- Recommended next step: wait for human validation.

## Guardrails

- Full training started: False
- DARN started: False
- STURM-Flood training started: False
- Physics loss started: False
- Raw data modified: False

## Problems And Warnings

- BatchNorm modules are kept in eval mode and their affine parameters are frozen during batch-size-1 UPerNet train-mode memory tests, because the default PSP pool scale 1 creates a 1x1 feature map that BatchNorm cannot train on with a single sample.

Human validation required before starting STEP 5M — choose full SOTA training if feasible, or proceed with best feasible baseline plus physics-informed loss preparation.
