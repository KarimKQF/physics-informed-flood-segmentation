# STEP 5R - Official Benchmark / Segmentation-Loss Parity Audit

Generated at: 2026-06-21T14:56:37+00:00

## Status

Result: DONE

No model training, physics-loss training, TerraMind training, DARN, or STURM-Flood training was launched. Raw Sen1Floods11 data, raw DEM data, and official split files were not modified.

## Why This Audit Was Needed

STEP 6B4 completed the DEM/topographic alignment, but adding topographic physics loss before checking segmentation-baseline parity would make the next comparison hard to interpret. STEP 5R checks whether the public TerraMind/PANGAEA-style result gap is plausibly caused by loss choice or by broader protocol differences.

## Local Search Summary

- Inventory rows written: 12747
- Inventory CSV: `inventory/local_config_search_results.csv`
- Inventory JSON: `inventory/local_config_search_results.json`
- Search scopes: repository, repo configs, STEP 5E-5P run configs, TerraTorch package when import metadata was available, cached configs/checkpoint metadata, and Hugging Face cache text metadata.

Important local hits:

- `configs/terramind/official_terramind_v1_base_sen1floods11.yaml`
- `configs/terramind/official_terramind_v1_base_tim_lulc_sen1floods11.yaml`
- STEP 5I config: `loss: dice`, `decoder: UNetDecoder`, `backbone: terramind_v1_base`
- STEP 5O/5P configs: `loss: dice`, `decoder: UperNetDecoder`, `backbone: terramind_v1_large`

## Official Config Finding

- Official IBM TerraMind Sen1Floods11 configs found locally: yes.
- Official loss identified from those configs: dice.
- Exact reproducible PANGAEA/TerraMind Sen1Floods11 benchmark config for the public 0.905-0.9078 mIoU claim: not found locally.
- Public PANGAEA repository confirms the benchmark family and supported loss options, but the exact TerraMind/Sen1Floods11 benchmark hparams remain unknown from local files.

## Dataset Policy Snapshot

- Included samples: 441
- Split sizes: {"bolivia": 15, "test": 89, "train": 251, "valid": 86}
- No-water rows kept: 47
- Warning-review rows kept: 69
- Excluded fully invalid/error tile IDs: Ghana_234935, Ghana_26376, Ghana_277, Ghana_5079, Ghana_83483
- Invalid label pixels are ignored via `ignore_index: -1`.

## Protocol Comparison

Structured comparison files:

- `metadata/step5r_protocol_comparison.csv`
- `metadata/step5r_protocol_comparison.json`

Core observations:

- Our STEP 5I/5O/5P and the official IBM TerraMind Sen1Floods11 configs all use Dice loss.
- Official IBM configs use batch size 8, 16-mixed precision, 100 epochs, D4 augmentation, AdamW lr 2e-5, and UNetDecoder.
- Our STEP 5O/5P TerraMind-L + UPerNet runs use batch size 1, fp32, no train augmentation, AdamW lr 1e-4, mIoU scheduler/early stopping, and BatchNorm eval policy for UPerNet.
- Our TerraMind-L UPerNet configs use feature indices [2, 5, 8, 11]; the official config comment labels [5, 11, 17, 23] as the large-version indices.

## Likely Gap Causes

### High Confidence

1. Official-public split/evaluation protocol is not fully reproduced locally.
   Evidence: Our filtered protocol uses 441 included tiles, separates 15 Bolivia tiles, keeps no_water and warning_review rows, and excludes five fully invalid LabelHand tiles. The exact PANGAEA/TerraMind Sen1Floods11 benchmark config behind the 0.905-0.9078 claim was not found locally.
   Action: Do not compare 0.85-0.86 directly to the public claim until the exact benchmark split and metric aggregation are verified.

2. Training recipe mismatch relative to the official IBM TerraMind config.
   Evidence: Official config uses batch_size=8, precision=16-mixed, max_epochs=100, AdamW lr=2e-5, ReduceLROnPlateau on val/loss, and D4 augmentation. STEP 5O/5P use batch_size=1, precision=32, lr=1e-4, no train_transform, UPerNet BatchNorm eval policy, and mIoU-driven early stopping.
   Action: Before physics loss, run classical ablations that isolate loss only, then optionally a separate recipe-parity run.

3. TerraMind-L UPerNet feature indices may not match the large-backbone recommendation.
   Evidence: STEP 5O/5P use SelectIndices [2, 5, 8, 11]. The IBM official Sen1Floods11 config comments mark [2, 5, 8, 11] for tiny/small/base and [5, 11, 17, 23] for large version.
   Action: Validate large-backbone UPerNet feature indices before treating TerraMind-L UPerNet as architecture-parity complete.

### Medium Confidence

4. Metric aggregation differences.
   Evidence: Our evaluator reports global pixel-confusion mIoU over valid pixels. The exact PANGAEA evaluator settings for Sen1Floods11 were not found locally.
   Action: Recompute tile-mean, event-mean, and global mIoU for existing predictions in a later audit if needed.

5. Composite CE+Dice or weighted CE+Dice could improve flood recall/precision balance, but it is not the official IBM loss.
   Evidence: Official IBM config and our current runs all use dice. PANGAEA supports cross_entropy, weighted_cross_entropy, and dice, but exact TerraMind Sen1Floods11 loss is unknown.
   Action: Run STEP 5S as a small controlled loss ablation against STEP 5O/5I, not as proof of official parity.

6. Checkpoint loading semantics differ from the official config.
   Evidence: Official config sets backbone_pretrained=true; our local configs set backbone_pretrained=false plus an explicit backbone_ckpt_path. Summaries verify checkpoints, but implementation semantics should stay documented.
   Action: Keep checkpoint hash/path in every future run summary.

### Low Confidence

7. Normalization mismatch.
   Evidence: Our configs and official IBM configs use the same TerraMind means/stds for S2L1C and S1GRD.
   Action: No immediate action unless raw preprocessing is changed.

8. Pure CE-vs-Dice loss mismatch.
   Evidence: STEP 5I, STEP 5O, STEP 5P, and the official IBM configs all use dice.
   Action: Keep Dice as STEP 5S control; use CE+Dice variants as robustness ablations.

## STEP 5S Recommendation

Recommended primary loss ablation order:

1. dice
2. ce_dice
3. weighted_ce_dice

Config stubs prepared only:

- `configs/step5s_terramind_l_upernet_dice_stub.yaml`
- `configs/step5s_terramind_l_upernet_ce_dice_stub.yaml`
- `configs/step5s_terramind_l_upernet_weighted_ce_dice_stub.yaml`

These stubs explicitly disable physics loss, topographic loss, and DEM input.

## Sources

- IBM TerraMind page: https://ibm.github.io/terramind/
- IBM TerraMind Sen1Floods11 config: https://raw.githubusercontent.com/IBM/terramind/main/configs/terramind_v1_base_sen1floods11.yaml
- IBM TerraMind TiM-LULC Sen1Floods11 config: https://raw.githubusercontent.com/IBM/terramind/main/configs/terramind_v1_base_tim_lulc_sen1floods11.yaml
- PANGAEA benchmark repository: https://github.com/VMarsocci/pangaea-bench

## Next Step

Human validation required before starting STEP 5S - segmentation-loss ablation before physics-informed training.
