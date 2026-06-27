# SegMAN-S N=50 Multi-Seed Loss Ablation — Aggregated Results

**Seeds**: [0, 1, 2, 3, 42]
**Conditions**: ['CE', 'Dice+CE', 'Dice+CE+Topo', 'Dice+CE+Topo+Shuffled']
**Loaded runs**: 20 / 20

## Val Split

| Condition | mIoU (mean±std) | IoU_water | F1_water | Prec_water | Rec_water | Topo_viol |
|-----------|-----------------|-----------|----------|------------|-----------|-----------|
| CE | 0.8233 ± 0.0135 | 0.6884 ± 0.0237 | 0.8153 ± 0.0166 | 0.8694 ± 0.0183 | 0.7679 ± 0.0238 | 0.0006 ± 0.0001 |
| Dice+CE | 0.8207 ± 0.0166 | 0.6855 ± 0.0292 | 0.8131 ± 0.0207 | 0.8329 ± 0.0193 | 0.7952 ± 0.0378 | 0.0007 ± 0.0002 |
| Dice+CE+Topo | 0.8240 ± 0.0131 | 0.6915 ± 0.0225 | 0.8174 ± 0.0158 | 0.8304 ± 0.0315 | 0.8062 ± 0.0299 | 0.0008 ± 0.0002 |
| Dice+CE+Topo+Shuffled | 0.8291 ± 0.0109 | 0.7002 ± 0.0193 | 0.8235 ± 0.0134 | 0.8379 ± 0.0216 | 0.8107 ± 0.0308 | 0.0009 ± 0.0002 |

## Test Split

| Condition | mIoU (mean±std) | IoU_water | F1_water | Prec_water | Rec_water | Topo_viol |
|-----------|-----------------|-----------|----------|------------|-----------|-----------|
| CE | 0.8368 ± 0.0168 | 0.7178 ± 0.0289 | 0.8354 ± 0.0198 | 0.8680 ± 0.0171 | 0.8055 ± 0.0292 | 0.0008 ± 0.0001 |
| Dice+CE | 0.8401 ± 0.0104 | 0.7247 ± 0.0177 | 0.8403 ± 0.0118 | 0.8432 ± 0.0141 | 0.8376 ± 0.0168 | 0.0009 ± 0.0002 |
| Dice+CE+Topo | 0.8465 ± 0.0114 | 0.7361 ± 0.0189 | 0.8479 ± 0.0126 | 0.8393 ± 0.0202 | 0.8569 ± 0.0109 | 0.0009 ± 0.0002 |
| Dice+CE+Topo+Shuffled | 0.8479 ± 0.0186 | 0.7382 ± 0.0319 | 0.8491 ± 0.0212 | 0.8454 ± 0.0199 | 0.8533 ± 0.0323 | 0.0010 ± 0.0002 |

## Bolivia Split

| Condition | mIoU (mean±std) | IoU_water | F1_water | Prec_water | Rec_water | Topo_viol |
|-----------|-----------------|-----------|----------|------------|-----------|-----------|
| CE | 0.8074 ± 0.0157 | 0.6804 ± 0.0260 | 0.8096 ± 0.0187 | 0.8542 ± 0.0425 | 0.7722 ± 0.0441 | 0.0015 ± 0.0003 |
| Dice+CE | 0.8119 ± 0.0271 | 0.6884 ± 0.0448 | 0.8148 ± 0.0312 | 0.8491 ± 0.0404 | 0.7854 ± 0.0512 | 0.0014 ± 0.0003 |
| Dice+CE+Topo | 0.8136 ± 0.0227 | 0.6920 ± 0.0359 | 0.8176 ± 0.0252 | 0.8444 ± 0.0688 | 0.7998 ± 0.0641 | 0.0015 ± 0.0003 |
| Dice+CE+Topo+Shuffled | 0.8205 ± 0.0268 | 0.7042 ± 0.0438 | 0.8258 ± 0.0305 | 0.8314 ± 0.0468 | 0.8244 ± 0.0586 | 0.0016 ± 0.0004 |

## Per-Seed Test mIoU Detail

| Condition | seed0 | seed1 | seed2 | seed3 | seed42 |
|-----------|-----|-----|-----|-----|-----|
| CE | 0.8420 | 0.8290 | 0.8119 | 0.8461 | 0.8552 |
| Dice+CE | 0.8389 | 0.8378 | 0.8580 | 0.8337 | 0.8321 |
| Dice+CE+Topo | 0.8496 | 0.8386 | 0.8574 | 0.8310 | 0.8559 |
| Dice+CE+Topo+Shuffled | 0.8637 | 0.8363 | 0.8472 | 0.8238 | 0.8685 |

## Notes

- DEM-shuffle scope verified: shuffle applies to training tiles only; eval uses real DEM.
- See `docs/segman_dem_shuffle_scope_check.md` for implementation audit.
- See `docs/segman_seed0_topo_audit.md` for seed0 baseline analysis.
- Aggregation script: `experiments_cvpr/segman/aggregate_multiseed_results.py`
