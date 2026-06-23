# STEP 6C/v3 Physics Warmup Final Report

## Scope

This report finalizes the completed STEP 6C/v3 physics warmup run. No new training, DARN run, STURM-Flood run, or raw-data modification was performed.

## Run

- Config: `configs/step6c_v3_terramind_l_upernet_dice_topographic_lambda05_warmup_5sa_dataloader.yaml`
- Run directory: `E:/flood_research/experiments/terramind_baseline/runs/step6c_v3_terramind_l_upernet_dice_topographic_lambda05_warmup_5sa_dataloader/`
- Best checkpoint: `E:/flood_research/experiments/terramind_baseline/runs/step6c_v3_terramind_l_upernet_dice_topographic_lambda05_warmup_5sa_dataloader/checkpoints/best_checkpoint.pt`
- Best epoch: 54
- Best validation mIoU: 0.8784941101579701
- Checkpoint epoch: 54
- Physics loss active: yes
- DEM as model input: no
- DEM used in loss only: yes
- Dataloader-side D4: yes

## Physics Schedule

- Epochs 1-5: `lambda_topo = 0`
- Epochs 6-19: warmup ramp
- Epochs 20-80: `lambda_topo = 0.5`
- Best epoch 54 evaluated with `lambda_topo = 0.5`

## Final Metrics

| Split | mIoU | IoU water | F1 water | Pred water pixels | Pred water fraction | loss_dice | loss_topo | topo_score | topo_violation_fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| valid | 0.878494 | 0.786425 | 0.880446 | 2270759 | 0.111889 | 0.144499 | 0.000187676 | 0.000187676 | 0.001483888 |
| test | 0.872132 | 0.778442 | 0.875420 | 2459204 | 0.119860 | 0.149937 | 0.000189086 | 0.000189086 | 0.001441069 |
| Bolivia/OOD | 0.846406 | 0.743236 | 0.852709 | 401867 | 0.140130 | 0.181189 | 0.000151526 | 0.000151526 | 0.002341030 |

Saved metrics:

- JSON: `E:/flood_research/experiments/terramind_baseline/runs/step6c_v3_terramind_l_upernet_dice_topographic_lambda05_warmup_5sa_dataloader/metrics/step6c_v3_physics_warmup_final_metrics.json`
- CSV: `E:/flood_research/experiments/terramind_baseline/runs/step6c_v3_terramind_l_upernet_dice_topographic_lambda05_warmup_5sa_dataloader/metrics/step6c_v3_physics_warmup_final_metrics.csv`

## Comparison Against STEP 5S-A

| Split | STEP 6C/v3 mIoU | STEP 5S-A mIoU | Delta mIoU | STEP 6C/v3 IoU water | STEP 5S-A IoU water | Delta IoU water | STEP 6C/v3 F1 water | STEP 5S-A F1 water | Delta F1 water |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| valid | 0.878494 | 0.881233 | -0.002738 | 0.786425 | 0.791499 | -0.005073 | 0.880446 | 0.883616 | -0.003170 |
| test | 0.872132 | 0.873051 | -0.000919 | 0.778442 | 0.780194 | -0.001753 | 0.875420 | 0.876527 | -0.001107 |
| Bolivia/OOD | 0.846406 | 0.843955 | +0.002451 | 0.743236 | 0.738797 | +0.004440 | 0.852709 | 0.849779 | +0.002930 |

Interpretation: STEP 6C/v3 is slightly below STEP 5S-A on valid and test, but slightly above STEP 5S-A on Bolivia/OOD. The effect size is small, so the physics run should be treated as promising for OOD but not yet a clear overall improvement.

## STEP 5S-A Topographic Metrics

STEP 5S-A topographic metrics were computed in a follow-up no-training evaluation using the same STEP 6C/v3 loss-only DEM dataloader path. DEM was not used as model input.

Command run:

```powershell
E:/flood_research/venvs/terramind-gpu/Scripts/python.exe scripts/evaluate_step5s_a_topographic_metrics.py --checkpoint E:/flood_research/experiments/terramind_baseline/runs/step5s_a_terramind_l_upernet_corrected_indices_dice_bs2_accum4/checkpoints/best_checkpoint.pt --config configs/step6c_v3_terramind_l_upernet_dice_topographic_lambda05_warmup_5sa_dataloader.yaml --output-dir E:/flood_research/experiments/terramind_baseline/runs/step5s_a_topographic_metrics_eval_only
```

Saved STEP 5S-A topo metrics:

- JSON: `E:/flood_research/experiments/terramind_baseline/runs/step5s_a_topographic_metrics_eval_only/metrics/step5s_a_topographic_metrics.json`
- CSV: `E:/flood_research/experiments/terramind_baseline/runs/step5s_a_topographic_metrics_eval_only/metrics/step5s_a_topographic_metrics.csv`

| Split | STEP 5S-A topo_score | STEP 6C/v3 topo_score | Delta | Relative change | STEP 5S-A violation frac | STEP 6C/v3 violation frac | Delta | Relative change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| valid | 0.000203058 | 0.000187676 | -0.000015383 | -7.576% | 0.001611107 | 0.001483888 | -0.000127219 | -7.896% |
| test | 0.000204709 | 0.000189086 | -0.000015623 | -7.632% | 0.001592144 | 0.001441069 | -0.000151075 | -9.489% |
| Bolivia/OOD | 0.000161123 | 0.000151526 | -0.000009597 | -5.957% | 0.002530160 | 0.002341030 | -0.000189129 | -7.475% |

STEP 6C/v3 reduced topographic inconsistency score and violation fraction on all three splits compared with STEP 5S-A. This physical-consistency gain comes with small valid/test segmentation declines and a small Bolivia/OOD segmentation gain.

## Recommendation

Do not launch STEP 6D automatically. Human/Claude should review the trade-off: STEP 6C/v3 improves topographic metrics consistently and slightly improves Bolivia/OOD segmentation, but slightly underperforms STEP 5S-A on valid/test segmentation. A next reasonable step is a small controlled lambda sweep around 0.5 or a fixed-quality comparison, but only after human validation.
