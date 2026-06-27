# SegMAN-S N=50 — Visual and Physical Analysis

**Date:** 2026-06-27
**Experiment:** SegMAN-S loss ablation (CE / Dice+CE / Dice+CE+Topo / Dice+CE+Topo+Shuffled)
**Seeds:** 0, 1, 2, 3, 42 — N_train = 50
**Script:** `experiments_cvpr/segman/analyze_segman_n50_visuals.py`

---

## 1. Objective

Assess whether the real-DEM topographic loss produces physically distinct predictions
compared with (a) the CE and Dice+CE baselines and (b) the DEM-shuffled control.
The key scientific question is whether the effect of the topographic loss reflects
genuine physical grounding or merely acts as a weak regularizer.

---

## 2. Artifacts found

| Artifact | Location | Count | Status |
|----------|----------|-------|--------|
| NPZ prediction files (pred + target, 512×512) | `E:/…/runs/{tag}/predictions/test/` | 6 per run × 20 runs = 120 | All present |
| Test manifest | `…step5e_tiny_unetdecoder_baseline/manifests/flood_test_step5e_filtered.txt` | 89 images | Present |
| Aligned DEM files | `E:/flood_research/data/derived/sen1floods11_topography/dem_aligned/` | 89 test tiles | Present |
| Sentinel-2 L1C imagery | `E:/…/HandLabeled/S2Hand/` | 446 tiles | Present |
| Summary JSON (per run) | `E:/…/runs/{tag}/metrics/{tag}_summary.json` | 20 | All status=done |
| Per-run epoch CSV | `E:/…/runs/{tag}/metrics/training_epoch_metrics.csv` | 20 | Present |

**Limitation on prediction coverage:** Each run saves only the first 6 test images from
the dataloader (deterministic order, no shuffle). These 6 samples cover Ghana flood
events only and are **not** a random or stratified sample of the 89 test images.
Full per-image analysis of all 89 images would require a re-inference pass (not done;
no GPU operation was performed in this analysis).

---

## 3. Selection criteria for the 6 analyzed images

The NPZ files were saved by `train_segman.py → save_pred_samples(max_samples=6)`,
which saves the first 6 batches of the test dataloader (fixed order, deterministic
across seeds and conditions). Target arrays are confirmed aligned across all 4
conditions × 5 seeds (identical ground truth per index).

Images:

| idx | Tile ID | GT water ratio | Notes |
|-----|---------|----------------|-------|
| 000 | Ghana_1078550 | 0.40 | High water coverage |
| 001 | Ghana_141271  | 0.66 | Dominant water |
| 002 | Ghana_167233  | 0.006 | Near-dry (only 0.6% water) |
| 003 | Ghana_313799  | 0.037 | Sparse water, high seed variance |
| 004 | Ghana_319168  | ~0 | Essentially no water |
| 005 | Ghana_359826  | 0.156 | Moderate water |

Images 002 and 004 are near-dry; IoU_water ≈ 0 for all conditions (degenerate
cases where water class is essentially absent). Results for these images are
omitted from the physical interpretation.

---

## 4. Per-image metric summary

Mean IoU_water (± std over 5 seeds) by condition:

| idx | Tile ID | CE | Dice+CE | Topo real | Topo shuffled |
|-----|---------|----|---------|-----------|--------------:|
| 000 | Ghana_1078550 | 0.764 ± 0.042 | 0.783 ± 0.029 | 0.782 ± 0.028 | **0.795 ± 0.019** |
| 001 | Ghana_141271  | 0.865 ± 0.018 | 0.873 ± 0.021 | **0.877 ± 0.021** | 0.877 ± 0.028 |
| 002 | Ghana_167233  | 0.000 | 0.000 | 0.000 | 0.005 |
| 003 | Ghana_313799  | **0.798** ± 0.200 | 0.752 ± 0.255 | **0.801** ± 0.143 | 0.752 ± 0.159 |
| 004 | Ghana_319168  | ~0 | ~0 | ~0 | ~0 |
| 005 | Ghana_359826  | 0.426 ± 0.037 | **0.438** ± 0.050 | 0.434 ± 0.060 | 0.420 ± 0.049 |

Mean deltas over the 6 images (mean over seeds):

| Comparison | ΔIoU_water |
|-----------|-----------|
| Dice+CE − CE | −0.0011 ± 0.0229 |
| Topo real − Dice+CE | **+0.0081 ± 0.0200** |
| Shuffled − Topo real | −0.0077 ± 0.0223 |

> Note: these deltas are over 6 non-representative images (Ghana flood events only).
> Full test set (89 images, all seeds) showed Shuffled slightly above Topo_real (+0.14%)
> on average. The two estimates are consistent within noise.

---

## 5. Topographic violation analysis

Computed from seed0 hard masks using the diagnostic rule:
> A pixel is a violation if it is predicted water, at least one 4-connected neighbour
> is predicted dry, and its DEM elevation exceeds that neighbour's elevation (margin=0 m).

This is a **diagnostic approximation**, not the exact differentiable training loss
(which uses soft logits and a warmup schedule). Results are qualitative.

| idx | Tile | CE | Dice+CE | Topo real | Topo shuffled |
|-----|------|----|---------|-----------|---------------|
| 000 | Ghana_1078550 | 0.0142 | 0.0144 | **0.0141** | 0.0186 |
| 001 | Ghana_141271  | 0.0121 | 0.0097 | **0.0082** | 0.0104 |
| 002 | Ghana_167233  | 0 | 0 | 0 | 0 |
| 003 | Ghana_313799  | 0.0009 | 0.0010 | 0.0025 | 0.0027 |
| 004 | Ghana_319168  | 0.0098 | 0.0102 | 0.0101 | 0.0135 |
| 005 | Ghana_359826  | 0.0123 | 0.0133 | 0.0125 | 0.0142 |

Observations:
- **Ghana_141271 (idx=001):** Topo real achieves the lowest topo violation fraction
  (0.0082) vs CE (0.0121) and Shuffled (0.0104). This is the clearest case where
  the real DEM variant reduces physical inconsistencies relative to the other conditions.
- **Ghana_1078550 (idx=000):** Topo real is marginally better than CE but the Shuffled
  variant shows *more* violations than CE — yet also achieves the highest IoU_water.
  This suggests the shuffled DEM may over-predict water in physically inconsistent
  locations but still match the GT better (the GT itself may contain such regions).
- **Ghana_313799 (idx=003):** Topo real and Shuffled both show higher violations than
  CE on this image, possibly because the topo term encourages coverage in ambiguous
  low-lying zones that the CE model avoids.
- Violation fractions are globally very low (<2%) across all conditions. The
  absolute differences between conditions are in the range 0.001–0.006 (less than 0.6%
  of pixels), which is within noise at N=6 images.

---

## 6. Does Topo real visibly reduce physical inconsistency?

**Partially, and only on selected images.** On Ghana_141271, the Topo real variant
shows a measurable reduction in topo violations versus all other conditions.
On the remaining images the differences are negligible or absent.

The multi-seed aggregate results (89 test images, 5 seeds) provide additional context:
- Topo real mean test mIoU: 0.8465 ± 0.0114
- Topo shuffled mean test mIoU: 0.8479 ± 0.0186
- Difference: +0.0014 ± noise — not statistically distinguishable with 5 seeds

Combined verdict: **there is weak evidence of reduced physical inconsistency on
specific images, but the signal is not consistent across the sample set, and the
IoU and topo-violation metrics do not demonstrate a robust physical DEM effect.**

---

## 7. Does DEM-Shuffled behave similarly to Topo real?

**Yes, largely.** On the 6 saved images:
- IoU_water differences between Topo real and Shuffled are all within the seed std
- Topo violation fraction differences are small and not directionally consistent
- On 4/6 images the Shuffled variant has equal or slightly higher violations than
  Topo real, which is the physically expected direction (real DEM should penalise
  violations more precisely). But the magnitude is tiny.

On the full 89-image test set, Shuffled is negligibly above Topo real (+0.14% mIoU).

This is consistent with the interpretation that the topographic term acts primarily
as a **geometric regularizer** (constraining prediction shape/smoothness via spatial
gradient-like constraints) rather than as a **physically-informed signal** that
specifically encodes DEM elevation structure.

---

## 8. Figures generated (locally, not committed to git)

Figures are saved locally to `reports/figures/segman_n50/` and are excluded from git
(`.gitignore` rule: `reports/figures/`). They can be regenerated at any time with:

```bash
python experiments_cvpr/segman/analyze_segman_n50_visuals.py \
    --exp-root E:/flood_research/experiments/segman \
    --output-dir reports/figures/segman_n50
```

| File | Description |
|------|-------------|
| `panels/panel_{i:03d}_{tile_id}.png` | 7-column comparison: S2 RGB / GT / CE / Dice+CE / Topo / Shuffled / DEM |
| `topo_violations/topo_viol_{i:03d}_{tile_id}.png` | DEM-overlaid violation maps (red = violation) |
| `summary_iou_water_per_image.png` | Bar chart IoU_water per image × condition |
| `tables/per_image_metrics.csv` | Full per-image per-seed metric table |
| `tables/delta_summary.csv` | Paired deltas per image |
| `tables/topo_violation_per_image.json` | Topo violation fractions per image × condition |

---

## 9. Limitations

1. **Only 6 of 89 test images** are available as NPZ predictions — the first 6 in
   dataloader order, all from Ghana. No Bolivia/OOD or other geographic region is
   represented in the visual analysis.
2. **No valid/bolivia NPZs** were saved during training — only the test split.
3. **Hard masks only** — the NPZs contain argmax predictions (uint8 0/1), not soft
   probabilities. Soft-logit topo violation analysis is not possible without re-inference.
4. **Topo violation is diagnostic**, not the training loss. The training loss uses
   soft logits with a warmup schedule; this analysis uses hard masks at evaluation time.
5. **All 6 images are Ghana flood events** — generalization to other geographies
   (Bolivia OOD, other Sen1Floods11 regions) cannot be assessed from these samples.
6. **N=5 seeds with 6 images** gives a very small sample for statistical conclusions.

---

## 10. Recommended next action

**Option A (preferred short-term):** Accept current results as preliminary evidence
and proceed to the next model in the roadmap (TerraMind baseline completion),
treating the SegMAN topographic loss result as "regularization plausible, physical
DEM effect not established."

**Option B (if physical DEM effect is the primary research question):** Run a
dedicated re-inference pass over all 89 test images for 4 conditions × 5 seeds
(CPU-only, no new training) to compute full per-image metrics and topo violations.
This would require:
```bash
# Command to generate (not launched — requires GPU or CPU inference):
python experiments_cvpr/segman/train_segman.py --eval-only \
    --config configs/segman/multiseed_n50/segman_dice_ce_topo_n50_seed0.yaml \
    --save-all-predictions
```
Requires explicit approval before launching.

**Option C (if DEM effect needs stronger testing):** Reformulate or reweight
the topographic constraint. Possible directions:
- Increase `lambda_topo` beyond 0.5
- Use elevation-weighted violation counts (not uniform)
- Experiment at N=100 or N=200 for reduced seed variance
- Use soft-probability topo loss at evaluation time, not just hard masks

---

## Scientific conclusion

> Based on the multi-seed aggregate results (N=50, 5 seeds, 20 runs, 89 test images)
> and the qualitative/physical inspection of 6 saved prediction samples, SegMAN-S is
> validated as a strong 15-channel low-data segmentation backbone (mean test
> mIoU 0.83–0.85 across conditions). However, the current topographic loss does not yet
> provide clear evidence of a DEM-specific physical effect. The real DEM version and
> DEM-shuffled version behave similarly on both IoU metrics and topo violation maps,
> suggesting that the term may act mainly as a weak regularizer. One image
> (Ghana_141271) shows a measurable reduction in topographic violations under Topo real,
> but this is not consistent across the sample. Further work should focus on more
> sensitive physical metrics, full-inference qualitative error analysis across all
> 89 test images, and possibly reformulating or reweighting the topographic constraint
> before launching large additional experiments.
