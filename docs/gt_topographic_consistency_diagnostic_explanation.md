# GT Topographic Consistency Diagnostic — Methodology

**Script**: `experiments_cvpr/segman/diagnose_gt_topographic_consistency.py`
**Type**: CPU-only, read-only. No model, no GPU, no training.

---

## Why This Diagnostic Is Needed

The N=50 and N=100 experiments showed that the topographic loss (`Dice+CE+Topo real`)
performs similarly to the shuffled control (`Dice+CE+Topo shuffled`), meaning the model
does not appear to exploit DEM-specific geographic information — the loss may act as a
generic spatial smoothness regularizer.

Before implementing a more sophisticated D8-based loss, we must verify that the signal
we want to encode actually **exists in the data at the label level**. Specifically:

> **Do the ground-truth flood masks sit on lower ground (real DEM) more than they would
> if the DEM were randomly reassigned to a different tile (shuffled DEM)?**

If the answer is **no** — i.e., the GT masks are not more topographically consistent
with the real DEM than with a shuffled one — then no DEM-only loss, however well
formulated, can produce a real-vs-shuffled separation during training. The issue would
be in the data, not the loss.

This diagnostic is the **prerequisite gate** for implementing the D8 loss.

---

## How Metrics Are Computed

All metrics operate on the **ground-truth label mask** and the **DEM array** only.
No model predictions are used.

### 1. Current-Loss Violation (`current_loss_weighted_viol_frac`)

Mirrors the formula of the deployed `TopographicInconsistencyLoss`, but with hard GT
labels instead of predicted probabilities:

```
For each directed 4-neighbor pair (i, j) where h_i > h_j, both pixels valid:
  violation_ij = 1 if GT[i] = water AND GT[j] = dry
  weight_ij    = h_i - h_j   (elevation difference, raw meters)

current_loss_weighted_viol_frac = sum(weight_ij * violation_ij) / sum(weight_ij)
```

This is the **ceiling** on what the current loss can achieve: if this metric does not
separate real from shuffled in GT, the current loss cannot separate them during training.

### 2. D8 Downstream Violation (`d8_weighted_viol_frac`)

Encodes the hydrological prior: water flows downhill, so upstream pixels should not be
water while their downstream neighbor is dry.

```
D8 downstream for pixel i:
  d(i) = argmax_{j in N8(i)} { (h_i - h_j) / dist(i,j) }   (steepest descent)
  drop_i = max(0, h_i - h_d(i))

Validity: m_i = (GT[i] != -1) AND isfinite(h_i)
Active:   active_i = m_i AND m_{d(i)} AND drop_i > 0   (i is actually upstream)
Weight:   w_i = min(1, drop_i / s0)                    (flat pixels get near-zero weight)
Violation: viol_i = GT[i]=water AND GT[d(i)]=dry

d8_weighted_viol_frac = sum(w_i * viol_i * active_i) / sum(w_i * active_i)
```

This is more DEM-specific than the 4-neighbor version because the downstream direction
`d(i)` is determined by the full 8-neighborhood and depends on the exact elevation field.
A shuffled DEM reroutes `d(i)` to a different neighbor, changing which pairs are penalized.

### 3. Elevation Distribution Statistics

```
delta_mean_elev   = mean(h at dry pixels) - mean(h at water pixels)
delta_median_elev = median(h at dry) - median(h at water)
```

A **positive delta** means water pixels are on lower ground than dry pixels — consistent
with the downhill-water prior. A **negative delta** indicates the opposite (water on
higher ground: levees, DSM buildings, flat plains).

### 4. Elevation AUC (`elev_auc`)

AUROC of `−elevation` as a predictor of the GT water label:

- AUC = 0.5 → elevation has no discriminative power for water prediction
- AUC > 0.5 → lower elevation predicts water better than random
- AUC = 1.0 → perfect: all water pixels below all dry pixels

This is a global, threshold-free measure of how much elevation encodes the GT flood extent.

---

## How to Interpret Real vs Shuffled

For each metric, the comparison is:

| Situation | Violation metrics (↓) | AUC / elev_delta (↑) | Interpretation |
|-----------|----------------------|----------------------|----------------|
| **Real clearly better** | real ≪ shuffled | real ≫ shuffled | DEM-specific signal exists in GT → implement D8 loss |
| **Real ≈ Shuffled** | real ≈ shuffled | real ≈ shuffled | No DEM-alignment signal at this resolution → report negative result |
| **Shuffled better** | real ≫ shuffled | real ≪ shuffled | DEM co-registration problem or wrong split DEM → investigate alignment |

The key numerical criterion is:
- **Paired delta** `(real − shuffled)` should be **negative** for violation metrics (real fewer violations)
- **Percentage of tiles where real is better** should be well above 50%
- The effect should be **consistent across val/test** splits (different tiles from train)

---

## Limitations

The following known confounds reduce the expected real-vs-shuffled gap and must be
reported alongside the results:

1. **DEM resolution mismatch (30 m vs 10 m)**
   The GLO-30 DEM is resampled from 30 m resolution, while S1/S2 imagery and labels
   are at 10 m. Each DEM pixel covers ~9 label pixels. Pixel-pair constraints at 10 m
   under a 30 m DEM carry noise from the resampling process.

2. **DSM vs DTM**
   Copernicus GLO-30 is a *Digital Surface Model* — it includes building rooftops, tree
   canopy, and infrastructure. In urban areas, flooded streets may appear at the same
   or lower elevation than nearby buildings, breaking the "water is low" prior.
   A bare-earth DTM (Digital Terrain Model) would be more appropriate but is not
   readily available at global scale.

3. **Urban drainage not represented**
   Flood water can be retained at locally high elevations by levees, concrete walls,
   basements, or pumped channels. The DEM does not encode these hydraulic controls.
   Urban tiles in Sen1Floods11 are systematically harder for DEM-based losses.

4. **Static topography vs. event-driven flooding**
   The DEM describes static terrain; the GT mask describes a specific flood *event*
   driven by upstream rainfall, dam releases, or storm surge. The D8 flow network
   derived from the DEM gives *potential* drainage paths, not *observed* water extent
   for any particular event.

5. **Flat floodplains**
   Many large flood events (Bangladesh, Pakistan, West Africa) occur on near-flat
   floodplains where D8 flow direction is ambiguous and elevation differences between
   water and non-water areas are very small (< 1 m over km distances). The slope
   weight `w_i = min(1, drop_i / s0)` with `s0 = 1.0 m` is designed to downweight
   these ambiguous flat regions, but this also reduces the effective signal.

---

## Output Files

| File | Content |
|------|---------|
| `reports/gt_topographic_consistency_n100_seed0.csv` | Per-tile metrics (real, shuffled, delta) |
| `reports/gt_topographic_consistency_n100_seed0.json` | Aggregated statistics per split |
| `docs/gt_topographic_consistency_n100_seed0.md` | Auto-generated summary table |

---

## Quick Commands

```powershell
# Quick debug (10 tiles from train split)
E:/flood_research/venvs/terramind-gpu/Scripts/python.exe `
    experiments_cvpr/segman/diagnose_gt_topographic_consistency.py `
    --split train --max-tiles 10

# Full N=100 train diagnostic (primary analysis)
E:/flood_research/venvs/terramind-gpu/Scripts/python.exe `
    experiments_cvpr/segman/diagnose_gt_topographic_consistency.py `
    --split train

# All splits with synthetic shuffle on val/test/bolivia
E:/flood_research/venvs/terramind-gpu/Scripts/python.exe `
    experiments_cvpr/segman/diagnose_gt_topographic_consistency.py `
    --split all --synthetic-shuffle
```
