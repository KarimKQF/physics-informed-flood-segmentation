"""
GT Topographic Consistency Diagnostic
======================================
CPU-only, no training, no model, no GPU.

Tests whether the ground-truth flood/water masks (Sen1Floods11) are more
topographically consistent with the real aligned DEM than with a shuffled DEM.

This is the required prerequisite before implementing a D8-based physics loss.
If GT masks are not more consistent with the real DEM than with shuffled DEM,
then no DEM-only loss is likely to produce a real-vs-shuffled separation.

Three metrics per tile:

1. current_loss_violation : weighted fraction of 4-neighbor pairs where
   (h_i > h_j, GT water at i, GT dry at j). Matches the formula used in
   the currently-trained TopographicInconsistencyLoss.

2. d8_violation : weighted fraction of D8-downstream pairs where GT water is
   upstream and GT dry is downstream (slope-weighted by drop / s0).

3. elev_auc : AUROC of "-elevation" as a predictor of GT water label.
   An AUC > 0.5 means water tends to occur at lower elevations.

For train tiles the comparison uses the Sattolo derangement shuffle map
(dem_shuffle_map_n100_seed0.json). For val/test/bolivia tiles a synthetic
cyclic-shift shuffle is computed from the tiles in that split.

Usage examples:
    # Quick 10-tile debug
    python experiments_cvpr/segman/diagnose_gt_topographic_consistency.py \\
        --split train --max-tiles 10

    # Full N=100 train set (primary diagnostic)
    python experiments_cvpr/segman/diagnose_gt_topographic_consistency.py \\
        --split train

    # All splits
    python experiments_cvpr/segman/diagnose_gt_topographic_consistency.py \\
        --split all
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import rasterio
except ImportError:
    sys.exit("rasterio not available. Run with the project venv.")

try:
    from sklearn.metrics import roc_auc_score
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
    print("WARNING: sklearn not available; elev_auc will not be computed.")

# ── Paths (mirrors step6c_v3_train conventions) ───────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]

LABEL_DIR    = Path("E:/flood_research/data/raw/sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand")
DEM_DIR      = Path("E:/flood_research/data/derived/sen1floods11_topography/dem_aligned")
DEM_PATTERN  = "{split}_{tile_id}_copernicus_glo30_dem_aligned.tif"
LABEL_PATTERN = "{tile_id}_LabelHand.tif"

# DEM split name: the DEM files use "valid" not "val"
SPLIT_DEM_NAME = {"train": "train", "val": "valid", "valid": "valid",
                  "test": "test", "bolivia": "bolivia"}

MANIFEST_TRAIN_N100_S0 = REPO_ROOT / "manifests/terramind_baseline/low_data_multiseed_n100/flood_train_low_data_n100_seed0.txt"
SHUFFLE_MAP_N100_S0    = REPO_ROOT / "manifests/terramind_baseline/low_data_multiseed_n100/dem_shuffle_map_n100_seed0.json"

MANIFESTS = {
    "train": MANIFEST_TRAIN_N100_S0,
    "val":   Path("E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/manifests/flood_valid_step5e_filtered.txt"),
    "test":  Path("E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/manifests/flood_test_step5e_filtered.txt"),
    "bolivia": Path("E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/manifests/flood_bolivia_step5e_filtered.txt"),
}

IGNORE_INDEX = -1
WATER_CLASS  = 1
DRY_CLASS    = 0
D8_OFFSETS   = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
D8_DISTS     = [math.sqrt(2), 1.0, math.sqrt(2), 1.0, 1.0, math.sqrt(2), 1.0, math.sqrt(2)]


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_label(tile_id: str) -> np.ndarray:
    path = LABEL_DIR / LABEL_PATTERN.format(tile_id=tile_id)
    if not path.exists():
        raise FileNotFoundError(f"Label missing: {path}")
    with rasterio.open(path) as src:
        return src.read(1).astype(np.int16)


def load_dem(split: str, tile_id: str) -> np.ndarray | None:
    dem_split = SPLIT_DEM_NAME.get(split, split)
    path = DEM_DIR / DEM_PATTERN.format(split=dem_split, tile_id=tile_id)
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def read_manifest(split: str) -> list[str]:
    p = MANIFESTS.get(split)
    if p is None or not p.exists():
        raise FileNotFoundError(f"Manifest not found for split '{split}': {p}")
    lines = p.read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip()]


def load_shuffle_map() -> dict[str, str]:
    if not SHUFFLE_MAP_N100_S0.exists():
        return {}
    raw = json.loads(SHUFFLE_MAP_N100_S0.read_text(encoding="utf-8"))
    return raw.get("mapping", {})


# ── Metric computations ───────────────────────────────────────────────────────

def _valid_mask(label: np.ndarray, dem: np.ndarray) -> np.ndarray:
    """Boolean mask: pixel is non-ignore and DEM is finite."""
    return (label != IGNORE_INDEX) & np.isfinite(dem)


def compute_current_loss_violation(
    label: np.ndarray, dem: np.ndarray
) -> dict[str, float]:
    """
    Mirrors TopographicInconsistencyLoss with hard GT labels.

    For each directed 4-neighbor pair (i→j) where h_i > h_j:
      - active if both valid
      - violation if GT water at i AND GT dry at j
      - weight = (h_i - h_j)  [elevation margin = 0, elevation_scale = 1.0]

    Returns:
      weighted_viol_frac   : sum(w_ij * viol_ij) / sum(w_ij * active_ij)
      unweighted_viol_frac : count(active_ij where viol) / count(active_ij)
      n_active_pairs       : number of directed pairs with h_i > h_j (both valid)
    """
    valid = _valid_mask(label, dem)
    water = (label == WATER_CLASS)
    dry   = (label == DRY_CLASS)

    H, W = dem.shape
    # 4-neighbor offsets (directed right + down, then bidirectional)
    offsets = [(0, 1), (1, 0)]

    w_viol_sum   = 0.0
    w_total_sum  = 0.0
    uw_viol_count = 0
    uw_total_count = 0

    for dy, dx in offsets:
        for pa, pb in [("a->b", False), ("b->a", True)]:
            if not pb:
                y_a_s, y_a_e = 0, H - dy if dy > 0 else H
                x_a_s, x_a_e = 0, W - dx if dx > 0 else W
                y_b_s, y_b_e = dy, H
                x_b_s, x_b_e = dx, W
            else:
                y_a_s, y_a_e = dy, H
                x_a_s, x_a_e = dx, W
                y_b_s, y_b_e = 0, H - dy if dy > 0 else H
                x_b_s, x_b_e = 0, W - dx if dx > 0 else W

            h_a  = dem  [y_a_s:y_a_e, x_a_s:x_a_e]
            h_b  = dem  [y_b_s:y_b_e, x_b_s:x_b_e]
            va   = valid[y_a_s:y_a_e, x_a_s:x_a_e]
            vb   = valid[y_b_s:y_b_e, x_b_s:x_b_e]
            wa   = water[y_a_s:y_a_e, x_a_s:x_a_e]
            da   = dry  [y_a_s:y_a_e, x_a_s:x_a_e]
            wb   = water[y_b_s:y_b_e, x_b_s:x_b_e]
            db   = dry  [y_b_s:y_b_e, x_b_s:x_b_e]

            delta = h_a - h_b
            active = va & vb & (delta > 0)
            weights = np.where(active, delta, 0.0)

            viol = active & wa & db   # a is high+water, b is low+dry

            w_viol_sum    += float(weights[viol].sum())
            w_total_sum   += float(weights[active].sum())
            uw_viol_count += int(viol.sum())
            uw_total_count += int(active.sum())

    eps = 1e-9
    return {
        "current_loss_weighted_viol_frac":   w_viol_sum  / (w_total_sum   + eps),
        "current_loss_unweighted_viol_frac": uw_viol_count / (uw_total_count + eps),
        "current_loss_n_active_pairs": uw_total_count,
    }


def _compute_d8(dem: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute D8 steepest-descent downstream pixel index.

    Returns (d_row, d_col, drop) where drop = h[i,j] - h[d_row[i,j], d_col[i,j]].
    For local sinks / flat pixels, downstream = self and drop = 0.
    """
    H, W = dem.shape
    pad = np.pad(dem, 1, mode="edge")  # replicate border to avoid NaN issues

    best_slope = np.full((H, W), -np.inf, dtype=np.float32)
    d_row = np.zeros((H, W), dtype=np.int32)
    d_col = np.zeros((H, W), dtype=np.int32)
    rows = np.arange(H, dtype=np.int32)[:, None]
    cols = np.arange(W, dtype=np.int32)[None, :]

    for (dy, dx), dist in zip(D8_OFFSETS, D8_DISTS):
        # Neighbor in padded array
        nbr = pad[1 + dy : H + 1 + dy, 1 + dx : W + 1 + dx]
        slope = (dem - nbr) / dist  # positive = downslope to neighbor

        better = slope > best_slope
        best_slope = np.where(better, slope, best_slope)
        n_row = np.clip(rows + dy, 0, H - 1).astype(np.int32)
        n_col = np.clip(cols + dx, 0, W - 1).astype(np.int32)
        d_row = np.where(better, n_row, d_row)
        d_col = np.where(better, n_col, d_col)

    # Sinks / flat: slope <= 0 → downstream = self, drop = 0
    is_sink = best_slope <= 0
    d_row = np.where(is_sink, rows, d_row)
    d_col = np.where(is_sink, cols, d_col)

    drop = np.maximum(0.0, dem - dem[d_row, d_col])
    return d_row, d_col, drop


def compute_d8_violation(
    label: np.ndarray,
    dem: np.ndarray,
    s0: float = 1.0,
) -> dict[str, float]:
    """
    D8 downstream-consistency violation on GT masks.

    For each pixel i:
      d(i)  = argmin_{j in N8} h_j  (steepest-descent downstream)
      drop_i = max(0, h_i - h_d(i))
      w_i   = min(1, drop_i / s0)          [flat pixels: w ≈ 0]
      active_i = valid_i AND valid_d(i) AND drop_i > 0  (i is upstream of d(i))
      violation_i = GT_water(i) AND GT_dry(d(i))

    Weighted violation fraction = sum(w_i * viol_i * active_i) / sum(w_i * active_i)

    Lower = better (fewer GT violations).
    """
    valid = _valid_mask(label, dem)
    water = (label == WATER_CLASS)
    dry   = (label == DRY_CLASS)

    d_row, d_col, drop = _compute_d8(dem)

    valid_d = valid[d_row, d_col]
    water_d = water[d_row, d_col]
    dry_d   = dry  [d_row, d_col]

    w = np.clip(drop / s0, 0.0, 1.0)
    active = valid & valid_d & (drop > 0)

    viol   = active & water & dry_d      # upstream water, downstream dry
    viol_rev = active & dry & water_d    # upstream dry, downstream water (note: not a "violation" by convention)

    eps = 1e-9
    w_active = float((w * active).sum())
    w_viol   = float((w * viol).sum())

    return {
        "d8_weighted_viol_frac":   w_viol / (w_active + eps),
        "d8_unweighted_viol_frac": int(viol.sum()) / (int(active.sum()) + eps),
        "d8_n_active_pixels":      int(active.sum()),
        "d8_s0": s0,
    }


def compute_elevation_stats(
    label: np.ndarray,
    dem: np.ndarray,
) -> dict[str, float]:
    """
    Per-tile elevation distribution diagnostics.

    For valid pixels:
      mean_elev_water, mean_elev_dry : mean DEM elevation of each GT class
      delta_elev = mean_elev_dry - mean_elev_water
          positive → water sits lower (expected for topographic priors)
          negative → water sits higher (levees, DSM artifacts, flat plains)
      elev_auc : AUROC of (-elevation) as predictor of GT water
          > 0.5 → water tends to be at lower elevations
          = 0.5 → no elevation discrimination
    """
    valid = _valid_mask(label, dem)
    water_mask = valid & (label == WATER_CLASS)
    dry_mask   = valid & (label == DRY_CLASS)

    result: dict[str, float] = {
        "n_water": int(water_mask.sum()),
        "n_dry":   int(dry_mask.sum()),
        "n_valid": int(valid.sum()),
    }

    if result["n_water"] == 0 or result["n_dry"] == 0:
        result.update({
            "mean_elev_water": float("nan"),
            "mean_elev_dry":   float("nan"),
            "median_elev_water": float("nan"),
            "median_elev_dry":   float("nan"),
            "delta_mean_elev":   float("nan"),
            "delta_median_elev": float("nan"),
            "elev_auc":          float("nan"),
        })
        return result

    h_water = dem[water_mask]
    h_dry   = dem[dry_mask]

    result["mean_elev_water"]   = float(h_water.mean())
    result["mean_elev_dry"]     = float(h_dry.mean())
    result["median_elev_water"] = float(np.median(h_water))
    result["median_elev_dry"]   = float(np.median(h_dry))
    result["delta_mean_elev"]   = result["mean_elev_dry"] - result["mean_elev_water"]
    result["delta_median_elev"] = result["median_elev_dry"] - result["median_elev_water"]

    if _HAS_SKLEARN:
        h_all    = dem[valid]
        lab_all  = (label[valid] == WATER_CLASS).astype(np.int32)
        if lab_all.sum() > 0 and (1 - lab_all).sum() > 0:
            result["elev_auc"] = float(roc_auc_score(lab_all, -h_all))
        else:
            result["elev_auc"] = float("nan")
    else:
        result["elev_auc"] = float("nan")

    return result


# ── Per-tile analysis ──────────────────────────────────────────────────────────

def analyse_tile(
    tile_id: str,
    split: str,
    shuffle_map: dict[str, str],
) -> dict[str, Any]:
    """
    Run all three metrics for a single tile, comparing real vs shuffled DEM.

    Returns a flat dict for CSV export.
    """
    row: dict[str, Any] = {
        "tile_id": tile_id,
        "split":   split,
        "dem_real_available":     False,
        "dem_shuffled_available": False,
    }

    label = load_label(tile_id)

    # ── Real DEM ──────────────────────────────────────────────────────────────
    dem_real = load_dem(split, tile_id)
    if dem_real is None:
        row["error"] = f"Real DEM missing for {split}/{tile_id}"
        return row

    row["dem_real_available"] = True

    real_cl  = compute_current_loss_violation(label, dem_real)
    real_d8  = compute_d8_violation(label, dem_real)
    real_el  = compute_elevation_stats(label, dem_real)

    for k, v in real_cl.items():
        row[f"real_{k}"] = v
    for k, v in real_d8.items():
        row[f"real_{k}"] = v
    for k, v in real_el.items():
        row[f"real_{k}"] = v

    # ── Shuffled DEM ──────────────────────────────────────────────────────────
    shuffled_tile = shuffle_map.get(tile_id)
    if shuffled_tile is not None:
        dem_shuf = load_dem(split, shuffled_tile)
        # The shuffled tile's DEM is always from the TRAIN split
        if dem_shuf is None:
            dem_shuf = load_dem("train", shuffled_tile)
        row["shuffled_tile_id"] = shuffled_tile
    else:
        # No shuffle map for this tile → no shuffled comparison
        dem_shuf = None
        row["shuffled_tile_id"] = None

    if dem_shuf is not None:
        row["dem_shuffled_available"] = True
        shuf_cl  = compute_current_loss_violation(label, dem_shuf)
        shuf_d8  = compute_d8_violation(label, dem_shuf)
        shuf_el  = compute_elevation_stats(label, dem_shuf)

        for k, v in shuf_cl.items():
            row[f"shuffled_{k}"] = v
        for k, v in shuf_d8.items():
            row[f"shuffled_{k}"] = v
        for k, v in shuf_el.items():
            row[f"shuffled_{k}"] = v

        # ── Deltas (real − shuffled). Violation: negative = real better ──────
        for key in (
            "current_loss_weighted_viol_frac",
            "current_loss_unweighted_viol_frac",
            "d8_weighted_viol_frac",
            "d8_unweighted_viol_frac",
            "delta_mean_elev",
            "elev_auc",
        ):
            r_val = row.get(f"real_{key}", float("nan"))
            s_val = row.get(f"shuffled_{key}", float("nan"))
            if not math.isnan(r_val) and not math.isnan(s_val):
                row[f"delta_{key}"] = r_val - s_val
            else:
                row[f"delta_{key}"] = float("nan")

    return row


# ── Synthetic shuffle for splits without a derangement map ────────────────────

def build_synthetic_shuffle(tile_ids: list[str]) -> dict[str, str]:
    """
    Cyclic shift: tile_i maps to tile_{i+1 mod N}.
    Used for val/test/bolivia where no Sattolo map exists.
    0 self-maps by construction when N > 1.
    """
    N = len(tile_ids)
    return {tile_ids[i]: tile_ids[(i + 1) % N] for i in range(N)}


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute mean/std and directional consistency across tiles."""
    metrics_viol = [
        "current_loss_weighted_viol_frac",
        "current_loss_unweighted_viol_frac",
        "d8_weighted_viol_frac",
        "d8_unweighted_viol_frac",
    ]
    metrics_elev = ["delta_mean_elev", "elev_auc"]

    agg: dict[str, Any] = {}
    n_tiles = len(rows)
    n_with_shuffle = sum(1 for r in rows if r.get("dem_shuffled_available"))

    agg["n_tiles"] = n_tiles
    agg["n_with_shuffle"] = n_with_shuffle

    for m in metrics_viol + metrics_elev:
        r_vals = [r[f"real_{m}"]  for r in rows if f"real_{m}"  in r and not math.isnan(r[f"real_{m}"])]
        s_vals = [r[f"shuffled_{m}"] for r in rows if f"shuffled_{m}" in r and not math.isnan(r[f"shuffled_{m}"])]
        d_vals = [r[f"delta_{m}"] for r in rows if f"delta_{m}" in r and not math.isnan(r[f"delta_{m}"])]

        def _stats(vs: list) -> dict:
            if not vs:
                return {"mean": float("nan"), "std": float("nan"), "n": 0}
            a = np.array(vs)
            return {"mean": float(a.mean()), "std": float(a.std(ddof=0 if len(a) < 2 else 1)), "n": len(a)}

        agg[f"real_{m}"]     = _stats(r_vals)
        agg[f"shuffled_{m}"] = _stats(s_vals)
        agg[f"delta_{m}"]    = _stats(d_vals)

        if d_vals:
            n_real_better = None
            if m in metrics_viol:
                # For violation metrics, lower = better → real better when delta < 0
                n_real_better = sum(1 for d in d_vals if d < 0)
            elif m == "elev_auc":
                # Higher AUC = better → real better when delta > 0
                n_real_better = sum(1 for d in d_vals if d > 0)
            elif m == "delta_mean_elev":
                # Larger delta (dry higher than water) = better → real better when delta > 0
                n_real_better = sum(1 for d in d_vals if d > 0)

            if n_real_better is not None:
                agg[f"pct_real_better_{m}"] = 100.0 * n_real_better / len(d_vals)

    return agg


# ── CSV / JSON / Markdown ─────────────────────────────────────────────────────

def save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    # Collect all keys maintaining insertion order
    all_keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    lines = [",".join(all_keys)]
    for row in rows:
        cells = []
        for k in all_keys:
            v = row.get(k, "")
            if isinstance(v, float):
                if math.isnan(v):
                    cells.append("")
                else:
                    cells.append(f"{v:.6f}")
            else:
                cells.append(str(v) if v is not None else "")
        lines.append(",".join(cells))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_json(obj: Any, path: Path) -> None:
    def _clean(x: Any) -> Any:
        if isinstance(x, float) and math.isnan(x):
            return None
        if isinstance(x, dict):
            return {k: _clean(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_clean(v) for v in x]
        return x
    path.write_text(json.dumps(_clean(obj), indent=2), encoding="utf-8")


def _fmt(v: Any) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def save_markdown(
    summary: dict[str, Any],
    per_split: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    lines: list[str] = []
    lines += [
        "# GT Topographic Consistency Diagnostic — N=100 seed0",
        "",
        "**No model, no GPU, no training. Read-only diagnostic.**",
        "",
        "## Purpose",
        "",
        "Before implementing a new D8-based physics loss, this diagnostic tests whether",
        "the *ground-truth* flood masks are more topographically consistent with the",
        "**real** aligned DEM than with a **shuffled** DEM.",
        "",
        "If the real DEM does not produce clearly lower violation rates than the shuffled",
        "DEM, then no DEM-only loss formulation is likely to produce a real > shuffled",
        "separation at training time — the issue would be in the data, not the loss.",
        "",
        "## Metrics",
        "",
        "| Metric | Description | Lower/Higher = Better |",
        "|--------|-------------|----------------------|",
        "| `current_loss_weighted_viol_frac` | Fraction of 4-neighbor elevation-weighted pairs where GT water sits higher than GT dry neighbor (mirrors current TopographicInconsistencyLoss formula) | **Lower** |",
        "| `d8_weighted_viol_frac` | Fraction of D8-downstream pairs (slope-weighted) where GT water is upstream of GT dry | **Lower** |",
        "| `delta_mean_elev` | mean(DEM at dry pixels) − mean(DEM at water pixels); positive = water is lower = expected | **Higher** |",
        "| `elev_auc` | AUROC of −elevation as predictor of GT water; 0.5 = no signal, 1.0 = perfect | **Higher** |",
        "",
        "## Interpretation",
        "",
        "- **Real DEM clearly better than shuffled** → DEM-specific signal exists in GT; implement D8 loss next.",
        "- **Real ≈ Shuffled** → DEM-only constraints carry no alignment-specific information at this resolution; report negative result.",
        "- **Shuffled better than real** → DEM alignment or formulation problem; investigate before implementing new loss.",
        "",
        "## Limitations",
        "",
        "1. **DEM resolution mismatch**: GLO-30 is 30 m; imagery is 10 m. Pixel-level constraints under a resampled DEM are partly meaningless.",
        "2. **DSM not DTM**: Copernicus GLO-30 includes buildings and canopy, not bare earth. Urban flood pixels on buildings appear higher than true ground.",
        "3. **Urban drainage not represented**: Water can be contained at high elevation by levees, walls, or culverts — DEM-only flow is wrong in dense urban areas.",
        "4. **Static topography vs. event flooding**: A DEM shows terrain, not water stage. A flood on a flat floodplain may not follow D8 at 30 m resolution.",
        "5. **Flat terrain instability**: On flat floodplains (characteristic of most large Sen1Floods11 flood events), D8 directions are ambiguous and slopes are near zero.",
        "",
    ]

    lines += ["## Results by Split", ""]
    for split_name, agg in per_split.items():
        n = agg.get("n_tiles", 0)
        n_shuf = agg.get("n_with_shuffle", 0)
        lines += [f"### Split: `{split_name}` ({n} tiles, {n_shuf} with shuffle)", ""]

        header = "| Metric | Real (mean±std) | Shuffled (mean±std) | Delta (real−shuffled) | % tiles real better |"
        sep    = "|--------|----------------|--------------------|-----------------------|---------------------|"
        lines += [header, sep]

        row_metrics = [
            ("current_loss_weighted_viol_frac", "Current-loss violation (↓)"),
            ("d8_weighted_viol_frac",           "D8 violation (↓)"),
            ("delta_mean_elev",                 "Elevation delta dry−water (↑)"),
            ("elev_auc",                        "Elevation AUC (↑)"),
        ]
        for key, label in row_metrics:
            r = agg.get(f"real_{key}", {})
            s = agg.get(f"shuffled_{key}", {})
            d = agg.get(f"delta_{key}", {})
            pct = agg.get(f"pct_real_better_{key}")
            r_str = f"{_fmt(r.get('mean'))} ± {_fmt(r.get('std'))}" if r else "N/A"
            s_str = f"{_fmt(s.get('mean'))} ± {_fmt(s.get('std'))}" if s else "N/A"
            d_str = f"{_fmt(d.get('mean'))} ± {_fmt(d.get('std'))}" if d else "N/A"
            p_str = f"{pct:.1f}%" if pct is not None else "N/A"
            lines.append(f"| {label} | {r_str} | {s_str} | {d_str} | {p_str} |")

        lines += [""]

    lines += [
        "## Overall Summary",
        "",
        f"- Total tiles processed: {summary.get('n_tiles_total', 'N/A')}",
        f"- Tiles with shuffle comparison: {summary.get('n_tiles_shuffled', 'N/A')}",
        f"- Splits: {summary.get('splits', 'N/A')}",
        "",
        "## Conclusion",
        "",
        "_Fill in after running the diagnostic._",
        "",
        "---",
        "*Generated by `experiments_cvpr/segman/diagnose_gt_topographic_consistency.py`.*",
    ]

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="GT topographic consistency diagnostic (CPU-only, no model)."
    )
    parser.add_argument(
        "--split", default="train",
        choices=["train", "val", "test", "bolivia", "all"],
        help="Which split(s) to process. 'all' = train+val+test+bolivia.",
    )
    parser.add_argument(
        "--manifest", default=None,
        help="Override manifest file path for the chosen split.",
    )
    parser.add_argument(
        "--dem-map", default=None,
        help="Override path to dem_tile_id_map JSON (default: N=100 seed0 shuffle map).",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory for output files (default: reports/ in repo root).",
    )
    parser.add_argument(
        "--max-tiles", type=int, default=None,
        help="Process at most N tiles per split (for quick debug).",
    )
    parser.add_argument(
        "--s0", type=float, default=1.0,
        help="D8 slope weight scale in meters (default: 1.0).",
    )
    parser.add_argument(
        "--synthetic-shuffle", action="store_true",
        help="For splits without a shuffle map, build a cyclic-shift synthetic shuffle.",
    )
    parser.add_argument(
        "--suffix", default="n100_seed0",
        help="Suffix for output filenames (default: n100_seed0).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load shuffle map
    dm_path = Path(args.dem_map) if args.dem_map else SHUFFLE_MAP_N100_S0
    shuffle_map_train: dict[str, str] = {}
    if dm_path.exists():
        raw = json.loads(dm_path.read_text(encoding="utf-8"))
        shuffle_map_train = raw.get("mapping", {})
        print(f"Loaded shuffle map: {len(shuffle_map_train)} entries from {dm_path.name}")
    else:
        print(f"WARNING: shuffle map not found at {dm_path}. No shuffle comparison for train.")

    splits = ["train", "val", "test", "bolivia"] if args.split == "all" else [args.split]

    all_rows: list[dict[str, Any]] = []
    per_split_rows: dict[str, list[dict[str, Any]]] = {}
    per_split_agg: dict[str, dict[str, Any]] = {}

    for split in splits:
        print(f"\n{'='*60}")
        print(f"  Split: {split}")
        print(f"{'='*60}")

        if args.manifest and len(splits) == 1:
            tiles = Path(args.manifest).read_text(encoding="utf-8").splitlines()
            tiles = [t.strip() for t in tiles if t.strip()]
        else:
            try:
                tiles = read_manifest(split)
            except FileNotFoundError as e:
                print(f"  SKIP: {e}")
                continue

        if args.max_tiles:
            tiles = tiles[: args.max_tiles]

        print(f"  Tiles: {len(tiles)}")

        # Build shuffle map for this split
        if split == "train":
            shuffle_map = shuffle_map_train
            print(f"  Shuffle map: Sattolo derangement ({len(shuffle_map)} entries)")
        elif args.synthetic_shuffle and len(tiles) > 1:
            shuffle_map = build_synthetic_shuffle(tiles)
            print(f"  Shuffle map: synthetic cyclic shift ({len(shuffle_map)} entries)")
        else:
            shuffle_map = {}
            print("  Shuffle map: none (no shuffled comparison for this split)")

        rows: list[dict[str, Any]] = []
        t0 = time.time()
        for idx, tile_id in enumerate(tiles):
            try:
                row = analyse_tile(tile_id, split, shuffle_map)
            except Exception as exc:
                row = {"tile_id": tile_id, "split": split, "error": str(exc)}
                print(f"  [{idx+1}/{len(tiles)}] ERROR {tile_id}: {exc}")
            else:
                has_shuf = row.get("dem_shuffled_available", False)
                viol_r   = row.get("real_d8_weighted_viol_frac", float("nan"))
                viol_s   = row.get("shuffled_d8_weighted_viol_frac", float("nan"))
                delta    = row.get("delta_d8_weighted_viol_frac", float("nan"))
                auc      = row.get("real_elev_auc", float("nan"))
                status   = "shuf=Y" if has_shuf else "shuf=N"
                print(
                    f"  [{idx+1:3d}/{len(tiles)}] {tile_id:<24} "
                    f"d8_viol real={_fmt(viol_r)} shuf={_fmt(viol_s)} "
                    f"delta={_fmt(delta)} AUC={_fmt(auc)} {status}"
                )
            rows.append(row)

        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s")

        per_split_rows[split] = rows
        agg = aggregate(rows)
        per_split_agg[split] = agg

        print(f"\n  === {split} aggregate ===")
        for key in ("current_loss_weighted_viol_frac", "d8_weighted_viol_frac",
                    "delta_mean_elev", "elev_auc"):
            r  = agg.get(f"real_{key}",     {})
            s  = agg.get(f"shuffled_{key}", {})
            d  = agg.get(f"delta_{key}",    {})
            pct = agg.get(f"pct_real_better_{key}")
            print(
                f"    {key}: real={_fmt(r.get('mean'))}+-{_fmt(r.get('std'))} "
                f"shuf={_fmt(s.get('mean'))}+-{_fmt(s.get('std'))} "
                f"delta={_fmt(d.get('mean'))}+-{_fmt(d.get('std'))} "
                f"real_better={f'{pct:.1f}%' if pct is not None else 'N/A'}"
            )

        all_rows.extend(rows)

    # ── Save outputs ──────────────────────────────────────────────────────────
    if not all_rows:
        print("\nNo tiles processed.")
        return 1

    splits_str = "_".join(splits)
    csv_path  = output_dir / f"gt_topographic_consistency_{args.suffix}.csv"
    json_path = output_dir / f"gt_topographic_consistency_{args.suffix}.json"

    docs_dir = REPO_ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    md_path = docs_dir / f"gt_topographic_consistency_{args.suffix}.md"

    summary = {
        "suffix":           args.suffix,
        "splits":           splits,
        "n_tiles_total":    len(all_rows),
        "n_tiles_shuffled": sum(1 for r in all_rows if r.get("dem_shuffled_available")),
        "shuffle_map_path": str(dm_path),
        "shuffle_map_entries": len(shuffle_map_train),
        "s0":               args.s0,
        "synthetic_shuffle_used": args.synthetic_shuffle,
        "per_split": per_split_agg,
    }

    save_csv(all_rows, csv_path)
    save_json({"summary": summary, "per_split": per_split_agg}, json_path)
    save_markdown(summary, per_split_agg, md_path)

    print(f"\n{'='*60}")
    print("  OUTPUT FILES")
    print(f"{'='*60}")
    print(f"  CSV    : {csv_path}")
    print(f"  JSON   : {json_path}")
    print(f"  MD     : {md_path}")

    # ── Final interpretation ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  INTERPRETATION GUIDE")
    print(f"{'='*60}")
    print("  For violation metrics (current_loss / d8): lower = better.")
    print("  Delta (real - shuffled) < 0 means real DEM shows FEWER violations => real is better.")
    print("  For AUC / elevation delta: real > shuffled is the expected signal direction.")
    print("")
    print("  Key criterion: Are real DEM violation rates clearly lower than shuffled?")
    print("  If yes => DEM-specific signal exists in GT => safe to implement D8 loss.")
    print("  If no  => No DEM-only loss will close the real/shuffled gap.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
