"""
CPU-only visual and physical analysis for the SegMAN-S N=50 multi-seed loss ablation.

Reads saved NPZ predictions (6 test images per run), computes per-image metrics,
loads Sentinel-2 RGB + DEM for visualization, generates comparison panels and
topographic violation maps.

Usage:
    python experiments_cvpr/segman/analyze_segman_n50_visuals.py \
        --exp-root E:/flood_research/experiments/segman \
        --output-dir reports/figures/segman_n50

Outputs (under --output-dir):
    panels/panel_{i:03d}_{tile_id}.png   -- 4-condition comparison (per image)
    topo_violations/topo_viol_{cond}_{i:03d}.png  -- topo violation maps
    tables/per_image_metrics.csv         -- full per-image metric table
    tables/delta_summary.csv             -- paired deltas
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

SEEDS = [0, 1, 2, 3, 42]
CONDITIONS = ["ce", "dice_ce", "dice_ce_topo", "dice_ce_topo_dem_shuffled"]
CONDITION_LABELS = {
    "ce":                        "CE",
    "dice_ce":                   "Dice+CE",
    "dice_ce_topo":              "Dice+CE+Topo",
    "dice_ce_topo_dem_shuffled": "Dice+CE+Topo+Shuf",
}
CONDITION_COLORS = {
    "ce":                        "#94a3b8",   # slate
    "dice_ce":                   "#38bdf8",   # sky
    "dice_ce_topo":              "#22d3ee",   # cyan
    "dice_ce_topo_dem_shuffled": "#fbbf24",   # amber
}

N_SAMPLES = 6  # NPZ files saved per run (pred_000 .. pred_005)

TEST_MANIFEST = Path(
    "E:/flood_research/experiments/terramind_baseline/runs/"
    "step5e_tiny_unetdecoder_baseline/manifests/flood_test_step5e_filtered.txt"
)
S2_ROOT = Path(
    "E:/flood_research/data/raw/sen1floods11/v1.1/data/flood_events/HandLabeled/S2Hand"
)
DEM_ROOT = Path("E:/flood_research/data/derived/sen1floods11_topography/dem_aligned")
DEM_PATTERN = "test_{tile_id}_copernicus_glo30_dem_aligned.tif"

# S2 RGB band indices (0-based from the 13-band stack: B04=3, B03=2, B02=1)
S2_RGB_IDX = (3, 2, 1)

BG_COLOR = "#0b0f1a"


# ---------------------------------------------------------------------------
# Helpers: tag resolution
# ---------------------------------------------------------------------------

def run_tag_for(condition: str, seed: int) -> str:
    if condition == "ce" and seed == 0:
        return "segman_ce_seed0_clean"
    return f"segman_{condition}_seed{seed}"


def npz_path(exp_root: Path, condition: str, seed: int, idx: int) -> Path:
    tag = run_tag_for(condition, seed)
    return exp_root / "runs" / tag / "predictions" / "test" / f"pred_{idx:03d}.npz"


# ---------------------------------------------------------------------------
# Helpers: metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Compute binary segmentation metrics, ignoring target == -1."""
    valid = target >= 0
    if valid.sum() == 0:
        return {k: float("nan") for k in
                ["iou_water", "f1_water", "precision_water", "recall_water",
                 "accuracy", "pred_water_ratio", "gt_water_ratio",
                 "tp", "fp", "tn", "fn"]}
    p = pred[valid].astype(bool)
    g = target[valid].astype(bool)
    tp = float((p & g).sum())
    fp = float((p & ~g).sum())
    tn = float((~p & ~g).sum())
    fn = float((~p & g).sum())
    eps = 1e-7
    iou_w     = tp / (tp + fp + fn + eps)
    prec_w    = tp / (tp + fp + eps)
    rec_w     = tp / (tp + fn + eps)
    f1_w      = 2 * prec_w * rec_w / (prec_w + rec_w + eps)
    acc       = (tp + tn) / (tp + fp + tn + fn + eps)
    pred_wr   = float(p.sum()) / float(valid.sum())
    gt_wr     = float(g.sum()) / float(valid.sum())
    return {
        "iou_water": iou_w, "f1_water": f1_w,
        "precision_water": prec_w, "recall_water": rec_w,
        "accuracy": acc,
        "pred_water_ratio": pred_wr, "gt_water_ratio": gt_wr,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


# ---------------------------------------------------------------------------
# Helpers: topo violation map (CPU, hard mask)
# ---------------------------------------------------------------------------

def compute_topo_violation_map(
    pred: np.ndarray,    # (H, W) uint8 0/1
    dem: np.ndarray,     # (H, W) float32
    margin: float = 0.0,
) -> np.ndarray:
    """
    Diagnostic topo violation map (not the differentiable training loss).
    A pixel is flagged as a violation source if:
      - it is predicted water
      - any of its 4 neighbours is predicted dry
      - its elevation > neighbour elevation + margin

    Returns (H, W) bool violation_map.
    """
    H, W = pred.shape
    water = pred.astype(bool)
    viol = np.zeros((H, W), dtype=bool)
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        neigh_y = np.clip(np.arange(H)[:, None] + dy, 0, H - 1)
        neigh_x = np.clip(np.arange(W)[None, :] + dx, 0, W - 1)
        neigh_water = water[neigh_y, neigh_x]
        neigh_elev  = dem[neigh_y, neigh_x]
        # violation: current is water, neighbour is dry, current elev > neigh elev + margin
        mask = water & ~neigh_water & (dem > neigh_elev + margin)
        # mask out border pixels that clamped to same index
        if dy == -1: mask[0, :]  = False
        if dy ==  1: mask[-1, :] = False
        if dx == -1: mask[:, 0]  = False
        if dx ==  1: mask[:, -1] = False
        viol |= mask
    return viol


def viol_fraction(viol_map: np.ndarray, valid_mask: np.ndarray | None = None) -> float:
    if valid_mask is not None:
        denom = float(valid_mask.sum())
    else:
        denom = float(viol_map.size)
    return float(viol_map.sum()) / (denom + 1e-9)


# ---------------------------------------------------------------------------
# Helpers: image loading
# ---------------------------------------------------------------------------

def load_s2_rgb(tile_id: str) -> np.ndarray | None:
    """Load S2 RGB (H, W, 3) uint8 for visualization, percentile-stretched."""
    try:
        import rasterio
        p = S2_ROOT / f"{tile_id}_S2Hand.tif"
        if not p.exists():
            return None
        with rasterio.open(p) as src:
            arr = src.read([S2_RGB_IDX[0] + 1,
                            S2_RGB_IDX[1] + 1,
                            S2_RGB_IDX[2] + 1]).astype(np.float32)
        # percentile stretch
        out = np.zeros_like(arr)
        for i in range(3):
            lo, hi = np.percentile(arr[i], 2), np.percentile(arr[i], 98)
            if hi > lo:
                out[i] = np.clip((arr[i] - lo) / (hi - lo), 0, 1)
            else:
                out[i] = 0.0
        return (out.transpose(1, 2, 0) * 255).astype(np.uint8)
    except Exception:
        return None


def load_dem(tile_id: str) -> np.ndarray | None:
    """Load DEM (H, W) float32."""
    try:
        import rasterio
        p = DEM_ROOT / DEM_PATTERN.format(tile_id=tile_id)
        if not p.exists():
            return None
        with rasterio.open(p) as src:
            return src.read(1).astype(np.float32)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core: load all predictions
# ---------------------------------------------------------------------------

def load_all_preds(exp_root: Path) -> dict[str, Any]:
    """
    Returns:
        preds[cond][seed][idx] = (pred 512x512 uint8, target 512x512 int16)
        missing: list of (cond, seed, idx) that could not be loaded
    """
    preds: dict[str, dict[int, dict[int, tuple]]] = {}
    missing = []
    for cond in CONDITIONS:
        preds[cond] = {}
        for seed in SEEDS:
            preds[cond][seed] = {}
            for idx in range(N_SAMPLES):
                p = npz_path(exp_root, cond, seed, idx)
                if not p.exists():
                    missing.append((cond, seed, idx))
                    continue
                d = np.load(p, allow_pickle=True)
                preds[cond][seed][idx] = (d["pred"], d["target"])
    return {"preds": preds, "missing": missing}


# ---------------------------------------------------------------------------
# Core: per-image metrics
# ---------------------------------------------------------------------------

def build_per_image_metrics(preds_data: dict) -> list[dict]:
    """
    For each (cond, seed, img_idx): compute metrics.
    Returns list of row dicts.
    """
    preds = preds_data["preds"]
    rows = []
    for cond in CONDITIONS:
        for seed in SEEDS:
            for idx in range(N_SAMPLES):
                entry = preds[cond][seed].get(idx)
                if entry is None:
                    continue
                pred, target = entry
                m = compute_metrics(pred, target)
                row = {
                    "condition": cond,
                    "condition_label": CONDITION_LABELS[cond],
                    "seed": seed,
                    "img_idx": idx,
                    **m,
                }
                rows.append(row)
    return rows


def aggregate_over_seeds(rows: list[dict]) -> dict[str, dict[int, dict]]:
    """
    Returns agg[cond][idx] = {metric: (mean, std), ...}
    """
    agg: dict[str, dict[int, dict]] = {c: {i: {} for i in range(N_SAMPLES)} for c in CONDITIONS}
    metrics_to_agg = ["iou_water", "f1_water", "precision_water", "recall_water",
                      "accuracy", "pred_water_ratio", "gt_water_ratio"]
    for cond in CONDITIONS:
        for idx in range(N_SAMPLES):
            seed_rows = [r for r in rows if r["condition"] == cond and r["img_idx"] == idx]
            for m in metrics_to_agg:
                vals = [r[m] for r in seed_rows if r[m] is not None and not math.isnan(r[m])]
                if not vals:
                    agg[cond][idx][m + "_mean"] = float("nan")
                    agg[cond][idx][m + "_std"]  = float("nan")
                    continue
                mu = sum(vals) / len(vals)
                sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / max(len(vals) - 1, 1)) if len(vals) > 1 else 0.0
                agg[cond][idx][m + "_mean"] = mu
                agg[cond][idx][m + "_std"]  = sd
    return agg


# ---------------------------------------------------------------------------
# Visualisation: panel per image
# ---------------------------------------------------------------------------

PANEL_COND_ORDER = ["ce", "dice_ce", "dice_ce_topo", "dice_ce_topo_dem_shuffled"]


def _mask_rgba(pred: np.ndarray, target: np.ndarray, kind: str = "pred") -> np.ndarray:
    """Return RGBA overlay for a prediction or GT mask."""
    H, W = pred.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    valid = target >= 0
    if kind == "gt":
        water = target == 1
        dry   = (target == 0) & valid
        rgba[water]  = [0, 150, 255, 200]   # blue = GT water
        rgba[dry]    = [60, 70, 90, 180]    # dark = GT dry
        rgba[~valid] = [30, 30, 30, 80]
    else:
        water = (pred == 1) & valid
        dry   = (pred == 0) & valid
        rgba[water]  = [34, 211, 238, 220]  # cyan = predicted water
        rgba[dry]    = [36, 48, 66, 180]    # dark = predicted dry
        rgba[~valid] = [30, 30, 30, 80]
    return rgba


def make_panel(
    idx: int,
    tile_id: str,
    preds_data: dict,
    agg: dict,
    dem: np.ndarray | None,
    s2_rgb: np.ndarray | None,
    output_path: Path,
) -> None:
    preds = preds_data["preds"]

    # pick seed0 as representative
    seed = 0

    n_cols = 7 if s2_rgb is not None else 6
    fig_w = 3.5 * n_cols
    fig_h = 4.5

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#0d1117")
    gs = gridspec.GridSpec(1, n_cols, figure=fig, wspace=0.05, left=0.02, right=0.98,
                           top=0.84, bottom=0.02)

    axes = [fig.add_subplot(gs[0, c]) for c in range(n_cols)]
    col = 0

    # ── S2 RGB ──────────────────────────────────────────────────────────────
    if s2_rgb is not None:
        ax = axes[col]; col += 1
        ax.imshow(s2_rgb)
        ax.set_title("S2 RGB", color="white", fontsize=9, pad=3)
        ax.axis("off")

    # ── Ground truth ─────────────────────────────────────────────────────────
    entry0 = preds["ce"][seed].get(idx)
    if entry0:
        _, target = entry0
    else:
        target = np.zeros((512, 512), dtype=np.int16)
    ax = axes[col]; col += 1
    ax.imshow(_mask_rgba(target, target, kind="gt"))
    ax.set_title("Ground Truth", color="white", fontsize=9, pad=3)
    ax.axis("off")

    # ── Per-condition predictions ─────────────────────────────────────────────
    for cond in PANEL_COND_ORDER:
        ax = axes[col]; col += 1
        entry = preds[cond][seed].get(idx)
        if entry is None:
            ax.set_facecolor("#111")
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            ax.axis("off")
            continue
        pred, tgt = entry
        ax.imshow(_mask_rgba(pred, tgt, kind="pred"))
        mu_iou = agg[cond][idx].get("iou_water_mean", float("nan"))
        sd_iou = agg[cond][idx].get("iou_water_std", 0.0)
        clr = CONDITION_COLORS[cond]
        title = f"{CONDITION_LABELS[cond]}\nIoU_w {mu_iou:.3f}±{sd_iou:.3f}"
        ax.set_title(title, color=clr, fontsize=8.5, pad=3)
        ax.axis("off")

    # ── DEM ──────────────────────────────────────────────────────────────────
    if dem is not None and col < len(axes):
        ax = axes[col]; col += 1
        im = ax.imshow(dem, cmap="terrain")
        ax.set_title("DEM (m)", color="#d4a373", fontsize=9, pad=3)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(
            labelcolor="white", labelsize=7)

    # ── Suptitle ─────────────────────────────────────────────────────────────
    fig.suptitle(
        f"Image {idx:03d} — {tile_id}  (seed0 representative; metrics mean±std over 5 seeds)",
        color="white", fontsize=10, y=0.97,
    )

    # Legend
    legend_patches = [
        mpatches.Patch(color="#0096ff", label="GT water"),
        mpatches.Patch(color="#22d3ee", label="Predicted water"),
        mpatches.Patch(color="#243042", label="Predicted dry"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3,
               framealpha=0.3, fontsize=8, labelcolor="white",
               bbox_to_anchor=(0.5, -0.01))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# Visualisation: topo violation comparison (4 conditions side by side)
# ---------------------------------------------------------------------------

def make_topo_panel(
    idx: int,
    tile_id: str,
    preds_data: dict,
    dem: np.ndarray,
    output_path: Path,
) -> dict[str, float]:
    """Create topo violation comparison panel for one image. Returns violation fractions."""
    preds = preds_data["preds"]
    seed = 0
    fracs: dict[str, float] = {}

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5), facecolor="#0d1117")
    for ax in axes:
        ax.set_facecolor("#111")
        ax.axis("off")

    for i, cond in enumerate(PANEL_COND_ORDER):
        ax = axes[i]
        entry = preds[cond][seed].get(idx)
        if entry is None:
            fracs[cond] = float("nan")
            continue
        pred, target = entry
        valid_mask = target >= 0

        viol = compute_topo_violation_map(pred, dem, margin=0.0)
        frac = viol_fraction(viol, valid_mask)
        fracs[cond] = frac

        # Render: background = DEM (greyscale), overlay violations in red
        dem_disp = (dem - dem.min()) / (dem.max() - dem.min() + 1e-7)
        rgb = np.stack([dem_disp] * 3, axis=-1)
        water_mask = pred.astype(bool)
        rgb[water_mask & ~viol] = [0.0, 0.6, 0.9]   # cyan = clean water
        rgb[viol] = [1.0, 0.15, 0.15]                # red = violation
        rgb[~valid_mask] *= 0.3
        ax.imshow(rgb, vmin=0, vmax=1)
        clr = CONDITION_COLORS[cond]
        ax.set_title(
            f"{CONDITION_LABELS[cond]}\nviol={frac:.4f}",
            color=clr, fontsize=9, pad=4,
        )

    fig.suptitle(
        f"Topo Violations — {tile_id}  (seed0 pred; red=water higher than dry neighbour)",
        color="white", fontsize=10, y=1.01,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return fracs


# ---------------------------------------------------------------------------
# Summary bar chart
# ---------------------------------------------------------------------------

def make_summary_chart(agg: dict, tile_ids: list[str], output_path: Path) -> None:
    """Bar chart of mean IoU_water per condition per image."""
    n_imgs = len(tile_ids)
    x = np.arange(n_imgs)
    width = 0.2
    offsets = [-1.5, -0.5, 0.5, 1.5]

    fig, axes = plt.subplots(2, 1, figsize=(max(10, 2 * n_imgs), 8),
                             facecolor="#0d1117")
    for ax in axes:
        ax.set_facecolor("#111")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#334155")
        ax.yaxis.label.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.title.set_color("white")

    ax0, ax1 = axes

    for i, cond in enumerate(PANEL_COND_ORDER):
        mus = [agg[cond][idx].get("iou_water_mean", float("nan")) for idx in range(n_imgs)]
        sds = [agg[cond][idx].get("iou_water_std", 0.0) for idx in range(n_imgs)]
        clr = CONDITION_COLORS[cond]
        ax0.bar(x + offsets[i] * width, mus, width, yerr=sds,
                label=CONDITION_LABELS[cond], color=clr, alpha=0.85,
                error_kw=dict(ecolor="white", capsize=3, lw=1))
    ax0.set_xticks(x)
    ax0.set_xticklabels([f"{i}\n{t[:12]}" for i, t in enumerate(tile_ids)],
                        fontsize=7.5, color="white")
    ax0.set_ylabel("IoU_water (mean ± std, 5 seeds)")
    ax0.set_title("Per-Image IoU_water — 4 Loss Conditions")
    ax0.legend(framealpha=0.3, labelcolor="white", fontsize=8)
    ax0.set_ylim(0, 1.05)

    # Δ chart: Topo_real − Dice+CE and Shuffled − Topo_real
    d_topo = np.array([
        agg["dice_ce_topo"][i].get("iou_water_mean", float("nan")) -
        agg["dice_ce"][i].get("iou_water_mean", float("nan"))
        for i in range(n_imgs)
    ])
    d_shuf = np.array([
        agg["dice_ce_topo_dem_shuffled"][i].get("iou_water_mean", float("nan")) -
        agg["dice_ce_topo"][i].get("iou_water_mean", float("nan"))
        for i in range(n_imgs)
    ])
    ax1.bar(x - 0.2, d_topo, 0.35, label="Topo − Dice+CE", color="#22d3ee", alpha=0.85)
    ax1.bar(x + 0.2, d_shuf, 0.35, label="Shuffled − Topo", color="#fbbf24", alpha=0.85)
    ax1.axhline(0, color="white", lw=0.8, ls="--")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{i}\n{t[:12]}" for i, t in enumerate(tile_ids)],
                        fontsize=7.5, color="white")
    ax1.set_ylabel("ΔIoU_water (mean over 5 seeds)")
    ax1.set_title("Paired Deltas per Image")
    ax1.legend(framealpha=0.3, labelcolor="white", fontsize=8)

    plt.tight_layout(pad=1.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp-root",   type=Path,
                    default=Path("E:/flood_research/experiments/segman"))
    ap.add_argument("--output-dir", type=Path,
                    default=REPO_ROOT / "reports" / "figures" / "segman_n50")
    ap.add_argument("--no-panels",   action="store_true",
                    help="Skip per-image panel generation")
    ap.add_argument("--no-topo",     action="store_true",
                    help="Skip topo violation maps")
    args = ap.parse_args(argv)

    out = args.output_dir
    panels_dir  = out / "panels"
    topo_dir    = out / "topo_violations"
    tables_dir  = out / "tables"
    for d in (panels_dir, topo_dir, tables_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── Read test manifest ──────────────────────────────────────────────────
    if not TEST_MANIFEST.exists():
        print(f"WARNING: test manifest not found: {TEST_MANIFEST}", file=sys.stderr)
        tile_ids = [f"img_{i:03d}" for i in range(N_SAMPLES)]
    else:
        all_tiles = TEST_MANIFEST.read_text().strip().splitlines()
        tile_ids  = all_tiles[:N_SAMPLES]
    print(f"Test images (first {N_SAMPLES}): {tile_ids}")

    # ── Load predictions ────────────────────────────────────────────────────
    print("\nLoading NPZ predictions ...")
    preds_data = load_all_preds(args.exp_root)
    if preds_data["missing"]:
        print(f"  WARNING: {len(preds_data['missing'])} missing NPZ files:")
        for m in preds_data["missing"]:
            print(f"    {m}")
    else:
        print(f"  Loaded {len(CONDITIONS) * len(SEEDS) * N_SAMPLES} NPZ files (no missing).")

    # ── Per-image metrics ───────────────────────────────────────────────────
    print("\nComputing per-image metrics ...")
    rows = build_per_image_metrics(preds_data)
    agg  = aggregate_over_seeds(rows)

    # Print summary table
    print(f"\n{'img':>5} {'tile':>18}  " +
          "  ".join(f"{'IoU_w ':>8}({CONDITION_LABELS[c][:10]})" for c in CONDITIONS))
    for idx, tile_id in enumerate(tile_ids):
        cells = "  ".join(
            f"{agg[c][idx].get('iou_water_mean', float('nan')):>12.4f}"
            f"±{agg[c][idx].get('iou_water_std', 0.0):.4f}"
            for c in CONDITIONS
        )
        print(f"  {idx:3d}  {tile_id:>18}  {cells}")

    # ── Write per-image CSV ─────────────────────────────────────────────────
    csv_path = tables_dir / "per_image_metrics.csv"
    metric_keys = ["iou_water", "f1_water", "precision_water", "recall_water",
                   "accuracy", "pred_water_ratio", "gt_water_ratio"]
    fieldnames = ["condition", "condition_label", "seed", "img_idx",
                  "tile_id"] + metric_keys
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            r2 = dict(r)
            r2["tile_id"] = tile_ids[r["img_idx"]] if r["img_idx"] < len(tile_ids) else ""
            w.writerow({k: ("" if (v is None or (isinstance(v, float) and math.isnan(v))) else v)
                        for k, v in r2.items() if k in fieldnames})
    print(f"\nWrote per-image CSV: {csv_path}")

    # ── Write delta summary CSV ─────────────────────────────────────────────
    delta_path = tables_dir / "delta_summary.csv"
    delta_rows = []
    for idx, tile_id in enumerate(tile_ids):
        base_ce  = agg["ce"][idx].get("iou_water_mean", float("nan"))
        base_dce = agg["dice_ce"][idx].get("iou_water_mean", float("nan"))
        base_top = agg["dice_ce_topo"][idx].get("iou_water_mean", float("nan"))
        base_shu = agg["dice_ce_topo_dem_shuffled"][idx].get("iou_water_mean", float("nan"))
        delta_rows.append({
            "img_idx":        idx,
            "tile_id":        tile_id,
            "iou_water_ce":   base_ce,
            "iou_water_dice_ce":  base_dce,
            "iou_water_topo":     base_top,
            "iou_water_shuffled": base_shu,
            "delta_dice_ce_minus_ce":     base_dce - base_ce,
            "delta_topo_minus_dice_ce":   base_top - base_dce,
            "delta_shuffled_minus_topo":  base_shu - base_top,
            "delta_shuffled_minus_dce":   base_shu - base_dce,
        })
    with delta_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(delta_rows[0].keys()))
        w.writeheader()
        for r in delta_rows:
            w.writerow({k: ("" if isinstance(v, float) and math.isnan(v) else v)
                        for k, v in r.items()})
    print(f"Wrote delta summary CSV: {delta_path}")

    # ── Summary bar chart ───────────────────────────────────────────────────
    chart_path = out / "summary_iou_water_per_image.png"
    print("\nGenerating summary bar chart ...")
    make_summary_chart(agg, tile_ids, chart_path)
    print(f"Wrote: {chart_path}")

    # ── Per-image panels ────────────────────────────────────────────────────
    topo_viol_all: dict[int, dict[str, float]] = {}
    if not args.no_panels or not args.no_topo:
        for idx, tile_id in enumerate(tile_ids):
            print(f"\n--- Image {idx:03d}: {tile_id} ---")
            s2_rgb = load_s2_rgb(tile_id)
            dem    = load_dem(tile_id)
            print(f"  S2 RGB: {'OK' if s2_rgb is not None else 'missing'}")
            print(f"  DEM: {'OK' if dem is not None else 'missing'}")

            if not args.no_panels:
                panel_path = panels_dir / f"panel_{idx:03d}_{tile_id}.png"
                make_panel(idx, tile_id, preds_data, agg, dem, s2_rgb, panel_path)
                print(f"  Panel: {panel_path}")

            if not args.no_topo and dem is not None:
                topo_path = topo_dir / f"topo_viol_{idx:03d}_{tile_id}.png"
                fracs = make_topo_panel(idx, tile_id, preds_data, dem, topo_path)
                topo_viol_all[idx] = fracs
                print(f"  Topo violations: " +
                      "  ".join(f"{CONDITION_LABELS[c]}={fracs.get(c, float('nan')):.4f}"
                                for c in CONDITIONS))
            elif dem is None and not args.no_topo:
                print(f"  Topo map: skipped (no DEM)")

    # ── Write topo violation JSON ───────────────────────────────────────────
    if topo_viol_all:
        topo_json = tables_dir / "topo_violation_per_image.json"
        payload = {
            str(idx): {
                "tile_id": tile_ids[idx],
                "violations": {c: v for c, v in vd.items()},
            }
            for idx, vd in topo_viol_all.items()
        }
        topo_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote topo violation JSON: {topo_json}")

    # ── Final summary ───────────────────────────────────────────────────────
    print("\n=== Analysis complete ===")
    print(f"  Per-image CSV:   {tables_dir / 'per_image_metrics.csv'}")
    print(f"  Delta CSV:       {tables_dir / 'delta_summary.csv'}")
    print(f"  Panels:          {panels_dir}  ({N_SAMPLES} images)")
    print(f"  Topo maps:       {topo_dir}")
    print(f"  Summary chart:   {chart_path}")

    # Print overall mean deltas
    print("\n=== Mean per-image deltas (IoU_water, mean over 6 samples × 5 seeds) ===")
    for dr in delta_rows:
        for k in ["delta_dice_ce_minus_ce", "delta_topo_minus_dice_ce",
                  "delta_shuffled_minus_topo", "delta_shuffled_minus_dce"]:
            pass
    vals_dce_ce  = [dr["delta_dice_ce_minus_ce"]    for dr in delta_rows if not math.isnan(dr["delta_dice_ce_minus_ce"])]
    vals_topo    = [dr["delta_topo_minus_dice_ce"]   for dr in delta_rows if not math.isnan(dr["delta_topo_minus_dice_ce"])]
    vals_shuf    = [dr["delta_shuffled_minus_topo"]  for dr in delta_rows if not math.isnan(dr["delta_shuffled_minus_topo"])]
    def _fmt(vs):
        if not vs: return "N/A"
        mu = sum(vs) / len(vs)
        sd = math.sqrt(sum((v-mu)**2 for v in vs) / max(len(vs)-1, 1))
        return f"{mu:+.4f} ± {sd:.4f}"
    print(f"  Dice+CE - CE:         {_fmt(vals_dce_ce)}")
    print(f"  Topo - Dice+CE:       {_fmt(vals_topo)}")
    print(f"  Shuffled - Topo:      {_fmt(vals_shuf)}")
    print()


if __name__ == "__main__":
    main()
