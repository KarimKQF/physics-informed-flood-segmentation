"""
Elevation AUC Diagnostic on Model Predictions
==============================================
Read-only diagnostic. Loads a trained SegMAN checkpoint and computes,
for each split (val / test / bolivia):

  - AUC(-elevation -> GT water)           [pooled + per-tile mean/std]
  - AUC(-elevation -> pred binary water)  [pooled + per-tile mean/std]
  - AUC(p_water -> GT water)              [model discrimination quality]
  - Elevation gap: mean(h_dry) - mean(h_water) for GT and pred
  - P(water | elevation-rank bin) curves, B bins  [pi_GT and pi_pred]
  - Shuffled-DEM sanity check (intra-tile random permutation, expect ~0.5)

No training, no loss changes, no config edits, no writes to run dirs.
Outputs go to reports/ and docs/ only.

Usage (smoke test):
    python experiments_cvpr/segman/diagnose_elevation_auc_predictions.py \\
        --config configs/segman/d8_n100_seed0/n100_seed0_dice_ce_d8_lambda100p0.yaml \\
        --split val --max-tiles 5

Full run (one split):
    python experiments_cvpr/segman/diagnose_elevation_auc_predictions.py \\
        --config configs/segman/d8_n100_seed0/n100_seed0_dice_ce_d8_lambda100p0.yaml \\
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
import torch
import yaml

# ── repository paths (mirrors train_segman.py) ───────────────────────────────
SEGMAN_ROOT = Path(__file__).resolve().parent
REPO_ROOT   = SEGMAN_ROOT.parents[1]
for _p in (str(SEGMAN_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from step6c_v3_train import TopographyDataModule          # noqa: E402
from model.segman_model import build_segman               # noqa: E402

# Reuse GT diagnostic helpers — DO NOT MODIFY that file
from diagnose_gt_topographic_consistency import (          # noqa: E402
    IGNORE_INDEX,
    WATER_CLASS,
    _valid_mask,
)

try:
    from sklearn.metrics import roc_auc_score
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
    print("WARNING: sklearn not available; AUC metrics will be NaN.")


# ── InputAssembler (copied from train_segman.py — do not import from there
#    to avoid side-effects of that module's top-level code) ─────────────────
class InputAssembler:
    """Concatenate per-modality image dict into a single normalized tensor."""

    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        data_args = config["data"]["init_args"]
        self.modalities = list(data_args["modalities"])
        means = data_args["means"]
        stds  = data_args["stds"]
        self.mean = {
            m: torch.tensor(means[m], dtype=torch.float32, device=device).view(1, -1, 1, 1)
            for m in self.modalities
        }
        self.std = {
            m: torch.tensor(stds[m], dtype=torch.float32, device=device).view(1, -1, 1, 1)
            for m in self.modalities
        }

    def __call__(self, image: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = []
        for m in self.modalities:
            x = image[m].float()
            parts.append((x - self.mean[m]) / self.std[m])
        return torch.cat(parts, dim=1)


def _get_target(batch: dict[str, Any]) -> torch.Tensor:
    mask = batch["mask"]
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]
    return mask.long()


def _get_topo(batch: dict[str, Any]) -> torch.Tensor | None:
    topo = batch.get("topography")
    if topo is None:
        return None
    if topo.ndim == 4 and topo.shape[1] == 1:
        topo = topo[:, 0]
    return topo.float()


# ── Per-tile metric computation ───────────────────────────────────────────────

def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if not _HAS_SKLEARN:
        return float("nan")
    try:
        if y_true.sum() == 0 or (1 - y_true).sum() == 0:
            return float("nan")
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def compute_tile_stats(
    h_valid:   np.ndarray,  # (N,) DEM values at valid pixels
    gt_valid:  np.ndarray,  # (N,) GT binary labels (0/1)
    pred_valid: np.ndarray, # (N,) predicted binary (0/1)
    p_valid:   np.ndarray,  # (N,) predicted probability of water
    n_bins:    int = 10,
    rng_seed:  int = 42,
) -> dict[str, Any]:
    """Compute per-tile elevation AUC and pi-curve stats."""
    N = len(h_valid)
    result: dict[str, Any] = {
        "n_valid":      N,
        "n_gt_water":   int(gt_valid.sum()),
        "n_pred_water": int(pred_valid.sum()),
    }

    neg_h = -h_valid  # score: larger = lower elevation = expected water

    # ── AUC metrics ──────────────────────────────────────────────────────────
    result["auc_elev_gt"]          = _safe_auc(gt_valid,   neg_h)
    result["auc_elev_pred_binary"] = _safe_auc(pred_valid, neg_h)
    result["auc_model_vs_gt"]      = _safe_auc(gt_valid,   p_valid)  # model discrimination

    # Shuffled-DEM sanity check (intra-tile random permutation)
    rng = np.random.default_rng(rng_seed)
    h_shuf = rng.permutation(neg_h)
    result["auc_shuffled_gt"]   = _safe_auc(gt_valid,   h_shuf)
    result["auc_shuffled_pred"] = _safe_auc(pred_valid, h_shuf)

    # ── Elevation gaps ────────────────────────────────────────────────────────
    gt_w  = gt_valid == 1
    gt_d  = gt_valid == 0
    pr_w  = pred_valid == 1
    pr_d  = pred_valid == 0

    def _mean_h(mask: np.ndarray) -> float:
        return float(h_valid[mask].mean()) if mask.sum() > 0 else float("nan")

    mhgw = _mean_h(gt_w)
    mhgd = _mean_h(gt_d)
    mhpw = _mean_h(pr_w)
    mhpd = _mean_h(pr_d)

    result["mean_h_gt_water"]   = mhgw
    result["mean_h_gt_dry"]     = mhgd
    result["elev_gap_gt"]       = mhgd - mhgw if math.isfinite(mhgd) and math.isfinite(mhgw) else float("nan")
    result["mean_h_pred_water"] = mhpw
    result["mean_h_pred_dry"]   = mhpd
    result["elev_gap_pred"]     = mhpd - mhpw if math.isfinite(mhpd) and math.isfinite(mhpw) else float("nan")

    # ── Pi curves: P(water | elevation-rank bin) ──────────────────────────────
    # bin 0 = lowest elevation, bin B-1 = highest elevation
    ranks   = np.argsort(np.argsort(h_valid))            # rank 0..N-1 (low→high elev)
    bin_idx = np.minimum(ranks * n_bins // max(N, 1), n_bins - 1).astype(np.int32)

    pi_gt   = []
    pi_pred = []
    n_bin   = []
    for b in range(n_bins):
        mb = bin_idx == b
        nb = int(mb.sum())
        n_bin.append(nb)
        pi_gt.append(  float(gt_valid[mb].mean())   if nb > 0 else float("nan"))
        pi_pred.append(float(pred_valid[mb].mean())  if nb > 0 else float("nan"))

    result["pi_gt"]   = pi_gt
    result["pi_pred"] = pi_pred
    result["n_bin"]   = n_bin

    return result


# ── Inference loop for one split ──────────────────────────────────────────────

def run_inference_split(
    model:     torch.nn.Module,
    assembler: InputAssembler,
    loader,
    device:    torch.device,
    n_bins:    int,
    max_tiles: int | None,
) -> list[dict[str, Any]]:
    """
    Run model inference on one split. Returns list of per-tile stat dicts.
    Assumes loader yields one tile per iteration (batch_size=1).
    """
    model.eval()
    tile_stats: list[dict[str, Any]] = []
    tile_idx = 0

    with torch.no_grad():
        for raw_batch in loader:
            if max_tiles is not None and tile_idx >= max_tiles:
                break

            # Move to device
            image = {k: v.to(device) for k, v in raw_batch["image"].items()}
            target = _get_target({**raw_batch, "mask": raw_batch["mask"]}).to(device)
            topo   = _get_topo(raw_batch)
            if topo is None:
                print(f"  tile {tile_idx}: no DEM in batch — skip")
                tile_idx += 1
                continue
            topo = topo.to(device)

            # Build 15-channel input
            x = assembler(image)

            # Forward
            logits = model(x)                          # [B, 2, H, W]
            probs  = torch.softmax(logits, dim=1)      # [B, 2, H, W]

            # Process each sample in the batch
            B = logits.shape[0]
            for b in range(B):
                if max_tiles is not None and tile_idx >= max_tiles:
                    break

                h_2d     = topo[b].cpu().numpy()       # [H, W]
                lab_2d   = target[b].cpu().numpy()     # [H, W]
                pred_2d  = torch.argmax(logits[b], dim=0).cpu().numpy()  # [H, W]
                p_w_2d   = probs[b, 1].cpu().numpy()   # [H, W]

                valid_2d = (lab_2d != IGNORE_INDEX) & np.isfinite(h_2d)
                n_valid  = int(valid_2d.sum())

                row: dict[str, Any] = {
                    "tile_index": tile_idx,
                    "n_valid":    n_valid,
                }

                if n_valid < 10:
                    row["skip_reason"] = "too_few_valid_pixels"
                    tile_stats.append(row)
                    tile_idx += 1
                    continue

                h_v    = h_2d[valid_2d].astype(np.float32)
                gt_v   = (lab_2d[valid_2d] == WATER_CLASS).astype(np.int32)
                pr_v   = pred_2d[valid_2d].astype(np.int32)
                p_v    = p_w_2d[valid_2d].astype(np.float32)

                stats = compute_tile_stats(h_v, gt_v, pr_v, p_v, n_bins=n_bins, rng_seed=tile_idx)
                row.update(stats)
                tile_stats.append(row)
                tile_idx += 1

    return tile_stats


# ── Pooled aggregation ────────────────────────────────────────────────────────

def aggregate_tiles(
    tile_stats: list[dict[str, Any]],
    n_bins: int,
    all_h:    np.ndarray,
    all_gt:   np.ndarray,
    all_pred: np.ndarray,
    all_p:    np.ndarray,
) -> dict[str, Any]:
    """
    Compute pooled (all pixels) and per-tile (mean/std) aggregate stats.
    Also computes pooled pi curves by summing bin counts.
    """
    valid_tiles = [t for t in tile_stats if "auc_elev_gt" in t]
    n_tiles = len(valid_tiles)

    agg: dict[str, Any] = {"n_tiles_total": len(tile_stats), "n_tiles_valid": n_tiles}

    # ── Pooled AUC (across all valid pixels) ─────────────────────────────────
    agg["pooled_auc_elev_gt"]          = _safe_auc(all_gt,   -all_h)
    agg["pooled_auc_elev_pred_binary"] = _safe_auc(all_pred, -all_h)
    agg["pooled_auc_model_vs_gt"]      = _safe_auc(all_gt,    all_p)

    rng = np.random.default_rng(0)
    h_shuf = rng.permutation(-all_h)
    agg["pooled_auc_shuffled_gt"]   = _safe_auc(all_gt,   h_shuf)
    agg["pooled_auc_shuffled_pred"] = _safe_auc(all_pred, h_shuf)

    # Pooled elevation gaps
    gt_w  = all_gt  == 1
    gt_d  = all_gt  == 0
    pr_w  = all_pred == 1
    pr_d  = all_pred == 0
    agg["pooled_mean_h_gt_water"]   = float(all_h[gt_w].mean()) if gt_w.sum() > 0 else float("nan")
    agg["pooled_mean_h_gt_dry"]     = float(all_h[gt_d].mean()) if gt_d.sum() > 0 else float("nan")
    agg["pooled_elev_gap_gt"]       = agg["pooled_mean_h_gt_dry"] - agg["pooled_mean_h_gt_water"]
    agg["pooled_mean_h_pred_water"] = float(all_h[pr_w].mean()) if pr_w.sum() > 0 else float("nan")
    agg["pooled_mean_h_pred_dry"]   = float(all_h[pr_d].mean()) if pr_d.sum() > 0 else float("nan")
    agg["pooled_elev_gap_pred"]     = agg["pooled_mean_h_pred_dry"] - agg["pooled_mean_h_pred_water"]

    # ── Pooled pi curves (accumulate bin counts) ──────────────────────────────
    N_all = len(all_h)
    ranks = np.argsort(np.argsort(all_h))
    bins  = np.minimum(ranks * n_bins // max(N_all, 1), n_bins - 1).astype(np.int32)

    pooled_pi_gt   = []
    pooled_pi_pred = []
    pooled_n_bin   = []
    for b in range(n_bins):
        mb = bins == b
        nb = int(mb.sum())
        pooled_n_bin.append(nb)
        pooled_pi_gt.append(  float(all_gt[mb].mean())   if nb > 0 else float("nan"))
        pooled_pi_pred.append(float(all_pred[mb].mean()) if nb > 0 else float("nan"))

    agg["pooled_pi_gt"]   = pooled_pi_gt
    agg["pooled_pi_pred"] = pooled_pi_pred
    agg["pooled_n_bin"]   = pooled_n_bin

    # ── Per-tile mean/std ─────────────────────────────────────────────────────
    def _per_tile_stats(key: str) -> dict[str, float]:
        vals = [t[key] for t in valid_tiles if key in t and math.isfinite(t[key])]
        if not vals:
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        a = np.array(vals)
        return {
            "mean": float(a.mean()),
            "std":  float(a.std(ddof=1) if len(a) > 1 else 0.0),
            "n":    len(a),
        }

    for k in ("auc_elev_gt", "auc_elev_pred_binary", "auc_model_vs_gt",
              "auc_shuffled_gt", "elev_gap_gt", "elev_gap_pred"):
        agg[f"per_tile_{k}"] = _per_tile_stats(k)

    # Pi curve mean/std across tiles per bin
    pi_gt_tile   = [[t["pi_gt"][b]   for t in valid_tiles if "pi_gt"   in t and math.isfinite(t["pi_gt"][b])]
                    for b in range(n_bins)]
    pi_pred_tile = [[t["pi_pred"][b] for t in valid_tiles if "pi_pred" in t and math.isfinite(t["pi_pred"][b])]
                    for b in range(n_bins)]

    agg["per_tile_pi_gt_mean"]   = [float(np.mean(v))  if v else float("nan") for v in pi_gt_tile]
    agg["per_tile_pi_gt_std"]    = [float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for v in pi_gt_tile]
    agg["per_tile_pi_pred_mean"] = [float(np.mean(v))  if v else float("nan") for v in pi_pred_tile]
    agg["per_tile_pi_pred_std"]  = [float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for v in pi_pred_tile]

    return agg


# ── Output writers ────────────────────────────────────────────────────────────

def _clean_for_json(obj: Any) -> Any:
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_for_json(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _fmt(v: Any, decimals: int = 4) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "N/A"
    return f"{v:.{decimals}f}"


def save_tile_csv(tile_stats: list[dict[str, Any]], path: Path) -> None:
    if not tile_stats:
        return
    scalar_keys = [k for k in tile_stats[0] if not isinstance(tile_stats[0][k], list)]
    all_keys: list[str] = []
    seen: set[str] = set()
    for row in tile_stats:
        for k in row:
            if k not in seen and not isinstance(row[k], list):
                all_keys.append(k)
                seen.add(k)

    lines = [",".join(all_keys)]
    for row in tile_stats:
        cells = []
        for k in all_keys:
            v = row.get(k, "")
            if isinstance(v, list):
                cells.append("")
            elif isinstance(v, float):
                cells.append("" if math.isnan(v) or math.isinf(v) else f"{v:.6f}")
            else:
                cells.append(str(v) if v is not None else "")
        lines.append(",".join(cells))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_markdown(
    results_by_split: dict[str, dict[str, Any]],
    run_tag: str,
    config_path: str,
    ckpt_path: str,
    output_path: Path,
    n_bins: int,
) -> None:
    lines: list[str] = []
    lines += [
        f"# Elevation AUC Diagnostic — {run_tag}",
        "",
        "**Read-only. No training. No config/loss/model changes.**",
        "",
        f"- Config: `{config_path}`",
        f"- Checkpoint: `{ckpt_path}`",
        "",
        "## Metrics",
        "",
        "| Metric | Description |",
        "|--------|-------------|",
        "| `AUC(-elev -> GT water)` | AUROC of negative elevation as predictor of GT flood label. ~0.8 in train data. |",
        "| `AUC(-elev -> pred binary)` | Same but predicting model's binary output. If < AUC_GT -> model under-uses elevation. |",
        "| `AUC(p_water -> GT)` | Standard model discrimination AUC. |",
        "| `AUC shuffled GT` | Sanity check: permuted elevation -> GT label. Should be ~0.5. |",
        "| `elev_gap_GT` | mean(h dry) - mean(h water) for GT labels. Positive -> water is lower. |",
        "| `elev_gap_pred` | Same for model predictions. |",
        "| `pi_GT(b)` | P(GT water | elevation-rank bin b). Should decrease as elevation increases. |",
        "| `pi_pred(b)` | P(pred water | elevation-rank bin b). |",
        "",
        "## Decision Rule",
        "",
        "- `AUC_pred >= AUC_GT` OR `pi_pred >= pi_GT` -> model already captures elevation structure -> **stop loss engineering**",
        "- `AUC_pred < AUC_GT` AND `pi_pred flatter than pi_GT` -> model under-uses elevation -> **propose ElevationPriorLoss plan**",
        "",
    ]

    for split, res in results_by_split.items():
        agg = res.get("aggregate", {})
        n_t = agg.get("n_tiles_total", "?")
        n_v = agg.get("n_tiles_valid", "?")
        lines += [f"## Split: `{split}` ({n_t} tiles, {n_v} with valid stats)", ""]

        # Key AUC table
        lines += [
            "### Pooled AUC",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| AUC(-elev -> GT water) [pooled] | {_fmt(agg.get('pooled_auc_elev_gt'))} |",
            f"| AUC(-elev -> pred binary) [pooled] | {_fmt(agg.get('pooled_auc_elev_pred_binary'))} |",
            f"| AUC(p_water -> GT) [pooled] | {_fmt(agg.get('pooled_auc_model_vs_gt'))} |",
            f"| AUC(shuffled elev -> GT) [sanity ~0.5] | {_fmt(agg.get('pooled_auc_shuffled_gt'))} |",
            f"| Elevation gap GT [m] | {_fmt(agg.get('pooled_elev_gap_gt'))} |",
            f"| Elevation gap pred [m] | {_fmt(agg.get('pooled_elev_gap_pred'))} |",
            "",
        ]

        # Per-tile mean/std
        def _pt(key: str) -> str:
            d = agg.get(f"per_tile_{key}", {})
            m, s, n = d.get("mean", float("nan")), d.get("std", float("nan")), d.get("n", 0)
            return f"{_fmt(m)} +/- {_fmt(s)} (n={n})"

        lines += [
            "### Per-tile AUC (mean +/- std)",
            "",
            "| Metric | Per-tile mean +/- std |",
            "|--------|----------------------|",
            f"| AUC(-elev -> GT water) | {_pt('auc_elev_gt')} |",
            f"| AUC(-elev -> pred binary) | {_pt('auc_elev_pred_binary')} |",
            f"| AUC(p_water -> GT) | {_pt('auc_model_vs_gt')} |",
            f"| AUC(shuffled -> GT) | {_pt('auc_shuffled_gt')} |",
            f"| Elevation gap GT [m] | {_pt('elev_gap_gt')} |",
            f"| Elevation gap pred [m] | {_pt('elev_gap_pred')} |",
            "",
        ]

        # Pi curves table
        pi_gt   = agg.get("pooled_pi_gt",   [float("nan")] * n_bins)
        pi_pred = agg.get("pooled_pi_pred", [float("nan")] * n_bins)
        bin_labels = [f"[{b}/{n_bins}]" for b in range(n_bins)]
        lines += [
            "### Pooled Pi Curves (P(water | elevation rank bin))",
            "Bin 0 = lowest elevation, Bin N-1 = highest elevation.",
            "",
            "| Bin (low->high elev) | pi_GT | pi_pred | pi_GT - pi_pred |",
            "|---------------------|-------|---------|----------------|",
        ]
        for b in range(n_bins):
            pg = pi_gt[b]
            pp = pi_pred[b]
            diff = (pg - pp) if math.isfinite(pg) and math.isfinite(pp) else float("nan")
            lines.append(f"| {bin_labels[b]} | {_fmt(pg)} | {_fmt(pp)} | {_fmt(diff)} |")
        lines += [""]

    lines += [
        "---",
        f"*Generated by `experiments_cvpr/segman/diagnose_elevation_auc_predictions.py`.*",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── Config loading ────────────────────────────────────────────────────────────

def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8-sig") as f:
        config = yaml.safe_load(f)
    # Inject DEM shuffle map if present (same logic as train_segman.py)
    dem_map_file = config.get("dem", {}).get("dem_tile_id_map_file")
    if dem_map_file and Path(dem_map_file).exists():
        with open(dem_map_file, encoding="utf-8") as f:
            config.setdefault("dem", {})["dem_tile_id_map"] = json.load(f).get("mapping", {})
    return config


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Elevation AUC diagnostic on SegMAN predictions.")
    ap.add_argument("--config", type=Path, required=True,
                    help="Path to YAML config (same format as train_segman configs).")
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="Checkpoint path. Default: {config.run_dir}/checkpoints/best_checkpoint.pt")
    ap.add_argument("--split", default="all",
                    choices=["val", "test", "bolivia", "all"],
                    help="Which split(s) to process.")
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cuda", "cpu"],
                    help="Device for inference (default: auto).")
    ap.add_argument("--bins", type=int, default=10,
                    help="Number of elevation rank bins for pi curves (default: 10).")
    ap.add_argument("--out-suffix", default=None,
                    help="Suffix for output filenames (default: run_tag from config).")
    ap.add_argument("--max-tiles", type=int, default=None,
                    help="Process at most N tiles per split (smoke test).")
    args = ap.parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    config = load_config(args.config)
    run_tag = config.get("run_tag", args.config.stem)
    suffix  = args.out_suffix or run_tag

    # ── Checkpoint path ───────────────────────────────────────────────────────
    run_dir = Path(config["run_dir"])
    ckpt_path = args.ckpt or (run_dir / "checkpoints" / "best_checkpoint.pt")
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        return 1

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Output dirs ───────────────────────────────────────────────────────────
    reports_dir = REPO_ROOT / "reports"
    docs_dir    = REPO_ROOT / "docs"
    reports_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    # ── Build model + load weights ────────────────────────────────────────────
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model_cfg = dict(config["model"])
    assembler = InputAssembler(config, device)
    model_cfg.setdefault("in_chans", sum(
        len(config["data"]["init_args"]["means"][m])
        for m in config["data"]["init_args"]["modalities"]
    ))

    if config.get("dem", {}).get("use_as_model_input", False):
        raise RuntimeError("Config has dem.use_as_model_input=true — DEM must not be a model input.")

    model = build_segman(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    best_ep  = ckpt.get("best_epoch", "?")
    best_m   = ckpt.get("best_validation_miou", float("nan"))
    print(f"Model: SegMAN-{model_cfg.get('variant','s')} ({n_params:.2f}M params) "
          f"best_ep={best_ep} best_miou={_fmt(best_m)}")

    splits = ["val", "test", "bolivia"] if args.split == "all" else [args.split]

    results_by_split: dict[str, dict[str, Any]] = {}

    for split in splits:
        print(f"\n{'='*60}")
        print(f"  Split: {split}")
        print(f"{'='*60}")
        t0 = time.time()

        try:
            # TopographyDataModule uses "valid" as the internal name for val
            dm_split = "valid" if split == "val" else split
            dm = TopographyDataModule(config, batch_size=1)
            dm.split = dm_split
            dm.setup("test")
            loader = dm.test_dataloader()
        except Exception as e:
            print(f"  SKIP: DataModule setup failed: {e}")
            continue

        tile_stats = run_inference_split(
            model, assembler, loader, device,
            n_bins=args.bins, max_tiles=args.max_tiles,
        )

        n_done = len(tile_stats)
        n_valid_tiles = sum(1 for t in tile_stats if "auc_elev_gt" in t)
        print(f"  Tiles processed: {n_done} (valid stats: {n_valid_tiles})")

        # Pooled arrays for pooled AUC
        h_all    = []
        gt_all   = []
        pred_all = []
        p_all    = []
        for t in tile_stats:
            if "auc_elev_gt" not in t:
                continue
            # We don't have the raw arrays stored; we need to recompute pooled stats
            # from the aggregate (we don't store raw pixel arrays). The per-tile stats
            # already have all we need for mean/std. For truly pooled AUC we'd need
            # raw pixels — but that would require a second pass or storing all pixels.
            # We use per-tile stats as proxy and note the pooled AUC is "per-tile weighted".

        # For pooled AUC: run a second pass accumulating raw pixels
        # This is the correct approach for a true pooled AUROC.
        print("  Running second pass for pooled (all-pixel) AUC ...")
        try:
            dm2 = TopographyDataModule(config, batch_size=1)
            dm2.split = dm_split
            dm2.setup("test")
            loader2 = dm2.test_dataloader()
        except Exception as e:
            print(f"  Second pass failed: {e}; using per-tile aggregation only.")
            loader2 = None

        if loader2 is not None:
            h_all_list    = []
            gt_all_list   = []
            pred_all_list = []
            p_all_list    = []
            model.eval()
            n_accum = 0
            with torch.no_grad():
                for raw_batch in loader2:
                    if args.max_tiles is not None and n_accum >= args.max_tiles:
                        break
                    image  = {k: v.to(device) for k, v in raw_batch["image"].items()}
                    target = _get_target({**raw_batch, "mask": raw_batch["mask"]}).to(device)
                    topo   = _get_topo(raw_batch)
                    if topo is None:
                        n_accum += raw_batch["mask"].shape[0]
                        continue
                    topo   = topo.to(device)
                    x      = assembler(image)
                    logits = model(x)
                    probs  = torch.softmax(logits, dim=1)

                    for b in range(logits.shape[0]):
                        h_2d   = topo[b].cpu().numpy()
                        lab_2d = target[b].cpu().numpy()
                        pred_2d = torch.argmax(logits[b], dim=0).cpu().numpy()
                        p_w_2d  = probs[b, 1].cpu().numpy()

                        valid_2d = (lab_2d != IGNORE_INDEX) & np.isfinite(h_2d)
                        if valid_2d.sum() < 1:
                            continue
                        h_all_list.append(h_2d[valid_2d].astype(np.float32))
                        gt_all_list.append((lab_2d[valid_2d] == WATER_CLASS).astype(np.int32))
                        pred_all_list.append(pred_2d[valid_2d].astype(np.int32))
                        p_all_list.append(p_w_2d[valid_2d].astype(np.float32))
                    n_accum += logits.shape[0]

            all_h    = np.concatenate(h_all_list)    if h_all_list    else np.array([])
            all_gt   = np.concatenate(gt_all_list)   if gt_all_list   else np.array([])
            all_pred = np.concatenate(pred_all_list) if pred_all_list else np.array([])
            all_p    = np.concatenate(p_all_list)    if p_all_list    else np.array([])
        else:
            all_h = all_gt = all_pred = all_p = np.array([])

        agg = aggregate_tiles(tile_stats, args.bins, all_h, all_gt, all_pred, all_p)

        elapsed = time.time() - t0
        print(f"  Elapsed: {elapsed:.1f}s")
        print(f"  Pooled AUC(-elev -> GT):          {_fmt(agg.get('pooled_auc_elev_gt'))}")
        print(f"  Pooled AUC(-elev -> pred binary): {_fmt(agg.get('pooled_auc_elev_pred_binary'))}")
        print(f"  Pooled AUC(model vs GT):          {_fmt(agg.get('pooled_auc_model_vs_gt'))}")
        print(f"  Pooled AUC(shuffled -> GT):       {_fmt(agg.get('pooled_auc_shuffled_gt'))} [expect ~0.5]")
        print(f"  Pooled elev_gap_GT [m]:           {_fmt(agg.get('pooled_elev_gap_gt'))}")
        print(f"  Pooled elev_gap_pred [m]:         {_fmt(agg.get('pooled_elev_gap_pred'))}")
        pt_gt   = agg.get("per_tile_auc_elev_gt",          {})
        pt_pred = agg.get("per_tile_auc_elev_pred_binary", {})
        print(f"  Per-tile AUC_GT:   {_fmt(pt_gt.get('mean'))} +/- {_fmt(pt_gt.get('std'))} (n={pt_gt.get('n')})")
        print(f"  Per-tile AUC_pred: {_fmt(pt_pred.get('mean'))} +/- {_fmt(pt_pred.get('std'))} (n={pt_pred.get('n')})")

        print(f"  Pi curves (pooled, bin 0=lowest elev):")
        pi_gt   = agg.get("pooled_pi_gt",   [])
        pi_pred = agg.get("pooled_pi_pred", [])
        for b in range(args.bins):
            pg = _fmt(pi_gt[b]   if b < len(pi_gt)   else float("nan"))
            pp = _fmt(pi_pred[b] if b < len(pi_pred) else float("nan"))
            print(f"    bin {b:2d}: pi_GT={pg}  pi_pred={pp}")

        results_by_split[split] = {
            "split":     split,
            "run_tag":   run_tag,
            "n_tiles":   n_done,
            "aggregate": agg,
            "per_tile":  tile_stats,
        }

        # Save per-split CSV
        csv_path = reports_dir / f"elevation_auc_predictions_{suffix}_{split}.csv"
        save_tile_csv(tile_stats, csv_path)
        print(f"  CSV: {csv_path}")

    if not results_by_split:
        print("No splits processed.")
        return 1

    # ── Save combined JSON ────────────────────────────────────────────────────
    json_out = {
        "run_tag":     run_tag,
        "config_path": str(args.config),
        "ckpt_path":   str(ckpt_path),
        "splits":      list(results_by_split.keys()),
        "n_bins":      args.bins,
        "results":     {
            split: {
                "n_tiles":   res["n_tiles"],
                "aggregate": res["aggregate"],
            }
            for split, res in results_by_split.items()
        },
    }
    json_path = reports_dir / f"elevation_auc_predictions_{suffix}.json"
    json_path.write_text(
        json.dumps(_clean_for_json(json_out), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nJSON: {json_path}")

    # ── Save Markdown ─────────────────────────────────────────────────────────
    md_path = docs_dir / f"elevation_auc_predictions_{suffix}.md"
    agg_by_split = {split: res for split, res in results_by_split.items()}
    save_markdown(
        results_by_split=agg_by_split,
        run_tag=run_tag,
        config_path=str(args.config),
        ckpt_path=str(ckpt_path),
        output_path=md_path,
        n_bins=args.bins,
    )
    print(f"MD:   {md_path}")

    # ── Decision summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  DECISION SUMMARY")
    print(f"{'='*60}")
    for split, res in results_by_split.items():
        agg = res["aggregate"]
        auc_gt   = agg.get("pooled_auc_elev_gt",          float("nan"))
        auc_pred = agg.get("pooled_auc_elev_pred_binary", float("nan"))
        gap_gt   = agg.get("pooled_elev_gap_gt",          float("nan"))
        gap_pred = agg.get("pooled_elev_gap_pred",        float("nan"))
        print(f"  [{split}] AUC_GT={_fmt(auc_gt)}  AUC_pred={_fmt(auc_pred)}  "
              f"gap_GT={_fmt(gap_gt)}m  gap_pred={_fmt(gap_pred)}m")
        if math.isfinite(auc_gt) and math.isfinite(auc_pred):
            if auc_pred >= auc_gt:
                print(f"    -> AUC_pred >= AUC_GT: model already captures elevation structure.")
            else:
                delta = auc_gt - auc_pred
                print(f"    -> AUC_pred < AUC_GT by {delta:.4f}: elevation gap in predictions.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
