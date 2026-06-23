"""
STEP 6C/v3 vs STEP 5S-A — no-training visual comparison tool.

Runs inference ONCE for both checkpoints over valid/test/Bolivia using the validated
STEP 6C/v3 dataloader (same splits, same loss-only aligned DEM via albumentations
additional_targets; DEM is NEVER a model input). Computes per-sample IoU water and
topographic violation fraction for both models, selects samples (prioritising where
6C/v3 reduces topographic violations, including neutral and a few regression cases to
avoid cherry-picking), and renders 8-panel comparison figures.

NO training. NO DARN. NO STURM. Raw data untouched. Existing runs untouched.

Outputs:
  reports/figures/step6c_v3_visual_comparison/*.png          (panels)
  reports/figures/step6c_v3_visual_comparison/index.csv      (ALL samples, ranked)
  reports/figures/step6c_v3_visual_comparison/selection_summary.json
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for _p in (str(SRC_ROOT), str(SCRIPTS_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import yaml  # noqa: E402
import step6c_lambda05_train as t6c  # noqa: E402
import step6c_v3_train as v3  # noqa: E402

CONFIG = REPO_ROOT / "configs" / "step6c_v3_terramind_l_upernet_dice_topographic_lambda05_warmup_5sa_dataloader.yaml"
CKPT_5SA = Path("E:/flood_research/experiments/terramind_baseline/runs/step5s_a_terramind_l_upernet_corrected_indices_dice_bs2_accum4/checkpoints/best_checkpoint.pt")
CKPT_V3 = Path("E:/flood_research/experiments/terramind_baseline/runs/step6c_v3_terramind_l_upernet_dice_topographic_lambda05_warmup_5sa_dataloader/checkpoints/best_checkpoint.pt")
OUT_DIR = REPO_ROOT / "reports" / "figures" / "step6c_v3_visual_comparison"
SPLITS = ["valid", "test", "bolivia"]
WATER = 1
IGNORE = -1


def get_tid(batch: dict[str, Any]) -> str:
    fn = batch["filename"]
    if isinstance(fn, dict):
        m = fn["mask"]
        m = m[0] if isinstance(m, (list, tuple)) else m
    elif isinstance(fn, (list, tuple)):
        f0 = fn[0]
        m = f0["mask"] if isinstance(f0, dict) else f0
    else:
        m = fn
    return Path(str(m)).stem.replace("_LabelHand", "")


def jsafe(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): jsafe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsafe(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        v = float(v)
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def violation_pixel_map(pred: np.ndarray, dem: np.ndarray, valid: np.ndarray,
                        margin: float = 0.0, neighborhood: str = "4") -> np.ndarray:
    """Boolean [H,W]: pixel is the HIGH+water endpoint of >=1 violated descending pair."""
    H, W = pred.shape
    vmap = np.zeros((H, W), dtype=bool)
    offsets = [(0, 1), (1, 0)] if neighborhood == "4" else [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dy, dx in offsets:
        ya0, ya1 = max(0, -dy), H - max(0, dy)
        xa0, xa1 = max(0, -dx), W - max(0, dx)
        yb0, yb1 = max(0, dy), H - max(0, -dy)
        xb0, xb1 = max(0, dx), W - max(0, -dx)
        pa, pb = pred[ya0:ya1, xa0:xa1], pred[yb0:yb1, xb0:xb1]
        ha, hb = dem[ya0:ya1, xa0:xa1], dem[yb0:yb1, xb0:xb1]
        va, vb = valid[ya0:ya1, xa0:xa1], valid[yb0:yb1, xb0:xb1]
        vpair = va & vb
        # A is higher than B
        viol_a = vpair & ((ha - hb - margin) > 0) & (pa == WATER) & (pb != WATER)
        vmap[ya0:ya1, xa0:xa1] |= viol_a
        # B is higher than A
        viol_b = vpair & ((hb - ha - margin) > 0) & (pb == WATER) & (pa != WATER)
        vmap[yb0:yb1, xb0:xb1] |= viol_b
    return vmap & valid


def rgb_preview(s2_chw: np.ndarray, idx=(3, 2, 1)) -> np.ndarray:
    rgb = np.stack([s2_chw[i] for i in idx], axis=-1).astype(np.float32)
    out = np.zeros_like(rgb)
    for c in range(3):
        ch = rgb[..., c]
        finite = ch[np.isfinite(ch)]
        if finite.size == 0:
            continue
        lo, hi = np.percentile(finite, (2, 98))
        if hi <= lo:
            hi = lo + 1.0
        out[..., c] = np.clip((ch - lo) / (hi - lo), 0, 1)
    return out


def seg_rgb(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[arr == 0] = (0.12, 0.12, 0.20)
    img[arr == WATER] = (0.15, 0.75, 0.95)
    img[arr == IGNORE] = (0.5, 0.5, 0.5)
    return img


def diff_rgb(p5: np.ndarray, pv: np.ndarray, valid: np.ndarray) -> np.ndarray:
    h, w = p5.shape
    img = np.full((h, w, 3), 0.12, dtype=np.float32)
    both = valid & (p5 == WATER) & (pv == WATER)
    only5 = valid & (p5 == WATER) & (pv != WATER)
    onlyv = valid & (pv == WATER) & (p5 != WATER)
    img[both] = (0.15, 0.75, 0.95)   # both water (agreement)
    img[only5] = (0.95, 0.45, 0.10)  # 5S-A water removed by v3
    img[onlyv] = (0.20, 0.85, 0.30)  # v3 added water
    return img


def dem_gray(dem: np.ndarray) -> np.ndarray:
    d = dem.astype(np.float32)
    finite = d[np.isfinite(d)]
    lo, hi = np.percentile(finite, (2, 98)) if finite.size else (0.0, 1.0)
    g = np.clip((d - lo) / (hi - lo + 1e-6), 0, 1)
    return g


def viol_overlay(dem: np.ndarray, viol: np.ndarray) -> np.ndarray:
    g = dem_gray(dem) * 0.6
    img = np.stack([g, g, g], axis=-1)
    img[viol] = (1.0, 0.08, 0.08)
    return img


def build_task_with_ckpt(config: dict[str, Any], ckpt_path: Path, device: torch.device):
    task = t6c.build_task(config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    task.load_state_dict(state)
    t6c.set_bn_eval(task)
    task.eval()
    return task


@torch.no_grad()
def run_model_over_all(task, config, device, physics) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for split in SPLITS:
        dm = v3.TopographyDataModule(config, batch_size=1, split=split)
        dm.setup("test")
        for batch in dm.test_dataloader():
            tid = get_tid(batch)
            b = t6c.move_batch(batch, device)
            logits = t6c.get_logits(task(b["image"]))
            target = task.squeeze_ground_truth(b["mask"]).long()
            pred = torch.argmax(logits.detach(), dim=1)
            conf = t6c.confusion(target, pred)
            m = t6c.metrics_from_conf(conf)
            tc = t6c.topographic_violation_counts(
                logits=logits.detach(), target=target,
                topography=b["topography"].squeeze(1),
                ignore_index=int(physics["ignore_index"]), water_class=int(physics["water_class"]),
                elevation_margin=float(physics["elevation_margin"]), neighborhood=str(physics["neighborhood"]),
            )
            out[(split, tid)] = {
                "pred": pred[0].to(torch.uint8).cpu().numpy(),
                "iou_water": m["iou_water"],
                "f1_water": m["f1_water"],
                "mean_iou": m["mean_iou"],
                "water_pred_pixels": int(m["tp"]) + int(m["fp"]),
                "valid_pixels": int(m["tn"]) + int(m["fp"]) + int(m["fn"]) + int(m["tp"]),
                "desc_pairs": int(tc["topo_descending_pair_count"]),
                "viol_pairs": int(tc["topo_violation_pair_count"]),
                "viol_frac": float(tc["topo_violation_fraction"]) if math.isfinite(tc["topo_violation_fraction"]) else math.nan,
            }
    return out


def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with CONFIG.open("r", encoding="utf-8-sig") as fh:
        config = yaml.safe_load(fh)
    physics = config["physics_loss"]
    print(f"[viz] device={device} loading 5S-A ...", flush=True)

    # --- Pass A: 5S-A over all (one model in VRAM at a time) ---
    task5 = build_task_with_ckpt(config, CKPT_5SA, device)
    res5 = run_model_over_all(task5, config, device, physics)
    del task5
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"[viz] 5S-A done ({len(res5)} samples). loading 6C/v3 ...", flush=True)

    # --- Pass B: 6C/v3 over all ---
    taskv = build_task_with_ckpt(config, CKPT_V3, device)
    resv = run_model_over_all(taskv, config, device, physics)
    del taskv
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"[viz] 6C/v3 done ({len(resv)} samples).", flush=True)

    # --- per-sample comparison rows ---
    rows: list[dict[str, Any]] = []
    for key in res5:
        if key not in resv:
            continue
        split, tid = key
        a, b = res5[key], resv[key]
        d_viol = (b["viol_frac"] - a["viol_frac"]) if (math.isfinite(a["viol_frac"]) and math.isfinite(b["viol_frac"])) else math.nan
        rows.append({
            "split": split, "sample_id": tid, "location": tid.split("_")[0],
            "iou_water_5sa": a["iou_water"], "iou_water_v3": b["iou_water"],
            "delta_iou_water": (b["iou_water"] - a["iou_water"]) if (math.isfinite(a["iou_water"]) and math.isfinite(b["iou_water"])) else math.nan,
            "f1_water_5sa": a["f1_water"], "f1_water_v3": b["f1_water"],
            "viol_frac_5sa": a["viol_frac"], "viol_frac_v3": b["viol_frac"], "delta_viol_frac": d_viol,
            "viol_pairs_5sa": a["viol_pairs"], "viol_pairs_v3": b["viol_pairs"], "desc_pairs": a["desc_pairs"],
            "water_frac_5sa": a["water_pred_pixels"] / a["valid_pixels"] if a["valid_pixels"] else math.nan,
            "water_frac_v3": b["water_pred_pixels"] / b["valid_pixels"] if b["valid_pixels"] else math.nan,
        })

    # --- aggregate evidence (all samples with valid descending pairs) ---
    valid_rows = [r for r in rows if math.isfinite(r["delta_viol_frac"]) and r["desc_pairs"] > 0]
    n = len(valid_rows)
    n_reduced = sum(1 for r in valid_rows if r["delta_viol_frac"] < 0)
    n_worse = sum(1 for r in valid_rows if r["delta_viol_frac"] > 0)
    n_equal = n - n_reduced - n_worse
    n_iou_better = sum(1 for r in valid_rows if math.isfinite(r["delta_iou_water"]) and r["delta_iou_water"] > 0)
    deltas = [r["delta_viol_frac"] for r in valid_rows]
    aggregate = {
        "n_samples_compared": n,
        "n_viol_reduced_by_v3": n_reduced,
        "n_viol_increased_by_v3": n_worse,
        "n_viol_equal": n_equal,
        "frac_samples_v3_reduces_violations": (n_reduced / n) if n else None,
        "mean_delta_viol_frac": float(np.mean(deltas)) if deltas else None,
        "median_delta_viol_frac": float(np.median(deltas)) if deltas else None,
        "n_iou_water_better_v3": n_iou_better,
    }

    # --- selection (anti-cherry-pick): positives + neutrals per split + a few regressions ---
    selected: list[dict[str, Any]] = []

    def tag_select(pool, tag, k):
        chosen = []
        for r in pool:
            if len(chosen) >= k:
                break
            if (r["split"], r["sample_id"]) in {(s["split"], s["sample_id"]) for s in selected}:
                continue
            r2 = dict(r)
            r2["category"] = tag
            chosen.append(r2)
        selected.extend(chosen)
        return chosen

    for split in SPLITS:
        srows = [r for r in valid_rows if r["split"] == split]
        # positives: 5S-A had violations, v3 reduces most (most negative delta)
        pos = sorted([r for r in srows if r["viol_pairs_5sa"] > 0 and r["delta_viol_frac"] < 0],
                     key=lambda r: r["delta_viol_frac"])
        tag_select(pos, "positive", 2)
        # neutral: smallest |delta|
        neu = sorted([r for r in srows], key=lambda r: abs(r["delta_viol_frac"]))
        tag_select(neu, "neutral", 1)
    # regressions (honesty): v3 increases violations most, any split
    reg = sorted([r for r in valid_rows if r["delta_viol_frac"] > 0], key=lambda r: -r["delta_viol_frac"])
    tag_select(reg, "regression", 2)

    selected_keys = {(s["split"], s["sample_id"]): s for s in selected}

    # --- write full index CSV (ALL samples, ranked by delta_viol_frac) ---
    rows_sorted = sorted(rows, key=lambda r: (r["delta_viol_frac"] if math.isfinite(r["delta_viol_frac"]) else 1e9))
    csv_path = OUT_DIR / "index.csv"
    fields = ["split", "sample_id", "location", "selected", "category", "figure",
              "iou_water_5sa", "iou_water_v3", "delta_iou_water",
              "f1_water_5sa", "f1_water_v3",
              "viol_frac_5sa", "viol_frac_v3", "delta_viol_frac",
              "viol_pairs_5sa", "viol_pairs_v3", "desc_pairs", "water_frac_5sa", "water_frac_v3"]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows_sorted:
            key = (r["split"], r["sample_id"])
            sel = selected_keys.get(key)
            rr = dict(r)
            rr["selected"] = bool(sel)
            rr["category"] = sel["category"] if sel else ""
            rr["figure"] = f"{r['split']}_{r['sample_id']}.png" if sel else ""
            w.writerow({k: jsafe(rr.get(k)) for k in fields})

    # --- Pass C: render selected (re-load images/dem/gt; preds from cache) ---
    margin = float(physics["elevation_margin"])
    neigh = str(physics["neighborhood"])
    rendered = 0
    for split in SPLITS:
        sel_in_split = [s for s in selected if s["split"] == split]
        if not sel_in_split:
            continue
        want = {s["sample_id"] for s in sel_in_split}
        dm = v3.TopographyDataModule(config, batch_size=1, split=split)
        dm.setup("test")
        for batch in dm.test_dataloader():
            tid = get_tid(batch)
            if tid not in want:
                continue
            s2 = batch["image"]["S2L1C"][0].cpu().numpy()
            gt = batch["mask"]
            gt = (gt[0] if gt.ndim == 3 else gt[0, 0]).cpu().numpy().astype(np.int64)
            dem = batch["topography"][0, 0].cpu().numpy().astype(np.float32)
            valid = gt != IGNORE
            p5 = res5[(split, tid)]["pred"].astype(np.int64)
            pv = resv[(split, tid)]["pred"].astype(np.int64)
            v5 = violation_pixel_map(p5, dem, valid, margin, neigh)
            vv = violation_pixel_map(pv, dem, valid, margin, neigh)
            meta = selected_keys[(split, tid)]

            fig, ax = plt.subplots(2, 4, figsize=(20, 10))
            ax[0, 0].imshow(rgb_preview(s2)); ax[0, 0].set_title("Sentinel-2 RGB (B4/B3/B2)")
            ax[0, 1].imshow(seg_rgb(gt)); ax[0, 1].set_title("Ground truth")
            ax[0, 2].imshow(seg_rgb(p5)); ax[0, 2].set_title(f"STEP 5S-A pred (IoU_w={meta['iou_water_5sa']:.3f})")
            ax[0, 3].imshow(seg_rgb(pv)); ax[0, 3].set_title(f"STEP 6C/v3 pred (IoU_w={meta['iou_water_v3']:.3f})")
            ax[1, 0].imshow(dem_gray(dem), cmap="terrain"); ax[1, 0].set_title("DEM (normalized elevation)")
            ax[1, 1].imshow(diff_rgb(p5, pv, valid)); ax[1, 1].set_title("Pred diff (orange=5S-A only, green=v3 only)")
            ax[1, 2].imshow(viol_overlay(dem, v5)); ax[1, 2].set_title(f"5S-A violations (frac={meta['viol_frac_5sa']:.5f}, n={meta['viol_pairs_5sa']})")
            ax[1, 3].imshow(viol_overlay(dem, vv)); ax[1, 3].set_title(f"6C/v3 violations (frac={meta['viol_frac_v3']:.5f}, n={meta['viol_pairs_v3']})")
            for a in ax.ravel():
                a.set_xticks([]); a.set_yticks([])
            dviol = meta["delta_viol_frac"]; diou = meta["delta_iou_water"]
            fig.suptitle(
                f"[{meta['category'].upper()}] {split} / {tid}  |  Δviol_frac={dviol:+.6f}  "
                f"({'v3 fewer' if dviol < 0 else ('v3 more' if dviol > 0 else 'equal')} violations)  |  ΔIoU_water={diou:+.4f}",
                fontsize=14,
            )
            legend = [Patch(facecolor=(0.15, 0.75, 0.95), label="water"),
                      Patch(facecolor=(0.12, 0.12, 0.20), label="background"),
                      Patch(facecolor=(1.0, 0.08, 0.08), label="topo violation")]
            fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False)
            fig.tight_layout(rect=[0, 0.03, 1, 0.96])
            fig_path = OUT_DIR / f"{split}_{tid}.png"
            fig.savefig(fig_path, dpi=110)
            plt.close(fig)
            rendered += 1
            print(f"[viz] rendered {fig_path.name} ({meta['category']})", flush=True)

    summary = {
        "step": "6C-v3-visual-comparison",
        "config": str(CONFIG),
        "ckpt_5sa": str(CKPT_5SA),
        "ckpt_v3": str(CKPT_V3),
        "dem_as_model_input": False,
        "splits": SPLITS,
        "aggregate_evidence": aggregate,
        "n_selected": len(selected),
        "n_rendered": rendered,
        "selected": [{k: jsafe(s.get(k)) for k in
                      ["split", "sample_id", "location", "category", "iou_water_5sa", "iou_water_v3",
                       "delta_iou_water", "viol_frac_5sa", "viol_frac_v3", "delta_viol_frac",
                       "viol_pairs_5sa", "viol_pairs_v3"]} for s in selected],
        "index_csv": str(csv_path),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    (OUT_DIR / "selection_summary.json").write_text(json.dumps(jsafe(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\n=== AGGREGATE ===")
    print(json.dumps(jsafe(aggregate), indent=2))
    print(f"rendered {rendered} panels -> {OUT_DIR}")
    print(f"elapsed {summary['elapsed_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
