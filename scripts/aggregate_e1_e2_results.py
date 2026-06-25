"""
Aggregate E1+E2 multi-seed N=50 results into a single comparison table.

Reads the final JSON from each run directory and assembles:
  - Per-run metrics
  - Collapsed flag (water_iou < 0.01 OR water_pred_pixels < 5000 at best ckpt)
  - Summary statistics: collapse rate per condition, mean±std of non-collapsed runs

Usage:
    python aggregate_e1_e2_results.py [--out results/e1_e2_multiseed_table.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

SEEDS = [0, 1, 2, 3, 42]
CONDITIONS = [
    ("dice_only",               "Dice-only"),
    ("physics_real_dem_lambda05",    "Physics real DEM"),
    ("physics_shuffled_dem_lambda05","Physics shuffled DEM"),
]
BASE_RUN_DIR = Path("E:/flood_research/experiments/terramind_baseline/runs")
REPO_ROOT = Path(__file__).resolve().parents[1]

COLLAPSE_WATER_IOU_THRESH = 0.01
COLLAPSE_WATER_PIXELS_THRESH = 5000

# seed42 existing run dirs (from previous runs, non-E1 naming)
SEED42_EXISTING = {
    "dice_only": "step5s_a_low_data_n50_seed42_dice",
    "physics_real_dem_lambda05": "step6c_v3_low_data_n50_seed42_lambda05_warmup",
}

# Known metrics for seed42 from completed runs (older format; injected directly)
SEED42_KNOWN_METRICS = {
    "dice_only": {
        "status": "done", "best_epoch": 1, "val_miou": 0.444924,
        "val_water_iou": 0.000095, "val_water_f1": 0.000189,
        "val_water_pred_pixels": 212, "val_topo_violation_fraction": None,
        "collapsed": True,
    },
    "physics_real_dem_lambda05": {
        "status": "passed", "best_epoch": 55, "val_miou": 0.8243,
        "val_water_iou": 0.6914, "val_water_f1": 0.8176,
        "val_water_pred_pixels": 2116688, "val_topo_violation_fraction": 8.36e-4,
        "collapsed": False,
    },
}


def is_collapsed(metrics_best: dict) -> bool:
    water_iou = metrics_best.get("val_iou_water", metrics_best.get("val_water_iou", 1.0))
    water_pix = metrics_best.get("val_water_pred_pixels", 1e9)
    return (water_iou < COLLAPSE_WATER_IOU_THRESH) or (water_pix < COLLAPSE_WATER_PIXELS_THRESH)


def find_metrics_json(run_dir: Path) -> Path | None:
    for name in run_dir.glob("metrics/*.json"):
        return name
    return None


def load_run_results(seed: int, condition_key: str) -> dict | None:
    tag = f"n50_seed{seed}_{condition_key}"
    run_dir = BASE_RUN_DIR / tag

    # For seed42, also try existing run dirs
    if seed == 42 and condition_key in SEED42_EXISTING and not run_dir.exists():
        tag_fallback = SEED42_EXISTING[condition_key]
        run_dir_fallback = BASE_RUN_DIR / tag_fallback
        if run_dir_fallback.exists():
            run_dir = run_dir_fallback
            tag = tag_fallback

    if not run_dir.exists():
        return None

    json_path = find_metrics_json(run_dir)
    if json_path is None:
        return {"status": "no_metrics_json", "run_dir": str(run_dir)}

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"status": f"json_parse_error: {e}", "run_dir": str(run_dir)}

    # "failed" = parity check failed (e.g. collapse), not a crash — still extract metrics.
    # Only bail out for actual error states that have no epoch_metrics.
    if data.get("status") not in ("passed", "completed", "failed", "early_stopped", None):
        return {"status": data.get("status", "unknown"), "run_dir": str(run_dir)}

    # Extract epoch-level best metrics
    best_epoch = data.get("best_epoch")
    best_val_miou = data.get("best_validation_miou")
    epoch_metrics = data.get("epoch_metrics", [])

    best_row = {}
    if best_epoch and epoch_metrics:
        rows = [e for e in epoch_metrics if e.get("epoch") == best_epoch]
        if rows:
            best_row = rows[0]

    # Also check step5s-A format: evaluations dict with best_epoch results
    if not best_row and "evaluations" in data:
        evals = data["evaluations"]
        if isinstance(evals, dict) and str(best_epoch) in evals:
            best_row = evals[str(best_epoch)]
        elif isinstance(evals, list):
            rows2 = [e for e in evals if e.get("epoch") == best_epoch]
            if rows2:
                best_row = rows2[0]

    def _first_not_none(d, *keys):
        for k in keys:
            v = d.get(k)
            if v is not None:
                return v
        return None

    val_water_iou = _first_not_none(best_row, "val_iou_water", "val_water_iou", "water_iou")
    val_water_f1  = _first_not_none(best_row, "val_f1_water", "val_water_f1", "water_f1")
    val_water_pix = _first_not_none(best_row, "val_water_pred_pixels", "water_pred_pixels")
    val_topo_viol = _first_not_none(best_row, "val_topo_violation_fraction", "topo_violation_fraction")

    collapsed = False
    if val_water_iou is not None:
        collapsed = (val_water_iou < COLLAPSE_WATER_IOU_THRESH) or (
            val_water_pix is not None and val_water_pix < COLLAPSE_WATER_PIXELS_THRESH
        )

    return {
        "seed": seed,
        "condition": condition_key,
        "run_dir": str(run_dir),
        "status": data.get("status", "unknown"),
        "best_epoch": best_epoch,
        "val_miou": best_val_miou,
        "val_water_iou": val_water_iou,
        "val_water_f1": val_water_f1,
        "val_water_pred_pixels": val_water_pix,
        "val_topo_violation_fraction": val_topo_viol,
        "collapsed": collapsed,
        # Test/Bolivia come from separate eval JSONs — filled if available
        "test_miou": None,
        "test_water_iou": None,
        "bolivia_miou": None,
        "bolivia_water_iou": None,
    }


def load_eval_results(seed: int, condition_key: str, row: dict) -> dict:
    """Try to load test/Bolivia metrics from a separate eval JSON if present."""
    run_dir = Path(row["run_dir"])
    for eval_json in run_dir.glob("metrics/*eval*.json"):
        try:
            d = json.loads(eval_json.read_text(encoding="utf-8"))
            metrics = d.get("metrics", {})
            if "test" in metrics:
                row["test_miou"] = metrics["test"].get("mean_iou") or metrics["test"].get("miou")
                row["test_water_iou"] = metrics["test"].get("iou_water")
            if "bolivia" in metrics:
                row["bolivia_miou"] = metrics["bolivia"].get("mean_iou") or metrics["bolivia"].get("miou")
                row["bolivia_water_iou"] = metrics["bolivia"].get("iou_water")
        except Exception:
            pass
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(REPO_ROOT / "results" / "e1_e2_multiseed_table.json"))
    args = parser.parse_args()
    out_path = Path(args.out)

    rows = []
    missing = []

    for seed in SEEDS:
        for cond_key, cond_label in CONDITIONS:
            # Inject known metrics for seed42 completed runs (older format)
            if seed == 42 and cond_key in SEED42_KNOWN_METRICS:
                row = dict(SEED42_KNOWN_METRICS[cond_key])
                run_tag = SEED42_EXISTING.get(cond_key, f"n50_seed42_{cond_key}")
                row.setdefault("run_dir", str(BASE_RUN_DIR / run_tag))
                row.update({"test_miou": None, "test_water_iou": None,
                             "bolivia_miou": None, "bolivia_water_iou": None})
                # Try to load eval results anyway
                row = load_eval_results(seed, cond_key, row)
                row["seed"] = seed
                row["condition"] = cond_key
                row["condition_label"] = cond_label
                rows.append(row)
                continue

            row = load_run_results(seed, cond_key)
            if row is None:
                missing.append(f"seed={seed} cond={cond_key}")
                row = {
                    "seed": seed, "condition": cond_key, "run_dir": "NOT FOUND",
                    "status": "not_started", "best_epoch": None,
                    "val_miou": None, "val_water_iou": None, "val_water_f1": None,
                    "val_water_pred_pixels": None, "val_topo_violation_fraction": None,
                    "collapsed": None,
                    "test_miou": None, "test_water_iou": None,
                    "bolivia_miou": None, "bolivia_water_iou": None,
                }
            else:
                row = load_eval_results(seed, cond_key, row)
            # Always ensure seed/condition keys are present (may be missing in partial dicts)
            row["seed"] = seed
            row["condition"] = cond_key
            row["condition_label"] = cond_label
            rows.append(row)

    # Summary statistics per condition
    summary_by_condition: dict[str, dict] = {}
    for cond_key, cond_label in CONDITIONS:
        cond_rows = [r for r in rows if r.get("condition") == cond_key and r.get("collapsed") is not None]
        n_total = len(cond_rows)
        n_collapsed = sum(1 for r in cond_rows if r["collapsed"])
        non_collapsed = [r for r in cond_rows if not r["collapsed"] and r["val_water_iou"] is not None]
        water_ious = [r["val_water_iou"] for r in non_collapsed]
        summary_by_condition[cond_key] = {
            "condition_label": cond_label,
            "n_seeds_done": n_total,
            "n_collapsed": n_collapsed,
            "collapse_rate": f"{n_collapsed}/{n_total}",
            "mean_water_iou_noncollapsed": round(statistics.mean(water_ious), 4) if water_ious else None,
            "std_water_iou_noncollapsed": round(statistics.stdev(water_ious), 4) if len(water_ious) > 1 else None,
        }

    # Print table
    header = (
        f"{'Seed':>5} | {'Condition':<30} | {'Status':<12} | "
        f"{'best_ep':>7} | {'val_mIoU':>8} | {'val_wIoU':>8} | "
        f"{'val_wpx':>9} | {'collapsed':>9}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        miou = f"{r.get('val_miou'):.4f}" if r.get("val_miou") is not None else "     -"
        wiou = f"{r.get('val_water_iou'):.4f}" if r.get("val_water_iou") is not None else "     -"
        wpx = f"{r.get('val_water_pred_pixels'):>9d}" if r.get("val_water_pred_pixels") is not None else "        -"
        coll = str(r.get("collapsed")) if r.get("collapsed") is not None else "pending"
        ep = f"{r.get('best_epoch'):>7}" if r.get("best_epoch") is not None else "      -"
        print(
            f"{r['seed']:>5} | {r['condition_label']:<30} | {r['status']:<12} | "
            f"{ep} | {miou:>8} | {wiou:>8} | {wpx} | {coll:>9}"
        )
    print(sep)

    print("\n=== Summary by condition ===")
    for cond_key, s in summary_by_condition.items():
        mean_w = f"{s['mean_water_iou_noncollapsed']:.4f}" if s["mean_water_iou_noncollapsed"] is not None else "-"
        std_w = f"{s['std_water_iou_noncollapsed']:.4f}" if s["std_water_iou_noncollapsed"] is not None else "-"
        print(
            f"  {s['condition_label']:<30}  collapse={s['collapse_rate']}  "
            f"water_IoU(non-coll)={mean_w}+/-{std_w}"
        )

    if missing:
        print(f"\n  Missing/not started: {len(missing)}")
        for m in missing:
            print(f"    {m}")

    # Write JSON
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "E1+E2 multi-seed N=50",
        "collapse_definition": {
            "criterion": "val_water_iou < 0.01 OR val_water_pred_pixels < 5000 at best checkpoint",
            "water_iou_threshold": COLLAPSE_WATER_IOU_THRESH,
            "water_pixels_threshold": COLLAPSE_WATER_PIXELS_THRESH,
        },
        "seeds": SEEDS,
        "conditions": [c for c, _ in CONDITIONS],
        "rows": rows,
        "summary_by_condition": summary_by_condition,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nTable written: {out_path}")


if __name__ == "__main__":
    main()
