"""
Aggregate SegMAN N=50 multi-seed loss-ablation results.

Reads all completed seed runs (seeds 0, 1, 2, 3, 42) and all four loss
variants.  Computes per-variant mean ± std across seeds.  Handles missing
runs gracefully (skips them with a warning).

CPU-only.  Does not load any checkpoint or use a GPU.

Outputs:
  reports/segman_n50_multiseed_results.csv
  reports/segman_n50_multiseed_results.json
  docs/segman_n50_multiseed_summary.md

Usage:
  python experiments_cvpr/segman/aggregate_multiseed_results.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = Path("E:/flood_research/experiments/segman/runs")

SEEDS = [0, 1, 2, 3, 42]

# (condition, seed) -> run_tag
def run_tag_for(condition: str, seed: int) -> str:
    if condition == "ce" and seed == 0:
        return "segman_ce_seed0_clean"
    suffix = f"seed{seed}"
    return f"segman_{condition}_{suffix}"


CONDITIONS = ["ce", "dice_ce", "dice_ce_topo", "dice_ce_topo_dem_shuffled"]

CONDITION_LABELS = {
    "ce":                           "CE",
    "dice_ce":                      "Dice+CE",
    "dice_ce_topo":                 "Dice+CE+Topo",
    "dice_ce_topo_dem_shuffled":    "Dice+CE+Topo+Shuffled",
}

SPLITS = ["val", "test", "bolivia"]

# Metric keys to extract from summary JSON
SCALAR_METRICS = [
    # identifiers
    "best_epoch", "last_epoch",
    # per-split metrics extracted from summary evaluations block
]

# Keys inside summary["evaluations"]["val"] etc.
EVAL_KEYS = [
    "mean_iou",
    "iou_background",
    "iou_water",
    "f1_background",
    "f1_water",
    "precision_background",
    "precision_water",
    "recall_background",
    "recall_water",
    "accuracy",
    "macro_f1",
    "topo_violation_fraction",
    "topo_violation_count",
    "water_pred_pixels",
    "water_gt_pixels",
    "water_pred_ratio",
    "water_gt_ratio",
    "tp_water",
    "fp_water",
    "tn_background",
    "fn_water",
    "ignored_pixels",
    "n_pixels_total",
]


def load_summary(tag: str) -> dict[str, Any] | None:
    path = RUNS_ROOT / tag / "metrics" / f"{tag}_summary.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("status") != "done":
            return None
        return data
    except Exception as e:
        print(f"  WARNING: could not parse {path}: {e}", file=sys.stderr)
        return None


def safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def extract_run_metrics(summary: dict[str, Any], tag: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "best_epoch":  summary.get("best_epoch"),
        "last_epoch":  summary.get("last_epoch"),
        "best_val_miou": safe_float(summary.get("best_validation_miou")),
        "status":      summary.get("status"),
        "tag":         tag,
    }

    evals = summary.get("evaluations") or summary.get("final_eval") or {}
    # Support two possible summary structures
    if not isinstance(evals, dict):
        evals = {}

    # summaries use "valid" as the validation key, so alias it to "val"
    _SPLIT_ALIASES: dict[str, list[str]] = {"val": ["val", "valid"]}
    for split in SPLITS:
        aliases = _SPLIT_ALIASES.get(split, [split]) + [f"final_{split}"]
        split_data = next((evals[a] for a in aliases if evals.get(a)), {}) or {}
        if not isinstance(split_data, dict):
            split_data = {}
        for k in EVAL_KEYS:
            row[f"{split}_{k}"] = safe_float(split_data.get(k))

    # Also extract training loss columns from the epoch CSV at best epoch
    csv_path = RUNS_ROOT / tag / "metrics" / "training_epoch_metrics.csv"
    if csv_path.exists():
        try:
            import csv as csv_mod
            rows = list(csv_mod.DictReader(csv_path.open(encoding="utf-8")))
            best_ep = summary.get("best_epoch")
            best_row = None
            for r in rows:
                try:
                    if int(float(r.get("epoch", -1))) == best_ep:
                        best_row = r
                        break
                except (TypeError, ValueError):
                    pass
            if best_row is None and rows:
                best_row = rows[-1]
            if best_row:
                for col in ["train_loss_total", "train_loss_ce", "train_loss_dice",
                            "train_loss_topo", "lambda_topo_loss", "lambda_topo_epoch",
                            "val_mean_iou", "learning_rate", "grad_norm"]:
                    row[f"best_epoch_{col}"] = safe_float(best_row.get(col))
        except Exception as e:
            print(f"  WARNING: could not read CSV for {tag}: {e}", file=sys.stderr)

    return row


def mean_std(vals: list[float | None]) -> tuple[float | None, float | None]:
    clean = [v for v in vals if v is not None]
    if not clean:
        return None, None
    n = len(clean)
    mu = sum(clean) / n
    if n == 1:
        return mu, None
    var = sum((x - mu) ** 2 for x in clean) / (n - 1)
    return mu, math.sqrt(var)


def fmt(v: float | None, prec: int = 5) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{prec}f}"


def main() -> None:
    print(f"Reading runs from: {RUNS_ROOT}")
    print(f"Seeds: {SEEDS}")
    print(f"Conditions: {CONDITIONS}")
    print()

    # ── collect all run rows ────────────────────────────────────────────────
    all_rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for seed in SEEDS:
        for cond in CONDITIONS:
            tag = run_tag_for(cond, seed)
            summary = load_summary(tag)
            if summary is None:
                missing.append(f"{tag} (seed={seed} cond={cond})")
                print(f"  MISSING/INCOMPLETE: {tag}")
                continue
            row = extract_run_metrics(summary, tag)
            row["seed"] = seed
            row["condition"] = cond
            row["condition_label"] = CONDITION_LABELS[cond]
            all_rows.append(row)
            bv = row.get("val_mean_iou")
            bt = row.get("test_mean_iou")
            print(f"  OK  {tag}  val_mIoU={fmt(bv,4)}  test_mIoU={fmt(bt,4)}")

    print(f"\nLoaded {len(all_rows)} runs, missing {len(missing)}.")
    if missing:
        print("Missing runs (skipped):")
        for m in missing:
            print(f"  {m}")

    # ── write per-run CSV ───────────────────────────────────────────────────
    reports_dir = REPO_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    docs_dir = REPO_ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Build column order
    fixed_cols = ["condition", "seed", "tag", "status", "best_epoch", "last_epoch", "best_val_miou"]
    split_cols = []
    for split in SPLITS:
        for k in EVAL_KEYS:
            split_cols.append(f"{split}_{k}")
    train_cols = [f"best_epoch_{c}" for c in ["train_loss_total", "train_loss_ce",
                   "train_loss_dice", "train_loss_topo", "lambda_topo_loss",
                   "lambda_topo_epoch", "val_mean_iou", "learning_rate", "grad_norm"]]
    all_cols = fixed_cols + split_cols + train_cols

    import csv as csv_mod
    csv_path = reports_dir / "segman_n50_multiseed_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv_mod.DictWriter(fh, fieldnames=all_cols, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(all_rows, key=lambda r: (r["condition"], r["seed"])):
            writer.writerow({k: ("" if row.get(k) is None else row[k]) for k in all_cols})
    print(f"\nWrote per-run CSV: {csv_path}")

    # ── compute per-condition aggregates ────────────────────────────────────
    aggregate: dict[str, Any] = {}
    for cond in CONDITIONS:
        cond_rows = [r for r in all_rows if r["condition"] == cond]
        seeds_done = sorted(r["seed"] for r in cond_rows)
        agg: dict[str, Any] = {
            "condition":       cond,
            "condition_label": CONDITION_LABELS[cond],
            "n_seeds":         len(cond_rows),
            "seeds_done":      seeds_done,
            "missing_seeds":   [s for s in SEEDS if s not in seeds_done],
        }
        # aggregate each numeric metric
        for col in split_cols + ["best_val_miou", "best_epoch"] + train_cols:
            vals = [r.get(col) for r in cond_rows]
            mu, sd = mean_std(vals)  # type: ignore[arg-type]
            agg[f"{col}_mean"] = mu
            agg[f"{col}_std"]  = sd
        aggregate[cond] = agg

    # ── write aggregate JSON ─────────────────────────────────────────────────
    json_payload = {
        "experiment": "SEGMAN_cvpr2025_loss_ablation",
        "model": "SegMAN-S",
        "n_train": 50,
        "seeds": SEEDS,
        "n_val": 86,
        "n_test": 89,
        "n_bolivia": 15,
        "conditions": CONDITIONS,
        "n_loaded": len(all_rows),
        "n_missing": len(missing),
        "missing_runs": missing,
        "per_run": all_rows,
        "aggregate_by_condition": aggregate,
    }

    def _safe(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_safe(v) for v in obj]
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        return obj

    json_path = reports_dir / "segman_n50_multiseed_results.json"
    json_path.write_text(
        json.dumps(_safe(json_payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote aggregate JSON: {json_path}")

    # ── write markdown summary ───────────────────────────────────────────────
    def _row_md(cond: str, split: str, metric: str) -> str:
        agg = aggregate.get(cond, {})
        mu = agg.get(f"{split}_{metric}_mean")
        sd = agg.get(f"{split}_{metric}_std")
        n  = agg.get("n_seeds", 0)
        if mu is None:
            return "N/A"
        if sd is None or n < 2:
            return f"{mu:.4f} (n={n})"
        return f"{mu:.4f} ± {sd:.4f}"

    md_lines = [
        "# SegMAN-S N=50 Multi-Seed Loss Ablation — Aggregated Results",
        "",
        f"**Seeds**: {SEEDS}",
        f"**Conditions**: {list(CONDITION_LABELS.values())}",
        f"**Loaded runs**: {len(all_rows)} / {len(SEEDS) * len(CONDITIONS)}",
        "",
    ]
    if missing:
        md_lines += ["**Missing runs** (excluded from statistics):", ""]
        for m in missing:
            md_lines.append(f"- {m}")
        md_lines.append("")

    for split in SPLITS:
        md_lines += [
            f"## {split.capitalize()} Split",
            "",
            "| Condition | mIoU (mean±std) | IoU_water | F1_water | Prec_water | Rec_water | Topo_viol |",
            "|-----------|-----------------|-----------|----------|------------|-----------|-----------|",
        ]
        for cond in CONDITIONS:
            label = CONDITION_LABELS[cond]
            n = aggregate.get(cond, {}).get("n_seeds", 0)
            row_md = (
                f"| {label} "
                f"| {_row_md(cond, split, 'mean_iou')} "
                f"| {_row_md(cond, split, 'iou_water')} "
                f"| {_row_md(cond, split, 'f1_water')} "
                f"| {_row_md(cond, split, 'precision_water')} "
                f"| {_row_md(cond, split, 'recall_water')} "
                f"| {_row_md(cond, split, 'topo_violation_fraction')} |"
            )
            md_lines.append(row_md)
        md_lines.append("")

    # Per-seed detail table for test split
    md_lines += [
        "## Per-Seed Test mIoU Detail",
        "",
        "| Condition | " + " | ".join(f"seed{s}" for s in SEEDS) + " |",
        "|-----------|" + "|".join("-----" for _ in SEEDS) + "|",
    ]
    for cond in CONDITIONS:
        cells = []
        for s in SEEDS:
            r = next((x for x in all_rows if x["condition"] == cond and x["seed"] == s), None)
            cells.append(fmt(r["test_mean_iou"] if r else None, 4))
        md_lines.append(f"| {CONDITION_LABELS[cond]} | " + " | ".join(cells) + " |")
    md_lines.append("")

    md_lines += [
        "## Notes",
        "",
        "- DEM-shuffle scope verified: shuffle applies to training tiles only; eval uses real DEM.",
        "- See `docs/segman_dem_shuffle_scope_check.md` for implementation audit.",
        "- See `docs/segman_seed0_topo_audit.md` for seed0 baseline analysis.",
        "- Aggregation script: `experiments_cvpr/segman/aggregate_multiseed_results.py`",
    ]

    md_path = docs_dir / "segman_n50_multiseed_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Wrote markdown summary: {md_path}")

    # Print quick summary to stdout
    print("\n=== Quick summary (test mIoU mean ± std) ===")
    for cond in CONDITIONS:
        agg = aggregate[cond]
        mu = agg.get("test_mean_iou_mean")
        sd = agg.get("test_mean_iou_std")
        n  = agg["n_seeds"]
        label = CONDITION_LABELS[cond]
        if mu is None:
            print(f"  {label}: N/A  (n={n})")
        elif sd is None:
            print(f"  {label}: {mu:.4f}  (n={n}, std not computable)")
        else:
            print(f"  {label}: {mu:.4f} ± {sd:.4f}  (n={n})")


if __name__ == "__main__":
    main()
