"""
Aggregate SegMAN N=100 D8 lambda=100 results (seed0).

Compares all 5 conditions:
  1. Dice+CE baseline (seed0)
  2. D8 real DEM   lambda=1  (seed0)
  3. D8 shuffled   lambda=1  (seed0)
  4. D8 real DEM   lambda=100 (seed0)  ← new
  5. D8 shuffled   lambda=100 (seed0)  ← new

Scientific question:
  Does lambda=100 give D8 contribution in the target 1-5% range?
  Does D8 real still beat D8 shuffled at lambda=100?
  Does higher lambda improve mIoU vs baseline, or does it hurt recall further?

Outputs:
  reports/segman_n100_d8_lambda100_seed0_results.csv
  reports/segman_n100_d8_lambda100_seed0_results.json
  docs/segman_n100_d8_lambda100_seed0_summary.md

Usage:
    python experiments_cvpr/segman/aggregate_segman_n100_d8_lambda100_seed0.py
"""
from __future__ import annotations

import csv as csv_mod
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT  = Path(__file__).resolve().parents[2]
RUNS_ROOT  = Path("E:/flood_research/experiments/segman/runs")

VARIANTS: list[tuple[str, str, float]] = [
    ("segman_n100_dice_ce_seed0",                   "dice_ce",                    0.0),
    ("segman_n100_d8_lambda1p0_seed0",              "dice_ce_d8",                 1.0),
    ("segman_n100_d8_dem_shuffled_lambda1p0_seed0", "dice_ce_d8_dem_shuffled",    1.0),
    ("segman_n100_d8_lambda100p0_seed0",            "dice_ce_d8",               100.0),
    ("segman_n100_d8_dem_shuffled_lambda100p0_seed0","dice_ce_d8_dem_shuffled", 100.0),
]

BASELINE_TAG  = "segman_n100_dice_ce_seed0"
D8_REAL_L1    = "segman_n100_d8_lambda1p0_seed0"
D8_SHUF_L1    = "segman_n100_d8_dem_shuffled_lambda1p0_seed0"
D8_REAL_L100  = "segman_n100_d8_lambda100p0_seed0"
D8_SHUF_L100  = "segman_n100_d8_dem_shuffled_lambda100p0_seed0"

SPLITS        = ["valid", "test", "bolivia"]
_SPLIT_ALIASES: dict[str, list[str]] = {"valid": ["valid", "val"]}

EVAL_KEYS_BASIC = [
    "mean_iou", "iou_water", "iou_background",
    "f1_water", "f1_background",
    "precision_water", "recall_water",
    "accuracy",
    "topo_violation_fraction",
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None

def fmt(v, p: int = 5) -> str:
    return "N/A" if v is None else f"{v:.{p}f}"

def fmt_delta(v, p: int = 5) -> str:
    if v is None:
        return "N/A"
    return f"{v:+.{p}f}"

def pct(v, p: int = 2) -> str:
    if v is None:
        return "N/A"
    return f"{v * 100:.{p}f}%"

def _sub(a, b):
    return (a - b) if (a is not None and b is not None) else None

# ── Data loading ──────────────────────────────────────────────────────────────
def load_summary(tag: str) -> Optional[dict]:
    path = RUNS_ROOT / tag / "metrics" / f"{tag}_summary.json"
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if d.get("status") == "done" else None
    except Exception as e:
        print(f"  WARN: {path}: {e}", file=sys.stderr)
        return None

def load_epoch_csv(tag: str) -> list[dict]:
    path = RUNS_ROOT / tag / "metrics" / "training_epoch_metrics.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv_mod.DictReader(f))

# ── Metric extraction ─────────────────────────────────────────────────────────
def get_split_eval(summary: dict, split: str) -> dict:
    evals   = summary.get("evaluations") or {}
    aliases = _SPLIT_ALIASES.get(split, [split]) + [f"final_{split}"]
    return next((evals[a] for a in aliases if evals.get(a)), {}) or {}

def derive_metrics(sd: dict) -> dict:
    """Compute derived metrics from raw confusion matrix."""
    tp = safe_float(sd.get("tp"))
    fp = safe_float(sd.get("fp"))
    fn = safe_float(sd.get("fn"))
    tn = safe_float(sd.get("tn"))
    vp = safe_float(sd.get("valid_pixel_count"))
    sw = safe_float(sd.get("support_water"))
    wp = safe_float(sd.get("water_pred_pixels"))

    def _d(a, b):
        return (a / b) if (a is not None and b is not None and b > 0) else None

    recall      = _d(tp, tp + fn if tp is not None and fn is not None else None)
    precision   = _d(tp, tp + fp if tp is not None and fp is not None else None)
    specificity = _d(tn, tn + fp if tn is not None and fp is not None else None)
    fpr         = _d(fp, fp + tn if fp is not None and tn is not None else None)
    fnr         = _d(fn, fn + tp if fn is not None and tp is not None else None)
    accuracy    = _d(tp + tn if tp is not None and tn is not None else None, vp)
    bal_acc     = (recall + specificity) / 2 if recall is not None and specificity is not None else None

    iou_w  = _d(tp, tp + fp + fn if tp is not None and fp is not None and fn is not None else None)
    iou_bg = _d(tn, tn + fp + fn if tn is not None and fp is not None and fn is not None else None)

    return {
        "precision_water":    precision,
        "recall_water":       recall,
        "specificity":        specificity,
        "fpr":                fpr,
        "fnr":                fnr,
        "accuracy":           accuracy,
        "balanced_accuracy":  bal_acc,
        "iou_water":          iou_w,
        "iou_background":     iou_bg,
        "gt_water_ratio":     _d(sw, vp),
        "pred_water_ratio":   _d(wp, vp),
        "pred_gt_ratio":      _d(wp, sw),
        "water_ratio_error":  abs(_d(wp, vp) - _d(sw, vp)) if _d(wp, vp) is not None and _d(sw, vp) is not None else None,
    }

def extract_row(tag: str, cond: str, lam: float) -> dict:
    s = load_summary(tag)
    row: dict[str, Any] = {
        "tag":         tag,
        "condition":   cond,
        "lambda_topo": lam,
        "status":      s.get("status") if s else "missing",
        "best_epoch":  s.get("best_epoch") if s else None,
        "last_epoch":  s.get("last_epoch") if s else None,
        "best_val_miou": safe_float(s.get("best_validation_miou")) if s else None,
    }
    if s is None:
        return row

    for split in SPLITS:
        sd  = get_split_eval(s, split)
        # Basic keys from JSON
        for k in EVAL_KEYS_BASIC + ["mean_iou", "f1_water"]:
            row[f"{split}_{k}"] = safe_float(sd.get(k))
        # Derived metrics
        dm = derive_metrics(sd)
        for k, v in dm.items():
            row[f"{split}_{k}"] = v

    # Per-epoch stats at best epoch
    rows = load_epoch_csv(tag)
    best = s.get("best_epoch")
    if rows and best is not None:
        ep_row = next((r for r in rows if r.get("epoch") == str(best)), None)
        if ep_row:
            for k in ("train_loss_total", "train_loss_dice", "train_loss_ce",
                      "train_loss_topo", "lambda_topo_epoch"):
                row[k] = safe_float(ep_row.get(k))
            # Effective D8 contribution
            lt  = safe_float(ep_row.get("lambda_topo_epoch"))
            ld8 = safe_float(ep_row.get("train_loss_topo"))
            ld  = safe_float(ep_row.get("train_loss_dice"))
            lce = safe_float(ep_row.get("train_loss_ce"))
            if all(v is not None for v in [lt, ld8, ld, lce]) and (ld + lce) > 0:
                row["eff_d8_contribution"] = (lt * ld8) / (ld + lce)
            else:
                row["eff_d8_contribution"] = None

    return row

# ── Delta computation ─────────────────────────────────────────────────────────
def delta(rows_by_tag: dict, tag_a: str, tag_b: str, key: str) -> Optional[float]:
    a = rows_by_tag.get(tag_a, {}).get(key)
    b = rows_by_tag.get(tag_b, {}).get(key)
    return _sub(a, b)

# ── Markdown ──────────────────────────────────────────────────────────────────
def build_markdown(rows_by_tag: dict, delta_table: list[dict]) -> str:
    L = []
    L += [
        "# SegMAN-S N=100 D8 Downstream Loss -- lambda=100 vs lambda=1 Comparison (seed0)",
        "",
        "**Scientific question**: Does lambda=100 give D8 contribution in the 1-5% target range?",
        "Does D8 real still beat D8 shuffled? Does precision/recall rebalance?",
        "",
        "**DEM note**: DEM is NEVER a model input. Used only in the D8 loss and topo eval metrics.",
        "",
    ]

    # mIoU table
    L += [
        "## mIoU Summary",
        "",
        "| Condition | lambda | Val mIoU | Test mIoU | Bolivia mIoU | Best ep |",
        "|-----------|--------|----------|-----------|--------------|---------|",
    ]
    short = {
        BASELINE_TAG: "Dice+CE baseline",
        D8_REAL_L1:   "D8 real  λ=1",
        D8_SHUF_L1:   "D8 shuf  λ=1",
        D8_REAL_L100: "D8 real  λ=100",
        D8_SHUF_L100: "D8 shuf  λ=100",
    }
    for tag, cond, lam in VARIANTS:
        r = rows_by_tag.get(tag, {})
        L.append(f"| {short.get(tag, tag)} | {lam} | "
                 f"{fmt(r.get('valid_mean_iou'))} | "
                 f"{fmt(r.get('test_mean_iou'))} | "
                 f"{fmt(r.get('bolivia_mean_iou'))} | "
                 f"{r.get('best_epoch', 'N/A')} |")
    L += [""]

    # Key delta table
    L += [
        "## Key Deltas (val mIoU, water metrics)",
        "",
        "| Comparison | Val mIoU | Val water IoU | Val Precision | Val Recall | Val Topo VF |",
        "|------------|----------|---------------|---------------|------------|------------|",
    ]
    comparisons = [
        ("λ=100 real − baseline",   D8_REAL_L100,  BASELINE_TAG),
        ("λ=100 shuf − baseline",   D8_SHUF_L100,  BASELINE_TAG),
        ("λ=100 real − λ=100 shuf", D8_REAL_L100,  D8_SHUF_L100),
        ("λ=100 real − λ=1 real",   D8_REAL_L100,  D8_REAL_L1),
        ("λ=100 shuf − λ=1 shuf",   D8_SHUF_L100,  D8_SHUF_L1),
        ("λ=1   real − baseline",   D8_REAL_L1,    BASELINE_TAG),
        ("λ=1   shuf − baseline",   D8_SHUF_L1,    BASELINE_TAG),
    ]
    for label, tag_a, tag_b in comparisons:
        dm = fmt_delta(delta(rows_by_tag, tag_a, tag_b, "valid_mean_iou"))
        dw = fmt_delta(delta(rows_by_tag, tag_a, tag_b, "valid_iou_water"))
        dp = fmt_delta(delta(rows_by_tag, tag_a, tag_b, "valid_precision_water"))
        dr = fmt_delta(delta(rows_by_tag, tag_a, tag_b, "valid_recall_water"))
        dt = fmt_delta(delta(rows_by_tag, tag_a, tag_b, "valid_topo_violation_fraction"), 6)
        L.append(f"| {label} | {dm} | {dw} | {dp} | {dr} | {dt} |")
    L += [""]

    # D8 contribution table
    L += [
        "## Effective D8 Contribution at Best Epoch",
        "",
        "| Condition | lambda | Raw D8 loss | lambda*D8 | dice+ce | Eff D8% | Best ep |",
        "|-----------|--------|-------------|-----------|---------|---------|---------|",
    ]
    for tag, cond, lam in VARIANTS:
        r  = rows_by_tag.get(tag, {})
        ld8 = r.get("train_loss_topo")
        ld  = r.get("train_loss_dice")
        lce = r.get("train_loss_ce")
        eff = r.get("eff_d8_contribution")
        lam_ep = r.get("lambda_topo_epoch")
        num    = (lam_ep * ld8) if lam_ep is not None and ld8 is not None else None
        den    = (ld + lce)     if ld is not None and lce is not None else None
        L.append(f"| {short.get(tag, tag)} | {lam} | {fmt(ld8, 7)} | "
                 f"{fmt(num, 7)} | {fmt(den, 5)} | {pct(eff)} | {r.get('best_epoch','N/A')} |")
    L += [""]

    # Detailed metrics by split
    L += ["## Detailed Metrics by Split", ""]
    metric_display = [
        ("mean_iou",             "mIoU",               5),
        ("iou_water",            "IoU water",           5),
        ("iou_background",       "IoU background",      5),
        ("f1_water",             "F1 water",            5),
        ("precision_water",      "Precision water",     5),
        ("recall_water",         "Recall water",        5),
        ("specificity",          "Specificity",         5),
        ("accuracy",             "Pixel accuracy",      5),
        ("balanced_accuracy",    "Balanced accuracy",   5),
        ("fpr",                  "FPR",                 5),
        ("fnr",                  "FNR",                 5),
        ("gt_water_ratio",       "GT water ratio",      4),
        ("pred_water_ratio",     "Pred water ratio",    4),
        ("pred_gt_ratio",        "Pred/GT ratio",       4),
        ("topo_violation_fraction","Topo violation",    6),
    ]
    for split in SPLITS:
        L += [f"### {split.capitalize()}", ""]
        hdrs = ["Metric", "Baseline", "D8 real λ=1", "D8 shuf λ=1",
                "D8 real λ=100", "D8 shuf λ=100"]
        L += ["| " + " | ".join(hdrs) + " |",
              "|" + "|".join(["---"] * len(hdrs)) + "|"]
        for key, label, p in metric_display:
            col = f"{split}_{key}"
            vals = [rows_by_tag.get(t, {}).get(col) for t, _, _ in VARIANTS]
            L.append(f"| {label} | " + " | ".join(fmt(v, p) for v in vals) + " |")
        L += [""]

    L += [
        "## Interpretation Guide",
        "",
        "- **Eff D8% < 0.1%**: still underpowered at lambda=100 (unexpected — review loss code).",
        "- **Eff D8% 1-5%**: target range — meaningful gradient influence.",
        "- **Eff D8% > 10%**: potentially overregularised — check if recall collapses further.",
        "- **Real > Shuffled (mIoU, water IoU)**: D8 is DEM-specific at lambda=100.",
        "- **Real > Baseline**: D8 provides performance benefit.",
        "- **Recall drops further vs lambda=1**: lambda=100 may be over-suppressing water predictions.",
        "",
        "## Notes",
        "",
        "- DEM is NOT a model input. Used only in D8 loss and topo eval metrics.",
        f"- Baseline: `{BASELINE_TAG}`",
        "- Aggregation: `experiments_cvpr/segman/aggregate_segman_n100_d8_lambda100_seed0.py`",
    ]
    return "\n".join(L) + "\n"

# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    print("=" * 70)
    print("SegMAN N=100 D8 Lambda=100 Aggregation (seed0)")
    print("=" * 70)

    rows_by_tag: dict[str, dict] = {}
    for tag, cond, lam in VARIANTS:
        s = load_summary(tag)
        if s is None:
            print(f"  MISSING or incomplete: {tag}")
            rows_by_tag[tag] = {
                "tag": tag, "condition": cond, "lambda_topo": lam, "status": "missing"
            }
        else:
            row = extract_row(tag, cond, lam)
            rows_by_tag[tag] = row
            print(f"  OK  {tag}: best_ep={s.get('best_epoch')} "
                  f"val_miou={fmt(safe_float(s.get('best_validation_miou')))} "
                  f"eff_d8={pct(row.get('eff_d8_contribution'))}")

    # Build delta table
    delta_table = []
    for split in SPLITS:
        for key, label, _ in [
            ("mean_iou",           "mIoU",           5),
            ("iou_water",          "IoU water",       5),
            ("precision_water",    "Precision water", 5),
            ("recall_water",       "Recall water",    5),
            ("topo_violation_fraction","Topo VF",     6),
        ]:
            col = f"{split}_{key}"
            def _g(t): return rows_by_tag.get(t, {}).get(col)
            delta_table.append({
                "split":              split,
                "metric":             label,
                "baseline":           fmt(_g(BASELINE_TAG)),
                "d8_real_l1":         fmt(_g(D8_REAL_L1)),
                "d8_shuf_l1":         fmt(_g(D8_SHUF_L1)),
                "d8_real_l100":       fmt(_g(D8_REAL_L100)),
                "d8_shuf_l100":       fmt(_g(D8_SHUF_L100)),
                "l100_real_vs_base":  fmt_delta(_sub(_g(D8_REAL_L100), _g(BASELINE_TAG))),
                "l100_shuf_vs_base":  fmt_delta(_sub(_g(D8_SHUF_L100), _g(BASELINE_TAG))),
                "l100_real_vs_shuf":  fmt_delta(_sub(_g(D8_REAL_L100), _g(D8_SHUF_L100))),
                "l100_vs_l1_real":    fmt_delta(_sub(_g(D8_REAL_L100), _g(D8_REAL_L1))),
                "l100_vs_l1_shuf":    fmt_delta(_sub(_g(D8_SHUF_L100), _g(D8_SHUF_L1))),
            })

    # Save CSV
    reports_dir = REPO_ROOT / "reports"
    docs_dir    = REPO_ROOT / "docs"
    reports_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    all_rows = list(rows_by_tag.values())
    csv_path = reports_dir / "segman_n100_d8_lambda100_seed0_results.csv"
    if all_rows:
        keys = sorted(set(k for r in all_rows for k in r.keys()))
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv_mod.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in all_rows:
                w.writerow({k: r.get(k, "") for k in keys})
    print(f"\nWrote: {csv_path}")

    json_path = reports_dir / "segman_n100_d8_lambda100_seed0_results.json"
    json_path.write_text(
        json.dumps({"runs": all_rows, "delta_table": delta_table,
                    "tags": {
                        "baseline":   BASELINE_TAG,
                        "d8_real_l1":  D8_REAL_L1,
                        "d8_shuf_l1":  D8_SHUF_L1,
                        "d8_real_l100": D8_REAL_L100,
                        "d8_shuf_l100": D8_SHUF_L100,
                    }},
                   indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    print(f"Wrote: {json_path}")

    md_path = docs_dir / "segman_n100_d8_lambda100_seed0_summary.md"
    md_path.write_text(build_markdown(rows_by_tag, delta_table), encoding="utf-8")
    print(f"Wrote: {md_path}")

    # Console summary
    print("\n" + "=" * 70)
    print("EFFECTIVE D8 CONTRIBUTION (at best epoch)")
    for tag, cond, lam in VARIANTS:
        r = rows_by_tag.get(tag, {})
        print(f"  {tag[-45:]:45s}: eff_d8={pct(r.get('eff_d8_contribution'))}  "
              f"val_miou={fmt(r.get('valid_mean_iou'))}")
    print("=" * 70)
    print("Val mIoU Deltas:")
    for label, tag_a, tag_b in [
        ("lambda=100 real vs baseline", D8_REAL_L100, BASELINE_TAG),
        ("lambda=100 real vs shuffled", D8_REAL_L100, D8_SHUF_L100),
        ("lambda=100 real vs lambda=1", D8_REAL_L100, D8_REAL_L1),
    ]:
        v = delta(rows_by_tag, tag_a, tag_b, "valid_mean_iou")
        print(f"  {label}: {fmt_delta(v)}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
