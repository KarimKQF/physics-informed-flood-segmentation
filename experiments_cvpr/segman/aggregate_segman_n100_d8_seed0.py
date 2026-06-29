"""
Aggregate SegMAN N=100 D8 downstream loss results (seed0).

Reads 2 completed D8 runs and compares them against the Dice+CE baseline
(segman_n100_dice_ce_seed0) and the best V1-topo lambda sweep run.

Scientific success criteria:
  Strong:  D8 real > D8 shuffled on test mIoU AND water IoU, AND D8 real >= baseline.
  Moderate: D8 real > D8 shuffled on test but D8 real < baseline (recall/overshoot).
  Negative: D8 real ~= D8 shuffled (loss still DEM-agnostic despite GT signal).

Outputs:
  reports/segman_n100_d8_seed0_results.csv
  reports/segman_n100_d8_seed0_results.json
  docs/segman_n100_d8_seed0_summary.md

Usage:
    python experiments_cvpr/segman/aggregate_segman_n100_d8_seed0.py
"""
from __future__ import annotations

import csv as csv_mod
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = Path("E:/flood_research/experiments/segman/runs")

BASELINE_TAG = "segman_n100_dice_ce_seed0"
D8_REAL_TAG  = "segman_n100_d8_lambda1p0_seed0"
D8_SHUF_TAG  = "segman_n100_d8_dem_shuffled_lambda1p0_seed0"

SPLITS        = ["val", "test", "bolivia"]
_SPLIT_ALIASES: dict[str, list[str]] = {"val": ["val", "valid"]}

EVAL_KEYS = [
    "mean_iou",
    "iou_water",
    "f1_water",
    "precision_water",
    "recall_water",
    "topo_violation_fraction",
]

VARIANTS: list[tuple[str, str, float]] = [
    (BASELINE_TAG, "dice_ce",              0.0),
    (D8_REAL_TAG,  "dice_ce_d8",           1.0),
    (D8_SHUF_TAG,  "dice_ce_d8_dem_shuffled", 1.0),
]


def safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def fmt(v: float | None, p: int = 5) -> str:
    return "N/A" if v is None else f"{v:.{p}f}"


def fmt_delta(v: float | None, p: int = 5) -> str:
    if v is None:
        return "N/A"
    return f"{v:+.{p}f}"


def load_summary(tag: str) -> dict[str, Any] | None:
    path = RUNS_ROOT / tag / "metrics" / f"{tag}_summary.json"
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if d.get("status") == "done" else None
    except Exception as e:
        print(f"  WARN: {path}: {e}", file=sys.stderr)
        return None


def load_epoch_csv(tag: str) -> list[dict[str, str]]:
    path = RUNS_ROOT / tag / "metrics" / "training_epoch_metrics.csv"
    if not path.exists():
        return []
    try:
        with path.open(encoding="utf-8") as fh:
            return list(csv_mod.DictReader(fh))
    except Exception:
        return []


def extract_metrics(summary: dict, tag: str, cond: str, lam: float) -> dict[str, Any]:
    row: dict[str, Any] = {
        "tag":         tag,
        "condition":   cond,
        "lambda_topo": lam,
        "status":      summary.get("status"),
        "best_epoch":  summary.get("best_epoch"),
        "last_epoch":  summary.get("last_epoch"),
        "best_val_miou": safe_float(summary.get("best_validation_miou")),
    }
    evals = summary.get("evaluations") or {}
    for split in SPLITS:
        aliases = _SPLIT_ALIASES.get(split, [split]) + [f"final_{split}"]
        sd      = next((evals[a] for a in aliases if evals.get(a)), {}) or {}
        for k in EVAL_KEYS:
            row[f"{split}_{k}"] = safe_float(sd.get(k))

    # Per-epoch loss stats at best epoch (from CSV).
    rows = load_epoch_csv(tag)
    best_ep = summary.get("best_epoch")
    if rows and best_ep is not None:
        ep_row = next((r for r in rows if r.get("epoch") == str(best_ep)), None)
        if ep_row:
            for k in ("train_loss_total", "train_loss_dice", "train_loss_ce",
                      "train_loss_topo", "lambda_topo_epoch"):
                row[k] = safe_float(ep_row.get(k))
            # Effective D8/topo contribution as fraction of dice+ce.
            lt  = safe_float(ep_row.get("lambda_topo_epoch"))
            lto = safe_float(ep_row.get("train_loss_topo"))
            ld  = safe_float(ep_row.get("train_loss_dice"))
            lce = safe_float(ep_row.get("train_loss_ce"))
            if all(v is not None for v in (lt, lto, ld, lce)) and (ld + lce) > 0:
                row["eff_d8_contribution"] = (lt * lto) / (ld + lce)
            else:
                row["eff_d8_contribution"] = None

    return row


def diff_row(a: dict, b: dict, key: str) -> float | None:
    va, vb = a.get(key), b.get(key)
    if va is None or vb is None:
        return None
    return va - vb


def main() -> int:
    print("=" * 70)
    print("SegMAN N=100 D8 Downstream Loss Aggregation")
    print("=" * 70)

    all_rows: list[dict[str, Any]] = []
    summaries: dict[str, dict | None] = {}
    for tag, cond, lam in VARIANTS:
        s = load_summary(tag)
        summaries[tag] = s
        if s is None:
            print(f"  MISSING or incomplete: {tag}")
            all_rows.append({"tag": tag, "condition": cond, "lambda_topo": lam, "status": "missing"})
        else:
            row = extract_metrics(s, tag, cond, lam)
            all_rows.append(row)
            print(f"  OK  {tag}: status={s.get('status')} best_ep={s.get('best_epoch')} "
                  f"best_val_miou={safe_float(s.get('best_validation_miou'))}")

    # Add delta columns: D8 real vs baseline, D8 shuffled vs baseline, D8 real vs shuffled.
    rows_by_tag = {r["tag"]: r for r in all_rows}
    base = rows_by_tag.get(BASELINE_TAG, {})
    real = rows_by_tag.get(D8_REAL_TAG, {})
    shuf = rows_by_tag.get(D8_SHUF_TAG, {})

    delta_rows = []
    for split in SPLITS:
        for k in EVAL_KEYS:
            col = f"{split}_{k}"
            delta_rows.append({
                "split": split, "metric": k,
                "baseline": fmt(base.get(col)),
                "d8_real":  fmt(real.get(col)),
                "d8_shuf":  fmt(shuf.get(col)),
                "real_vs_baseline": fmt_delta(diff_row(real, base, col)),
                "shuf_vs_baseline": fmt_delta(diff_row(shuf, base, col)),
                "real_vs_shuf":     fmt_delta(diff_row(real, shuf, col)),
            })

    # Save CSV.
    reports_dir = REPO_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path  = reports_dir / "segman_n100_d8_seed0_results.csv"
    json_path = reports_dir / "segman_n100_d8_seed0_results.json"
    doc_path  = REPO_ROOT / "docs" / "segman_n100_d8_seed0_summary.md"

    if all_rows:
        keys = list(all_rows[0].keys())
        for r in all_rows:
            for k in keys:
                r.setdefault(k, None)
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv_mod.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nWrote: {csv_path}")

    payload = {
        "runs": all_rows,
        "delta_table": delta_rows,
        "tags": {"baseline": BASELINE_TAG, "d8_real": D8_REAL_TAG, "d8_shuffled": D8_SHUF_TAG},
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Wrote: {json_path}")

    # ---- Markdown summary ----
    b_val  = fmt(base.get("val_mean_iou"))
    b_test = fmt(base.get("test_mean_iou"))
    b_bol  = fmt(base.get("bolivia_mean_iou"))
    r_val  = fmt(real.get("val_mean_iou"))
    r_test = fmt(real.get("test_mean_iou"))
    r_bol  = fmt(real.get("bolivia_mean_iou"))
    s_val  = fmt(shuf.get("val_mean_iou"))
    s_test = fmt(shuf.get("test_mean_iou"))
    s_bol  = fmt(shuf.get("bolivia_mean_iou"))

    dr_val  = fmt_delta(diff_row(real, shuf, "val_mean_iou"))
    dr_test = fmt_delta(diff_row(real, shuf, "test_mean_iou"))
    dr_bol  = fmt_delta(diff_row(real, shuf, "bolivia_mean_iou"))

    md_lines = [
        "# SegMAN-S N=100 D8 Downstream Loss -- seed0 Results",
        "",
        f"**N_train**: 100  |  **Seed**: 0  |  **Loss**: D8DownstreamLoss (lambda=1.0)",
        f"**Baseline** (`segman_n100_dice_ce_seed0`): "
        f"val={b_val}  test={b_test}  bolivia={b_bol}",
        "",
        "## Scientific Question",
        "",
        "> Does the D8 slope-weighted downstream consistency loss show a DEM-specific",
        "> effect (real > shuffled) that was absent from the V1 4-neighbour loss?",
        "",
        "## mIoU Summary",
        "",
        "| Condition | val mIoU | test mIoU | bolivia mIoU |",
        "|-----------|----------|-----------|--------------|",
        f"| Baseline (Dice+CE) | {b_val} | {b_test} | {b_bol} |",
        f"| D8 real DEM        | {r_val} | {r_test} | {r_bol} |",
        f"| D8 shuffled DEM    | {s_val} | {s_test} | {s_bol} |",
        f"| **Real - Shuffled**  | **{dr_val}** | **{dr_test}** | **{dr_bol}** |",
        "",
        "## Detailed Metrics by Split",
        "",
    ]

    for split in SPLITS:
        md_lines += [f"### {split.capitalize()}", ""]
        md_lines += ["| Metric | Baseline | D8 real | D8 shuffled | Real-Baseline | Shuf-Baseline | Real-Shuffled |",
                     "|--------|----------|---------|-------------|---------------|---------------|---------------|"]
        for k in EVAL_KEYS:
            col = f"{split}_{k}"
            md_lines.append(
                f"| {k} | {fmt(base.get(col))} | {fmt(real.get(col))} | {fmt(shuf.get(col))} "
                f"| {fmt_delta(diff_row(real, base, col))} "
                f"| {fmt_delta(diff_row(shuf, base, col))} "
                f"| {fmt_delta(diff_row(real, shuf, col))} |"
            )
        md_lines.append("")

    # Training loss at best epoch.
    md_lines += [
        "## Training Loss at Best Epoch",
        "",
        "| Condition | loss_total | loss_d8 | eff_d8_contribution | best_ep |",
        "|-----------|------------|---------|---------------------|---------|",
    ]
    for tag, cond, lam in VARIANTS:
        r = rows_by_tag.get(tag, {})
        md_lines.append(
            f"| {cond} | {fmt(r.get('train_loss_total'))} "
            f"| {fmt(r.get('train_loss_topo'), 6)} "
            f"| {fmt(r.get('eff_d8_contribution'), 4)} "
            f"| {r.get('best_epoch', 'N/A')} |"
        )

    md_lines += [
        "",
        "## Interpretation",
        "",
        "- **Real-Shuffled > 0**: real DEM provides benefit (D8 captured DEM-specific signal).",
        "- **Real-Shuffled <= 0**: shuffled ~= real (D8 still DEM-agnostic; rethink formulation).",
        "- **Real-Baseline > 0**: D8 loss improves over Dice+CE (strong success).",
        "- **Real-Baseline < 0 and Real > Shuffled**: D8 captures signal but hurts overall mIoU",
        "  (possible over-suppression of recall; try smaller lambda or different tau).",
        "",
        "## Notes",
        "",
        f"- Baseline: `{BASELINE_TAG}`",
        "- GT diagnostic confirmed strong DEM-specific signal (AUC 0.79-0.86 real vs 0.48-0.51 shuffled).",
        "- V1 topo loss (4-neighbour, lambda 1/2/4) showed no robust real>shuffled signal.",
        "- D8 loss uses 8-neighbour steepest descent; shuffled DEM reroutes downstream direction.",
        "- Aggregation: `experiments_cvpr/segman/aggregate_segman_n100_d8_seed0.py`",
        "- DEM is NOT a model input. Used only in loss and eval topo metrics.",
    ]

    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Wrote: {doc_path}")

    # Interpretation block.
    print("\n" + "=" * 70)
    print("mIoU comparison (val / test / bolivia):")
    print(f"  Baseline:      val={b_val}  test={b_test}  bolivia={b_bol}")
    print(f"  D8 real:       val={r_val}  test={r_test}  bolivia={r_bol}")
    print(f"  D8 shuffled:   val={s_val}  test={s_test}  bolivia={s_bol}")
    print(f"  Real-Shuffled: val={dr_val}  test={dr_test}  bolivia={dr_bol}")
    print("=" * 70)

    # Water IoU comparison.
    for split in SPLITS:
        col = f"{split}_iou_water"
        print(f"  water IoU {split}: baseline={fmt(base.get(col))}  "
              f"D8_real={fmt(real.get(col))}  D8_shuf={fmt(shuf.get(col))}  "
              f"real-shuf={fmt_delta(diff_row(real, shuf, col))}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
