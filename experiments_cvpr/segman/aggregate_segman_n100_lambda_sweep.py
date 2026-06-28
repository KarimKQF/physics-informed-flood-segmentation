"""
Aggregate SegMAN N=100 lambda_topo stress test results (seed0).

Reads 6 completed runs (3 lambdas x 2 conditions) and compares them against the
existing N=100 seed0 Dice+CE baseline (segman_n100_dice_ce_seed0).

Key scientific interpretation:
  - If real DEM beats shuffled at a stronger lambda, lambda=0.5 was too weak.
  - If shuffled >= real at all lambdas, the issue is the loss formulation, not lambda.
  - If high lambda degrades mIoU, the constraint is too strong.

Outputs:
  reports/segman_n100_lambda_sweep_seed0_results.csv
  reports/segman_n100_lambda_sweep_seed0_results.json
  docs/segman_n100_lambda_sweep_seed0_summary.md

Usage:
    python experiments_cvpr/segman/aggregate_segman_n100_lambda_sweep.py
"""
from __future__ import annotations

import csv as csv_mod
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT  = Path(__file__).resolve().parents[2]
RUNS_ROOT  = Path("E:/flood_research/experiments/segman/runs")

BASELINE_TAG = "segman_n100_dice_ce_seed0"

LAMBDAS     = [0.5, 1.0, 2.0, 4.0]
CONDITIONS  = ["dice_ce_topo", "dice_ce_topo_dem_shuffled"]
SPLITS      = ["val", "test", "bolivia"]
_SPLIT_ALIASES: dict[str, list[str]] = {"val": ["val", "valid"]}

EVAL_KEYS = [
    "mean_iou", "iou_water", "f1_water",
    "precision_water", "recall_water",
    "topo_violation_fraction",
]

SWEEP_VARIANTS = [
    # (tag, cond, lambda)
    ("segman_n100_topo_lambda1p0_seed0",              "dice_ce_topo",              1.0),
    ("segman_n100_topo_dem_shuffled_lambda1p0_seed0", "dice_ce_topo_dem_shuffled", 1.0),
    ("segman_n100_topo_lambda2p0_seed0",              "dice_ce_topo",              2.0),
    ("segman_n100_topo_dem_shuffled_lambda2p0_seed0", "dice_ce_topo_dem_shuffled", 2.0),
    ("segman_n100_topo_lambda4p0_seed0",              "dice_ce_topo",              4.0),
    ("segman_n100_topo_dem_shuffled_lambda4p0_seed0", "dice_ce_topo_dem_shuffled", 4.0),
]


def safe_float(v: Any) -> float | None:
    if v is None: return None
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def fmt(v: float | None, p: int = 5) -> str:
    return "N/A" if v is None else f"{v:.{p}f}"


def fmt_pm(mu: float | None, sd: float | None) -> str:
    if mu is None: return "N/A"
    return f"{mu:.5f}" if sd is None else f"{mu:.5f} +/- {sd:.5f}"


def load_summary(tag: str) -> dict[str, Any] | None:
    path = RUNS_ROOT / tag / "metrics" / f"{tag}_summary.json"
    if not path.exists(): return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if d.get("status") == "done" else None
    except Exception as e:
        print(f"  WARN: {path}: {e}", file=sys.stderr)
        return None


def extract_metrics(summary: dict, tag: str, cond: str, lam: float) -> dict[str, Any]:
    row: dict[str, Any] = {
        "tag": tag, "condition": cond, "lambda_topo": lam,
        "status": summary.get("status"),
        "best_epoch": summary.get("best_epoch"),
        "last_epoch": summary.get("last_epoch"),
        "best_val_miou": safe_float(summary.get("best_validation_miou")),
    }
    evals = summary.get("evaluations") or {}
    for split in SPLITS:
        aliases = _SPLIT_ALIASES.get(split, [split]) + [f"final_{split}"]
        sd = next((evals[a] for a in aliases if evals.get(a)), {}) or {}
        for k in EVAL_KEYS:
            row[f"{split}_{k}"] = safe_float(sd.get(k))

        # Per-epoch best-epoch loss info (from epoch CSV if available)
        row[f"{split}_train_loss_total"]  = None
        row[f"{split}_train_loss_topo"]   = None
        row[f"{split}_train_loss_dice_ce"] = None

    # Try to pull per-epoch train losses at best epoch from CSV
    csv_path = RUNS_ROOT / tag / "metrics" / "training_epoch_metrics.csv"
    be = row.get("best_epoch")
    if csv_path.exists() and be is not None:
        try:
            rows = list(csv_mod.DictReader(csv_path.open(encoding="utf-8")))
            best_row = next((r for r in rows if int(float(r.get("epoch", -1))) == int(be)), None)
            if best_row:
                row["train_loss_total"]     = safe_float(best_row.get("train_loss_total"))
                row["train_loss_topo"]      = safe_float(best_row.get("train_loss_topo"))
                row["train_loss_dice"]      = safe_float(best_row.get("train_loss_dice"))
                row["train_loss_ce"]        = safe_float(best_row.get("train_loss_ce"))
                row["lambda_topo_epoch"]    = safe_float(best_row.get("lambda_topo_epoch"))
        except Exception as e:
            print(f"  WARN: CSV parse error for {tag}: {e}", file=sys.stderr)

    # Effective topo contribution at best epoch
    lt = row.get("train_loss_topo")
    lt_ep = row.get("lambda_topo_epoch")
    ld = row.get("train_loss_dice")
    lc = row.get("train_loss_ce")
    if lt is not None and lt_ep is not None:
        eff_lam = lt_ep if lt_ep is not None else lam
        dce_sum = (ld or 0.0) + (lc or 0.0)
        if dce_sum > 1e-9:
            row["effective_topo_contribution"] = (eff_lam * lt) / dce_sum
        else:
            row["effective_topo_contribution"] = None
    else:
        row["effective_topo_contribution"] = None

    return row


def _safe_json(obj: Any) -> Any:
    if isinstance(obj, dict):  return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_safe_json(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj): return None
    return obj


def main() -> None:
    print(f"Runs root: {RUNS_ROOT}")
    print(f"Baseline: {BASELINE_TAG}\n")

    # ── Baseline ──────────────────────────────────────────────────────────────
    baseline_summary = load_summary(BASELINE_TAG)
    if baseline_summary is None:
        print(f"WARNING: baseline {BASELINE_TAG} not found/complete — deltas will be N/A")
    baseline = extract_metrics(baseline_summary, BASELINE_TAG, "dice_ce", 0.0) if baseline_summary else {}

    def baseline_val(split: str, metric: str) -> float | None:
        return baseline.get(f"{split}_{metric}")

    # ── Load sweep runs ───────────────────────────────────────────────────────
    all_rows: list[dict] = []
    missing: list[str] = []

    for tag, cond, lam in SWEEP_VARIANTS:
        s = load_summary(tag)
        if s is None:
            missing.append(tag); print(f"  MISSING: {tag}")
            continue
        row = extract_metrics(s, tag, cond, lam)
        all_rows.append(row)
        vm = row.get("val_mean_iou"); tm = row.get("test_mean_iou")
        print(f"  OK  {tag}  val_mIoU={fmt(vm)}  test_mIoU={fmt(tm)}")

    print(f"\nLoaded {len(all_rows)}/6 runs, {len(missing)} missing.")

    # ── Compute paired deltas against baseline ────────────────────────────────
    for row in all_rows:
        for split in SPLITS:
            bv = baseline_val(split, "mean_iou")
            rv = row.get(f"{split}_mean_iou")
            row[f"{split}_delta_vs_baseline"] = (rv - bv) if (rv is not None and bv is not None) else None

    # delta Topo real − Shuffled (same lambda)
    real_rows: dict[float, dict] = {}
    shuf_rows: dict[float, dict] = {}
    for row in all_rows:
        lam = row["lambda_topo"]
        if row["condition"] == "dice_ce_topo":           real_rows[lam] = row
        if row["condition"] == "dice_ce_topo_dem_shuffled": shuf_rows[lam] = row

    deltas_by_lambda: list[dict] = []
    for lam in [1.0, 2.0, 4.0]:
        r = real_rows.get(lam); s = shuf_rows.get(lam)
        d: dict[str, Any] = {"lambda_topo": lam}
        for split in SPLITS:
            rv = r.get(f"{split}_mean_iou") if r else None
            sv = s.get(f"{split}_mean_iou") if s else None
            bv = baseline_val(split, "mean_iou")
            d[f"{split}_topo_real"]              = rv
            d[f"{split}_topo_shuffled"]          = sv
            d[f"{split}_real_minus_baseline"]    = (rv - bv) if (rv is not None and bv is not None) else None
            d[f"{split}_shuffled_minus_baseline"]= (sv - bv) if (sv is not None and bv is not None) else None
            d[f"{split}_real_minus_shuffled"]    = (rv - sv) if (rv is not None and sv is not None) else None
        deltas_by_lambda.append(d)

    # ── CSV ───────────────────────────────────────────────────────────────────
    reports = REPO_ROOT / "reports"; docs = REPO_ROOT / "docs"
    reports.mkdir(parents=True, exist_ok=True); docs.mkdir(parents=True, exist_ok=True)

    metric_cols = [f"{s}_{k}" for s in SPLITS for k in EVAL_KEYS] + \
                  [f"{s}_delta_vs_baseline" for s in SPLITS] + \
                  ["train_loss_total", "train_loss_topo", "train_loss_dice", "train_loss_ce",
                   "lambda_topo_epoch", "effective_topo_contribution"]
    fixed = ["condition", "lambda_topo", "tag", "status", "best_epoch", "last_epoch",
             "best_val_miou"] + metric_cols

    csv_path = reports / "segman_n100_lambda_sweep_seed0_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv_mod.DictWriter(fh, fieldnames=fixed, extrasaction="ignore")
        w.writeheader()
        for row in sorted(all_rows, key=lambda r: (r["lambda_topo"], r["condition"])):
            w.writerow({k: ("" if row.get(k) is None else row[k]) for k in fixed})
    print(f"Wrote CSV: {csv_path}")

    # ── JSON ──────────────────────────────────────────────────────────────────
    payload = {
        "experiment": "SEGMAN_cvpr2025_n100_lambda_sweep_seed0",
        "model": "SegMAN-S", "n_train": 100, "seed": 0,
        "lambdas_tested": [1.0, 2.0, 4.0],
        "baseline_tag": BASELINE_TAG,
        "baseline_test_miou": baseline_val("test", "mean_iou"),
        "baseline_bolivia_miou": baseline_val("bolivia", "mean_iou"),
        "n_loaded": len(all_rows), "n_missing": len(missing),
        "missing_runs": missing,
        "per_run": all_rows,
        "deltas_by_lambda": deltas_by_lambda,
    }
    json_path = reports / "segman_n100_lambda_sweep_seed0_results.json"
    json_path.write_text(json.dumps(_safe_json(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote JSON: {json_path}")

    # ── Markdown summary ──────────────────────────────────────────────────────
    def _pm(row: dict | None, split: str, metric: str) -> str:
        return fmt(row.get(f"{split}_{metric}") if row else None)

    btest  = fmt(baseline_val("test", "mean_iou"))
    bval   = fmt(baseline_val("val", "mean_iou"))
    bbol   = fmt(baseline_val("bolivia", "mean_iou"))

    lines = [
        "# SegMAN-S N=100 Lambda Stress Test — seed0 Results",
        "",
        f"**N_train**: 100  |  **Seed**: 0  |  **Lambdas tested**: 1.0, 2.0, 4.0",
        f"**Baseline** (`segman_n100_dice_ce_seed0`): val={bval}  test={btest}  bolivia={bbol}",
        "",
        "## Scientific Question",
        "",
        "> Does a stronger lambda_topo reveal a real DEM effect that was absent at lambda=0.5?",
        "> Key criterion: **Topo real > Topo shuffled** at some lambda, consistently.",
        "",
    ]
    if missing:
        lines += ["**Missing runs** (excluded):", ""]
        for m in missing: lines.append(f"- {m}")
        lines.append("")

    for split in SPLITS:
        lines += [
            f"## {split.capitalize()} mIoU by Lambda",
            "",
            "| Lambda | Topo real | Topo shuffled | Real−Baseline | Shuf−Baseline | Real−Shuffled |",
            "|--------|-----------|---------------|---------------|---------------|---------------|",
        ]
        for d in deltas_by_lambda:
            lam = d["lambda_topo"]
            tr   = fmt(d.get(f"{split}_topo_real"))
            ts   = fmt(d.get(f"{split}_topo_shuffled"))
            drb  = fmt(d.get(f"{split}_real_minus_baseline"), 5)
            dsb  = fmt(d.get(f"{split}_shuffled_minus_baseline"), 5)
            drs  = fmt(d.get(f"{split}_real_minus_shuffled"), 5)
            lines.append(f"| {lam} | {tr} | {ts} | {drb} | {dsb} | {drs} |")
        lines.append("")

    lines += [
        "## Loss at Best Epoch (train, by lambda)",
        "",
        "| Lambda | Condition | loss_total | loss_topo | eff_topo_contribution | best_ep |",
        "|--------|-----------|------------|-----------|----------------------|---------|",
    ]
    for row in sorted(all_rows, key=lambda r: (r["lambda_topo"], r["condition"])):
        lam  = row["lambda_topo"]
        cond = row["condition"]
        lt   = fmt(row.get("train_loss_total"), 5)
        ltp  = fmt(row.get("train_loss_topo"), 5)
        etc  = fmt(row.get("effective_topo_contribution"), 4)
        be   = row.get("best_epoch", "?")
        lines.append(f"| {lam} | {cond} | {lt} | {ltp} | {etc} | {be} |")
    lines.append("")

    lines += [
        "## Interpretation",
        "",
        "- **Real−Shuffled > 0**: real DEM provides benefit over random noise at this lambda.",
        "- **Real−Shuffled ≤ 0**: shuffled control is equal or better → DEM content not captured.",
        "- **Real−Baseline < 0**: lambda too strong; topo loss hurts mIoU.",
        "",
        "Interpretation by lambda:",
        "",
    ]
    for d in deltas_by_lambda:
        lam = d["lambda_topo"]
        drs = d.get("test_real_minus_shuffled")
        drb = d.get("test_real_minus_baseline")
        if drs is None: verdict = "N/A (missing runs)"
        elif drs > 0.005:  verdict = f"Real > Shuffled (+{drs:.4f}) — DEM effect present at this lambda"
        elif drs < -0.005: verdict = f"Shuffled > Real ({drs:+.4f}) — DEM content still not captured"
        else:              verdict = f"No clear signal (Real−Shuffled={drs:+.4f}, within noise)"
        if drb is not None and drb < -0.01:
            verdict += " [WARN: lambda may be too strong, degrades mIoU vs baseline]"
        lines.append(f"- **lambda={lam}**: {verdict}")

    lines += [
        "",
        "## Notes",
        "",
        "- Baseline: `segman_n100_dice_ce_seed0` (lambda_topo=0.0 throughout).",
        "- This is a single-seed stress test. Multi-seed replication needed before final conclusions.",
        "- DEM is NOT a model input. Used only in topographic loss and eval topo metrics.",
        "- Aggregation: `experiments_cvpr/segman/aggregate_segman_n100_lambda_sweep.py`",
    ]

    md_path = docs / "segman_n100_lambda_sweep_seed0_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote markdown: {md_path}")

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n=== Key test mIoU deltas (Real − Shuffled) by lambda ===")
    for d in deltas_by_lambda:
        drs = d.get("test_real_minus_shuffled")
        drb = d.get("test_real_minus_baseline")
        print(f"  lambda={d['lambda_topo']}: Real−Shuffled={fmt(drs,5)}  Real−Baseline={fmt(drb,5)}")


if __name__ == "__main__":
    main()
