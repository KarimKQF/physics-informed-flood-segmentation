"""
Aggregate SegMAN N=100 diagnostic results.

Reads the 9 completed runs (3 conditions x 3 seeds), computes per-condition
mean ± std, and produces paired deltas for the key scientific comparisons:
  1. Topo real - Dice+CE
  2. Shuffled  - Topo real
  3. Shuffled  - Dice+CE

Main scientific criterion: Topo real > Topo shuffled consistently across seeds.

CPU-only. Does not load checkpoints or use GPU.

Outputs:
  reports/segman_n100_diagnostic_results.csv
  reports/segman_n100_diagnostic_results.json
  docs/segman_n100_diagnostic_summary.md

Usage:
  python experiments_cvpr/segman/aggregate_segman_n100_diagnostic.py
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

SEEDS = [0, 1, 2]
CONDITIONS = ["dice_ce", "dice_ce_topo", "dice_ce_topo_dem_shuffled"]
CONDITION_LABELS = {
    "dice_ce":                      "Dice+CE",
    "dice_ce_topo":                 "Dice+CE+Topo",
    "dice_ce_topo_dem_shuffled":    "Dice+CE+Topo+Shuffled",
}
SPLITS = ["val", "test", "bolivia"]
_SPLIT_ALIASES: dict[str, list[str]] = {"val": ["val", "valid"]}

EVAL_KEYS = [
    "mean_iou", "iou_water", "f1_water",
    "precision_water", "recall_water", "accuracy",
    "topo_violation_fraction",
]


def run_tag(cond: str, seed: int) -> str:
    return f"segman_n100_{cond}_seed{seed}"


def safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


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


def extract_metrics(summary: dict[str, Any], tag: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "tag":          tag,
        "status":       summary.get("status"),
        "best_epoch":   summary.get("best_epoch"),
        "last_epoch":   summary.get("last_epoch"),
        "best_val_miou": safe_float(summary.get("best_validation_miou")),
    }
    evals = summary.get("evaluations") or {}
    for split in SPLITS:
        aliases = _SPLIT_ALIASES.get(split, [split]) + [f"final_{split}"]
        split_data = next((evals[a] for a in aliases if evals.get(a)), {}) or {}
        for k in EVAL_KEYS:
            row[f"{split}_{k}"] = safe_float(split_data.get(k))
    return row


def mean_std(vals: list[float | None]) -> tuple[float | None, float | None]:
    clean = [v for v in vals if v is not None]
    if not clean:
        return None, None
    mu = sum(clean) / len(clean)
    if len(clean) == 1:
        return mu, None
    sd = math.sqrt(sum((x - mu) ** 2 for x in clean) / (len(clean) - 1))
    return mu, sd


def fmt(v: float | None, prec: int = 4) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{prec}f}"


def fmt_pm(mu: float | None, sd: float | None) -> str:
    if mu is None:
        return "N/A"
    if sd is None:
        return f"{mu:.4f}"
    return f"{mu:.4f} +/- {sd:.4f}"


def main() -> None:
    print(f"Runs root: {RUNS_ROOT}")
    print(f"Seeds: {SEEDS} | Conditions: {CONDITIONS}\n")

    # ── Collect rows ──────────────────────────────────────────────────────
    all_rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for seed in SEEDS:
        for cond in CONDITIONS:
            tag = run_tag(cond, seed)
            summary = load_summary(tag)
            if summary is None:
                missing.append(tag)
                print(f"  MISSING/INCOMPLETE: {tag}")
                continue
            row = extract_metrics(summary, tag)
            row["seed"]            = seed
            row["condition"]       = cond
            row["condition_label"] = CONDITION_LABELS[cond]
            all_rows.append(row)
            bv = row.get("val_mean_iou")
            bt = row.get("test_mean_iou")
            print(f"  OK  {tag}  val_mIoU={fmt(bv)}  test_mIoU={fmt(bt)}")

    print(f"\nLoaded {len(all_rows)}/9 runs, {len(missing)} missing.")
    if missing:
        print("Missing:", missing)

    # ── Per-run CSV ───────────────────────────────────────────────────────
    reports = REPO_ROOT / "reports"
    docs    = REPO_ROOT / "docs"
    reports.mkdir(parents=True, exist_ok=True)
    docs.mkdir(parents=True, exist_ok=True)

    metric_cols = [f"{s}_{k}" for s in SPLITS for k in EVAL_KEYS]
    fixed_cols  = ["condition", "seed", "tag", "status", "best_epoch",
                   "last_epoch", "best_val_miou"] + metric_cols
    csv_path = reports / "segman_n100_diagnostic_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv_mod.DictWriter(fh, fieldnames=fixed_cols, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(all_rows, key=lambda r: (r["condition"], r["seed"])):
            writer.writerow({k: ("" if row.get(k) is None else row[k]) for k in fixed_cols})
    print(f"Wrote CSV: {csv_path}")

    # ── Aggregate per condition ───────────────────────────────────────────
    aggregate: dict[str, Any] = {}
    for cond in CONDITIONS:
        crows = [r for r in all_rows if r["condition"] == cond]
        agg: dict[str, Any] = {
            "condition":       cond,
            "condition_label": CONDITION_LABELS[cond],
            "n_seeds":         len(crows),
            "seeds_done":      sorted(r["seed"] for r in crows),
            "missing_seeds":   [s for s in SEEDS if s not in [r["seed"] for r in crows]],
        }
        for col in metric_cols + ["best_val_miou", "best_epoch"]:
            mu, sd = mean_std([r.get(col) for r in crows])
            agg[f"{col}_mean"] = mu
            agg[f"{col}_std"]  = sd
        aggregate[cond] = agg

    # ── Per-seed paired deltas ────────────────────────────────────────────
    def get_val(cond: str, seed: int, metric: str) -> float | None:
        r = next((x for x in all_rows if x["condition"] == cond and x["seed"] == seed), None)
        return r.get(metric) if r else None

    deltas_test: list[dict] = []
    for seed in SEEDS:
        d_topo_vs_dce    = None
        d_shuf_vs_topo   = None
        d_shuf_vs_dce    = None
        v_dce   = get_val("dice_ce",                   seed, "test_mean_iou")
        v_topo  = get_val("dice_ce_topo",              seed, "test_mean_iou")
        v_shuf  = get_val("dice_ce_topo_dem_shuffled", seed, "test_mean_iou")
        if v_topo is not None and v_dce is not None:
            d_topo_vs_dce  = v_topo - v_dce
        if v_shuf is not None and v_topo is not None:
            d_shuf_vs_topo = v_shuf - v_topo
        if v_shuf is not None and v_dce is not None:
            d_shuf_vs_dce  = v_shuf - v_dce
        deltas_test.append({
            "seed": seed,
            "test_miou_dice_ce":    v_dce,
            "test_miou_topo_real":  v_topo,
            "test_miou_topo_shuf":  v_shuf,
            "delta_topo_real_minus_dce":    d_topo_vs_dce,
            "delta_shuffled_minus_topo":    d_shuf_vs_topo,
            "delta_shuffled_minus_dce":     d_shuf_vs_dce,
        })

    # ── Global delta means ────────────────────────────────────────────────
    def _mean_deltas(key: str) -> tuple[float | None, float | None]:
        vals = [d[key] for d in deltas_test if d[key] is not None]
        return mean_std(vals)

    mu_topo_dce,  sd_topo_dce  = _mean_deltas("delta_topo_real_minus_dce")
    mu_shuf_topo, sd_shuf_topo = _mean_deltas("delta_shuffled_minus_topo")
    mu_shuf_dce,  sd_shuf_dce  = _mean_deltas("delta_shuffled_minus_dce")

    # ── JSON ──────────────────────────────────────────────────────────────
    def _safe(obj: Any) -> Any:
        if isinstance(obj, dict):   return {k: _safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [_safe(v) for v in obj]
        if isinstance(obj, float) and not math.isfinite(obj): return None
        return obj

    payload = {
        "experiment": "SEGMAN_cvpr2025_n100_diagnostic",
        "model": "SegMAN-S", "n_train": 100,
        "seeds": SEEDS, "conditions": CONDITIONS,
        "n_loaded": len(all_rows), "n_missing": len(missing),
        "missing_runs": missing,
        "per_run": all_rows,
        "aggregate_by_condition": aggregate,
        "per_seed_test_deltas": deltas_test,
        "global_test_deltas": {
            "topo_real_minus_dice_ce":    {"mean": mu_topo_dce,  "std": sd_topo_dce},
            "shuffled_minus_topo_real":   {"mean": mu_shuf_topo, "std": sd_shuf_topo},
            "shuffled_minus_dice_ce":     {"mean": mu_shuf_dce,  "std": sd_shuf_dce},
        },
    }
    json_path = reports / "segman_n100_diagnostic_results.json"
    json_path.write_text(
        json.dumps(_safe(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(f"Wrote JSON: {json_path}")

    # ── Markdown summary ──────────────────────────────────────────────────
    def _pm(cond: str, split: str, metric: str) -> str:
        agg = aggregate.get(cond, {})
        mu = agg.get(f"{split}_{metric}_mean")
        sd = agg.get(f"{split}_{metric}_std")
        return fmt_pm(mu, sd)

    lines = [
        "# SegMAN-S N=100 Diagnostic — Aggregated Results",
        "",
        "**N_train**: 100  |  **Seeds**: 0, 1, 2  |  **Conditions**: Dice+CE / Topo real / Topo shuffled",
        "",
        "## Key Scientific Question",
        "",
        "> Does the real DEM topographic loss beat the shuffled-DEM control at N=100?",
        "> Main criterion: **Topo real > Topo shuffled consistently across seeds.**",
        "",
    ]
    if missing:
        lines += ["**Missing runs** (excluded from stats):", ""]
        for m in missing: lines.append(f"- {m}")
        lines.append("")

    for split in SPLITS:
        lines += [
            f"## {split.capitalize()} Split",
            "",
            "| Condition | mIoU | IoU_water | F1_water | Prec_water | Rec_water | Topo_viol |",
            "|-----------|------|-----------|----------|------------|-----------|-----------|",
        ]
        for cond in CONDITIONS:
            lines.append(
                f"| {CONDITION_LABELS[cond]} "
                f"| {_pm(cond, split, 'mean_iou')} "
                f"| {_pm(cond, split, 'iou_water')} "
                f"| {_pm(cond, split, 'f1_water')} "
                f"| {_pm(cond, split, 'precision_water')} "
                f"| {_pm(cond, split, 'recall_water')} "
                f"| {_pm(cond, split, 'topo_violation_fraction')} |"
            )
        lines.append("")

    # Per-seed detail
    lines += [
        "## Per-Seed Test mIoU",
        "",
        "| Condition | seed0 | seed1 | seed2 |",
        "|-----------|-------|-------|-------|",
    ]
    for cond in CONDITIONS:
        cells = [fmt(get_val(cond, s, "test_mean_iou")) for s in SEEDS]
        lines.append(f"| {CONDITION_LABELS[cond]} | " + " | ".join(cells) + " |")
    lines.append("")

    # Paired deltas
    lines += [
        "## Paired Deltas (test mIoU)",
        "",
        "| Comparison | " + " | ".join(f"seed{s}" for s in SEEDS) + " | Mean +/- Std |",
        "|------------|" + "|".join("-------" for _ in SEEDS) + "|-------------|",
    ]
    for key, label, (mu, sd) in [
        ("delta_topo_real_minus_dce",  "Topo real - Dice+CE",    (mu_topo_dce,  sd_topo_dce)),
        ("delta_shuffled_minus_topo",  "Shuffled  - Topo real",  (mu_shuf_topo, sd_shuf_topo)),
        ("delta_shuffled_minus_dce",   "Shuffled  - Dice+CE",    (mu_shuf_dce,  sd_shuf_dce)),
    ]:
        per_seed = [fmt(d[key], 5) for d in deltas_test]
        lines.append(
            f"| {label} | " + " | ".join(per_seed) + f" | {fmt_pm(mu, sd)} |"
        )
    lines.append("")

    # Interpretation
    def _sign(mu, sd, key_label):
        if mu is None: return f"  - {key_label}: N/A (runs missing)"
        trend = "positive" if mu > 0.005 else ("negative" if mu < -0.005 else "neutral (noise)")
        sd_str = f" +/- {sd:.4f}" if sd is not None else ""
        return f"  - {key_label}: {mu:+.4f}{sd_str}  ->  **{trend}**"

    lines += [
        "## Interpretation",
        "",
        "Compared to N=50 (where std dominated signal):",
        "",
        _sign(mu_topo_dce,  sd_topo_dce,  "Topo real - Dice+CE:    does topo help over baseline?"),
        _sign(mu_shuf_topo, sd_shuf_topo, "Shuffled  - Topo real:  is shuffled >= real? (null effect)"),
        _sign(mu_shuf_dce,  sd_shuf_dce,  "Shuffled  - Dice+CE:    does any topo term help?"),
        "",
        "**DEM effect established** if: Topo real - Dice+CE > 0 AND Shuffled - Topo real <= 0 consistently.",
        "",
        "## Notes",
        "",
        "- N=100 is a strict superset of N=50 (same seed): positions 0-49 in shuffled order are identical.",
        "- DEM is NOT a model input. Used only in topographic loss during training.",
        "- Eval-time topo metrics always use the real DEM for physical coherence measurement.",
        "- Aggregation script: `experiments_cvpr/segman/aggregate_segman_n100_diagnostic.py`",
    ]

    md_path = docs / "segman_n100_diagnostic_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote markdown: {md_path}")

    # ── Print quick summary ───────────────────────────────────────────────
    print("\n=== Key deltas (test mIoU) ===")
    print(f"  Topo real - Dice+CE :  {fmt_pm(mu_topo_dce,  sd_topo_dce)}")
    print(f"  Shuffled  - Topo real: {fmt_pm(mu_shuf_topo, sd_shuf_topo)}")
    print(f"  Shuffled  - Dice+CE :  {fmt_pm(mu_shuf_dce,  sd_shuf_dce)}")
    if mu_topo_dce is not None and mu_shuf_topo is not None:
        real_beats_shuf = mu_shuf_topo < -0.001
        print(f"\n  => Real DEM beats shuffled? {'YES' if real_beats_shuf else 'NO (within noise)'}")


if __name__ == "__main__":
    main()
