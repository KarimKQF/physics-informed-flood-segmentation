"""
Post-hoc diagnostic for SegMAN N=100 D8 downstream loss, seed0.
Parts B-I: per-epoch traces, matched-epoch analysis, D8 activation,
full eval metrics, DEM geometry, interpretation, and output files.

No training launched. No model/loss code modified.
"""
from __future__ import annotations

import csv as csv_mod
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import rasterio
    HAS_RIO = True
except ImportError:
    HAS_RIO = False

# ── Constants ──────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parents[2]
RUNS_ROOT  = Path("E:/flood_research/experiments/segman/runs")
DEM_ROOT   = Path("E:/flood_research/data/derived/sen1floods11_topography/dem_aligned")

VAL_MANIFEST   = Path("E:/flood_research/experiments/terramind_baseline/runs/"
                      "step5e_tiny_unetdecoder_baseline/manifests/flood_valid_step5e_filtered.txt")
TRAIN_MANIFEST = (REPO_ROOT / "manifests/terramind_baseline/low_data_multiseed_n100"
                  / "flood_train_low_data_n100_seed0.txt")
DEM_PATTERN    = "{split}_{tile_id}_copernicus_glo30_dem_aligned.tif"

TAGS = {
    "baseline": "segman_n100_dice_ce_seed0",
    "d8_real":  "segman_n100_d8_lambda1p0_seed0",
    "d8_shuf":  "segman_n100_d8_dem_shuffled_lambda1p0_seed0",
}

SPLITS         = ["valid", "test", "bolivia"]
MATCHED_EPOCHS = [22, 31, 34]

D8_S0  = 1.0
D8_TAU = 0.05
_D8_DY   = (-1, -1, -1,  0,  0,  1,  1,  1)
_D8_DX   = (-1,  0,  1, -1,  1, -1,  0,  1)
_D8_DIST = (math.sqrt(2), 1., math.sqrt(2), 1., 1., math.sqrt(2), 1., math.sqrt(2))

# ── Formatting helpers ─────────────────────────────────────────────────────────
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

def _int(v) -> str:
    return "N/A" if v is None else f"{int(v):,}"

# ── Data loading ───────────────────────────────────────────────────────────────
def load_epoch_csv(tag: str) -> list[dict]:
    path = RUNS_ROOT / tag / "metrics" / "training_epoch_metrics.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv_mod.DictReader(f))

def load_summary(tag: str) -> Optional[dict]:
    path = RUNS_ROOT / tag / "metrics" / f"{tag}_summary.json"
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if d.get("status") == "done" else None
    except Exception as e:
        print(f"WARN: {path}: {e}", file=sys.stderr)
        return None

# ── Per-epoch trace computation ────────────────────────────────────────────────
def compute_epoch_traces(label: str, rows: list[dict]) -> list[dict]:
    traces = []
    for r in rows:
        lam  = safe_float(r.get("lambda_topo_epoch"))
        lt   = safe_float(r.get("train_loss_topo"))
        ld   = safe_float(r.get("train_loss_dice"))
        lce  = safe_float(r.get("train_loss_ce"))
        ep   = safe_float(r.get("epoch"))

        if lam is not None and lt is not None and ld is not None and lce is not None and (ld + lce) > 0:
            eff_d8 = (lam * lt) / (ld + lce)
        else:
            eff_d8 = None

        traces.append({
            "tag":                  label,
            "epoch":                int(ep) if ep is not None else None,
            "lambda_epoch":         lam,
            "train_loss_total":     safe_float(r.get("train_loss_total")),
            "train_loss_dice":      ld,
            "train_loss_ce":        lce,
            "train_loss_d8":        lt,
            "train_miou":           safe_float(r.get("train_miou")),
            "val_miou":             safe_float(r.get("val_miou")),
            "val_iou_water":        safe_float(r.get("val_iou_water")),
            "val_f1_water":         safe_float(r.get("val_f1_water")),
            "val_precision":        safe_float(r.get("val_precision_water")),
            "val_recall":           safe_float(r.get("val_recall_water")),
            "val_topo_vf":          safe_float(r.get("val_topo_violation_fraction")),
            "val_pred_px":          safe_float(r.get("val_water_pred_pixels")),
            "eff_d8_contribution":  eff_d8,
            "learning_rate":        safe_float(r.get("learning_rate")),
            "is_best":              r.get("improved", "").strip().lower() == "true",
        })
    return traces

def get_epoch_row(traces: list[dict], epoch: int) -> Optional[dict]:
    return next((t for t in traces if t["epoch"] == epoch), None)

# ── Full eval metrics from summary JSON ───────────────────────────────────────
def compute_eval_metrics(summary: dict, split: str) -> dict:
    evals   = summary.get("evaluations") or {}
    aliases = [split, "valid" if split == "val" else split]
    sd      = next((evals[a] for a in aliases if evals.get(a)), {}) or {}
    if not sd:
        return {}

    tp = safe_float(sd.get("tp"))
    fp = safe_float(sd.get("fp"))
    fn = safe_float(sd.get("fn"))
    tn = safe_float(sd.get("tn"))
    vp = safe_float(sd.get("valid_pixel_count"))
    sw = safe_float(sd.get("support_water"))
    wp = safe_float(sd.get("water_pred_pixels"))

    def _div(a, b):
        return (a / b) if (a is not None and b is not None and b > 0) else None

    recall      = _div(tp, tp + fn if tp and fn is not None else None)
    precision   = _div(tp, tp + fp if tp and fp is not None else None)
    specificity = _div(tn, tn + fp if tn is not None and fp is not None else None)
    fpr         = _div(fp, fp + tn if fp is not None and tn is not None else None)
    fnr         = _div(fn, fn + tp if fn is not None and tp is not None else None)

    iou_w  = _div(tp, (tp + fp + fn) if tp is not None and fp is not None and fn is not None else None)
    iou_bg = _div(tn, (tn + fp + fn) if tn is not None and fp is not None and fn is not None else None)
    miou   = (iou_w + iou_bg) / 2 if iou_w is not None and iou_bg is not None else None

    f1_w  = _div(2 * tp if tp else None, (2 * tp + fp + fn) if tp is not None and fp is not None and fn is not None else None)
    f1_bg = _div(2 * tn if tn else None, (2 * tn + fn + fp) if tn is not None and fn is not None and fp is not None else None)
    macro_f1 = (f1_w + f1_bg) / 2 if f1_w is not None and f1_bg is not None else None

    accuracy    = _div(tp + tn if tp is not None and tn is not None else None, vp)
    bal_acc     = (recall + specificity) / 2 if recall is not None and specificity is not None else None

    gt_ratio   = _div(sw, vp)
    pred_ratio = _div(wp, vp)
    ratio_err  = abs(pred_ratio - gt_ratio) if pred_ratio is not None and gt_ratio is not None else None
    pred_gt_r  = _div(pred_ratio, gt_ratio)

    return {
        "split":                 split,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "valid_pixel_count":     vp,
        "support_water":         sw,
        "water_pred_pixels":     wp,
        "accuracy":              accuracy,
        "balanced_accuracy":     bal_acc,
        "iou_water":             iou_w,
        "iou_background":        iou_bg,
        "mean_iou":              miou,
        "f1_water":              f1_w,
        "f1_background":         f1_bg,
        "macro_f1":              macro_f1,
        "precision_water":       precision,
        "recall_water":          recall,
        "specificity":           specificity,
        "fpr":                   fpr,
        "fnr":                   fnr,
        "gt_water_ratio":        gt_ratio,
        "pred_water_ratio":      pred_ratio,
        "water_ratio_error":     ratio_err,
        "pred_gt_ratio":         pred_gt_r,
        "topo_violation_fraction":      safe_float(sd.get("topo_violation_fraction")),
        "topo_descending_pairs": safe_float(sd.get("topo_descending_pair_count")),
        "topo_violation_pairs":  safe_float(sd.get("topo_violation_pair_count")),
    }

# ── D8 activation stats from per-epoch data ───────────────────────────────────
def analyze_d8_activation(traces_by_tag: dict) -> dict:
    results = {}
    for label, traces in traces_by_tag.items():
        eff = [t["eff_d8_contribution"] for t in traces if t["eff_d8_contribution"] is not None]
        lt  = [t["train_loss_d8"]       for t in traces if t["train_loss_d8"] is not None]
        lam = [t["lambda_epoch"]        for t in traces if t["lambda_epoch"] is not None]

        if not eff:
            results[label] = {"error": "no effective D8 data"}
            continue

        max_eff = max(eff)
        max_ep  = next((t["epoch"] for t in traces if t["eff_d8_contribution"] == max_eff), None)

        results[label] = {
            "max_eff_d8":          max_eff,
            "max_eff_d8_epoch":    max_ep,
            "mean_eff_d8":         float(np.mean(eff)),
            "eff_d8_at_best_ep":   None,
            "ever_above_0p1pct":   any(e > 0.001 for e in eff),
            "ever_above_1pct":     any(e > 0.01  for e in eff),
            "ever_above_3pct":     any(e > 0.03  for e in eff),
            "max_lambda":          max(lam) if lam else None,
            "max_raw_d8_loss":     max(lt)  if lt  else None,
            "all_eff_d8":          eff,
        }
    return results

# ── DEM geometry (Part F) ──────────────────────────────────────────────────────
def compute_d8_stats_tile(dem: np.ndarray, s0: float = D8_S0) -> Optional[dict]:
    if dem is None:
        return None
    H, W   = dem.shape
    valid  = ~np.isnan(dem)
    dem_f  = np.where(valid, dem.astype(np.float32), -1e9)
    pad    = np.pad(dem_f, 1, mode="constant", constant_values=1e9)

    slopes = []
    for dy, dx, dist in zip(_D8_DY, _D8_DX, _D8_DIST):
        nbr   = pad[dy + 1: dy + 1 + H, dx + 1: dx + 1 + W]
        slope = (dem_f - nbr) / dist
        slopes.append(slope)

    d8_slope = np.stack(slopes, axis=0).max(axis=0)
    drop     = np.clip(d8_slope, 0.0, None)
    w        = np.clip(drop / s0, 0.0, 1.0)

    drop_v = drop[valid]
    w_v    = w[valid]
    if w_v.size == 0:
        return None

    n = w_v.size
    return {
        "n_pixels":        n,
        "frac_w0":         float(np.sum(w_v == 0) / n),
        "frac_w_pos":      float(np.sum(w_v >  0) / n),
        "frac_w_gt01":     float(np.sum(w_v > 0.1) / n),
        "frac_w_gt05":     float(np.sum(w_v > 0.5) / n),
        "frac_w_eq1":      float(np.sum(w_v >= 1.0) / n),
        "frac_0_to_01":    float(np.sum((w_v > 0) & (w_v <= 0.1)) / n),
        "mean_drop_m":     float(drop_v.mean()),
        "median_drop_m":   float(np.median(drop_v)),
        "mean_w":          float(w_v.mean()),
        "max_drop_m":      float(drop_v.max()),
    }

def load_dem_tile(split_prefix: str, tile_id: str) -> Optional[np.ndarray]:
    fname = DEM_PATTERN.format(split=split_prefix, tile_id=tile_id)
    path  = DEM_ROOT / fname
    if not path.exists():
        return None
    try:
        with rasterio.open(path) as ds:
            arr = ds.read(1).astype(np.float32)
            nd  = ds.nodata
            if nd is not None:
                arr[arr == nd] = np.nan
        return arr
    except Exception as e:
        warnings.warn(f"Could not read {path}: {e}")
        return None

def compute_dem_geometry(split_prefix: str, manifest: Path) -> dict:
    if not HAS_RIO:
        return {"error": "rasterio not available"}
    if not manifest.exists():
        return {"error": f"manifest not found: {manifest}"}

    tile_ids = [l.strip() for l in manifest.read_text().splitlines() if l.strip()]
    agg = dict(n_tiles=0, n_ok=0, total_px=0,
               s_w0=0., s_wpos=0., s_wgt01=0., s_wgt05=0., s_weq1=0., s_0to01=0.,
               drops=[], weights=[])

    for tid in tile_ids:
        agg["n_tiles"] += 1
        dem = load_dem_tile(split_prefix, tid)
        if dem is None:
            continue
        st = compute_d8_stats_tile(dem)
        if st is None:
            continue
        n = st["n_pixels"]
        agg["n_ok"]     += 1
        agg["total_px"] += n
        agg["s_w0"]     += st["frac_w0"]    * n
        agg["s_wpos"]   += st["frac_w_pos"] * n
        agg["s_wgt01"]  += st["frac_w_gt01"]* n
        agg["s_wgt05"]  += st["frac_w_gt05"]* n
        agg["s_weq1"]   += st["frac_w_eq1"] * n
        agg["s_0to01"]  += st["frac_0_to_01"]* n
        agg["drops"].append(st["mean_drop_m"])
        agg["weights"].append(st["mean_w"])

    N = agg["total_px"]
    if N == 0:
        return {"error": "no valid pixels"}

    return {
        "split_prefix":    split_prefix,
        "n_tiles":         agg["n_tiles"],
        "n_tiles_ok":      agg["n_ok"],
        "total_pixels":    N,
        "frac_w0":         agg["s_w0"]    / N,
        "frac_w_pos":      agg["s_wpos"]  / N,
        "frac_w_gt01":     agg["s_wgt01"] / N,
        "frac_w_gt05":     agg["s_wgt05"] / N,
        "frac_w_eq1":      agg["s_weq1"]  / N,
        "frac_0_to_01":    agg["s_0to01"] / N,
        "mean_drop_tiles": float(np.mean(agg["drops"]))   if agg["drops"]   else None,
        "mean_w_tiles":    float(np.mean(agg["weights"])) if agg["weights"] else None,
    }

# ── Interpretation block ───────────────────────────────────────────────────────
def generate_interpretation(traces_by_tag, eval_by_tag, d8_act, dem_geom, val_vp) -> list[str]:
    lines = []

    b_tr = traces_by_tag.get("baseline", [])
    r_tr = traces_by_tag.get("d8_real",  [])
    s_tr = traces_by_tag.get("d8_shuf",  [])

    # G1 — trajectory vs early-stop
    r22 = get_epoch_row(r_tr, 22); s22 = get_epoch_row(s_tr, 22); b22 = get_epoch_row(b_tr, 22)
    r31 = get_epoch_row(r_tr, 31); s31 = get_epoch_row(s_tr, 31)

    gap22 = (r22["val_miou"] or 0) - (s22["val_miou"] or 0) if r22 and s22 else None
    gap31 = (r31["val_miou"] or 0) - (s31["val_miou"] or 0) if r31 and s31 else None

    lines += ["### G1. Real > Shuffled: consistent trajectory or early-stopping artifact?", ""]
    if gap22 is not None and gap31 is not None:
        lines += [
            f"Real–Shuffled gap at ep22 (shuf's best): {fmt_delta(gap22)}",
            f"Real–Shuffled gap at ep31 (real's best): {fmt_delta(gap31)}",
            "",
        ]
        if gap31 > gap22 * 1.3:
            lines.append("**Conclusion**: Gap grows from ep22 to ep31 (+{:.5f} → +{:.5f}). "
                         "NOT purely early-stopping: the real-DEM model continues to improve "
                         "relative to shuffled after shuffled's early-stop point. A genuine "
                         "trajectory difference exists.".format(gap22, gap31))
        elif gap31 < gap22 * 0.5:
            lines.append("**Conclusion**: Gap shrinks substantially after ep22. Most of the "
                         "Real-Shuffled gap was present at shuffled's early-stop. Early-stopping "
                         "is a significant confound.")
        else:
            lines.append(f"**Conclusion**: Gap is relatively stable (ep22={fmt_delta(gap22)}, "
                         f"ep31={fmt_delta(gap31)}). Both trajectory difference and early-stopping "
                         "contribute; neither alone explains the gap.")
    lines.append("")

    # G2 — underpowered or satisfied?
    r_d8 = d8_act.get("d8_real", {})
    max_eff = r_d8.get("max_eff_d8") or 0
    lines += ["### G2. D8 underpowered or satisfied at convergence?", ""]
    lines += [
        f"Max effective D8 contribution: {pct(max_eff)} at epoch {r_d8.get('max_eff_d8_epoch')}",
        f"Ever >0.1%: {r_d8.get('ever_above_0p1pct')}  |  Ever >1%: {r_d8.get('ever_above_1pct')}",
        "",
    ]
    if not r_d8.get("ever_above_0p1pct"):
        target = 0.01
        scale  = int(target / max_eff) if max_eff > 0 else "∞"
        lines.append(
            f"**UNDERPOWERED.** D8 contribution never exceeded 0.1%. This is structural: "
            f"the loss magnitude is negligible relative to Dice+CE throughout training. "
            f"To reach 1% effective contribution, lambda would need to be ~{scale}× larger. "
            f"A 'satisfied' loss would show high contribution mid-training decaying to zero; "
            f"here the contribution is near-zero from the start.")
    elif not r_d8.get("ever_above_1pct"):
        lines.append("**BORDERLINE UNDERPOWERED.** Max contribution 0.1–1%. Loss is active "
                     "but weak; gradient impact is marginal.")
    else:
        lines.append("**ADEQUATELY POWERED.** D8 reached >1% contribution, indicating genuine "
                     "gradient influence during training.")
    lines.append("")

    # G3 — physical consistency vs conservatism?
    lines += ["### G3. Does D8-real improve physical consistency or mainly reduce predictions?", ""]
    for sp in ["valid", "test", "bolivia"]:
        b = eval_by_tag.get("baseline",{}).get(sp, {})
        r = eval_by_tag.get("d8_real",  {}).get(sp, {})
        if not b or not r:
            continue
        tv_b = b.get("topo_violation_fraction")
        tv_r = r.get("topo_violation_fraction")
        pr_b = b.get("pred_water_ratio")
        pr_r = r.get("pred_water_ratio")
        dt   = fmt_delta((tv_r - tv_b) if tv_r and tv_b else None, 7)
        lines.append(f"{sp}: topo_vf baseline={fmt(tv_b,6)}  real={fmt(tv_r,6)} (Δ={dt})  |  "
                     f"pred_water baseline={pct(pr_b)}  real={pct(pr_r)}")
    lines += [
        "",
        "Topo violation fraction can fall simply because fewer pixels are predicted as water "
        "(fewer descending pairs to violate). The predicted water ratio drop is the confound. "
        "See G4 for the shuffled comparison.",
        ""
    ]

    # G4 — shuffled lower topo_vf due to conservatism?
    lines += ["### G4. Does shuffled have lower topo violations due to conservatism?", ""]
    for sp in ["valid", "test", "bolivia"]:
        b  = eval_by_tag.get("baseline",{}).get(sp, {})
        r  = eval_by_tag.get("d8_real",  {}).get(sp, {})
        s  = eval_by_tag.get("d8_shuf",  {}).get(sp, {})
        if not (b and r and s):
            continue
        tv_b = b.get("topo_violation_fraction")
        tv_r = r.get("topo_violation_fraction")
        tv_s = s.get("topo_violation_fraction")
        pr_b = b.get("pred_water_ratio")
        pr_r = r.get("pred_water_ratio")
        pr_s = s.get("pred_water_ratio")
        lines.append(f"{sp}: topo_vf: base={fmt(tv_b,6)}  real={fmt(tv_r,6)}  shuf={fmt(tv_s,6)}  |  "
                     f"pred_water: base={pct(pr_b)}  real={pct(pr_r)}  shuf={pct(pr_s)}")
        if tv_s is not None and tv_r is not None and tv_s < tv_r:
            lines.append(f"  → shuffled topo_vf < real: conservatism (under-prediction) is driving "
                         f"lower violation, not DEM-correct routing.")
    lines.append("")

    # G5 — recall drop from underprediction?
    lines += ["### G5. Is the recall drop caused by underprediction of water?", ""]
    for sp in ["valid", "test", "bolivia"]:
        b = eval_by_tag.get("baseline",{}).get(sp, {})
        r = eval_by_tag.get("d8_real",  {}).get(sp, {})
        if not (b and r):
            continue
        pgr_b = b.get("pred_gt_ratio"); pgr_r = r.get("pred_gt_ratio")
        fnr_b = b.get("fnr");           fnr_r = r.get("fnr")
        lines.append(f"{sp}: pred/GT ratio: baseline={fmt(pgr_b)}  real={fmt(pgr_r)}  |  "
                     f"FNR: baseline={pct(fnr_b)}  real={pct(fnr_r)}")
    lines += [
        "",
        "pred/GT < 1.0 confirms under-prediction. Higher FNR confirms more missed water. "
        "The D8 hinge loss penalises p_upstream > p_downstream, which trains the model to "
        "lower water probability for pixels that are upstream of dry pixels — systematically "
        "suppressing predictions in valid flood zones if they are topographically upstream.",
        ""
    ]

    # G6 — is lambda=1.0 too weak?
    lines += ["### G6. Is lambda=1.0 too weak?", ""]
    if max_eff > 0:
        scale = 0.01 / max_eff
        lines += [
            f"Max effective contribution at lambda=1.0: {pct(max_eff)}",
            f"To reach 1% target: need lambda ≈ {scale:.0f}  (i.e. ~{scale:.0f}× current).",
            f"**Yes, lambda=1.0 is too weak by a factor of ~{scale:.0f}.**",
        ]
    else:
        lines += ["Max effective contribution is zero or undefined. Lambda is certainly too weak.", ""]
    lines.append("")

    # G7 summary
    lines += ["### G7. Recommended next step (summary)", ""]
    dem_v = dem_geom.get("valid", {})
    frac_gt01 = dem_v.get("frac_w_gt01")
    if max_eff < 0.001:
        lines.append("**Option B: Rescale lambda before multi-seed.** "
                     "D8 is structurally underpowered. Seeds 1/2 would replicate a null result. "
                     "Fix the scale first.")
    elif frac_gt01 is not None and frac_gt01 < 0.05:
        lines.append("**Option C then B: Reduce s0 to address DEM flatness, then rescale lambda.** "
                     "The DEM has very few pixels with w > 0.1, indicating structural starvation "
                     "from the s0 normalisation.")
    else:
        lines.append("**Option A: Multi-seed with current config.** "
                     "D8 has sufficient activation. Proceed to seeds 1/2.")
    lines.append("")
    return lines

# ── Plots ──────────────────────────────────────────────────────────────────────
def make_plots(traces_by_tag: dict, val_vp: int, fig_dir: Path) -> list[str]:
    if not HAS_MPL:
        return []
    fig_dir.mkdir(parents=True, exist_ok=True)
    created = []

    colors = {"baseline": "#1f77b4", "d8_real": "#2ca02c", "d8_shuf": "#d62728"}
    labels = {"baseline": "Dice+CE Baseline", "d8_real": "D8 Real DEM", "d8_shuf": "D8 Shuffled DEM"}
    best_eps = {"baseline": 34, "d8_real": 31, "d8_shuf": 22}

    def _save(fig, name):
        p = fig_dir / name
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        created.append(str(p))

    def _base_plot(ylabel, title):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.set_title(title)
        for ep in MATCHED_EPOCHS:
            ax.axvline(ep, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)
        ax.grid(True, alpha=0.3)
        return fig, ax

    # 1. Val mIoU
    fig, ax = _base_plot("Val mIoU", "Val mIoU vs Epoch — D8 seed0 (SegMAN-S N=100)")
    for label, traces in traces_by_tag.items():
        ep = [t["epoch"] for t in traces]; m = [t["val_miou"] for t in traces]
        ax.plot(ep, m, color=colors[label], label=labels[label], lw=1.5)
        bep = best_eps.get(label)
        if bep:
            bm = next((t["val_miou"] for t in traces if t["epoch"] == bep), None)
            if bm:
                ax.scatter([bep], [bm], color=colors[label], s=70, zorder=5)
                ax.axvline(bep, color=colors[label], ls="--", alpha=0.3, lw=0.8)
    ax.legend(); _save(fig, "val_miou_vs_epoch.png")

    # 2. Water precision
    fig, ax = _base_plot("Val Precision (water)", "Val Water Precision vs Epoch")
    for label, traces in traces_by_tag.items():
        ep = [t["epoch"] for t in traces]; v = [t["val_precision"] for t in traces]
        ax.plot(ep, v, color=colors[label], label=labels[label], lw=1.5)
    ax.legend(); _save(fig, "val_precision_vs_epoch.png")

    # 3. Water recall
    fig, ax = _base_plot("Val Recall (water)", "Val Water Recall vs Epoch")
    for label, traces in traces_by_tag.items():
        ep = [t["epoch"] for t in traces]; v = [t["val_recall"] for t in traces]
        ax.plot(ep, v, color=colors[label], label=labels[label], lw=1.5)
    ax.legend(); _save(fig, "val_recall_vs_epoch.png")

    # 4. Raw D8 loss
    fig, ax = _base_plot("Train D8 Loss (raw)", "Raw D8 Training Loss vs Epoch")
    for label, traces in traces_by_tag.items():
        if label == "baseline":
            continue
        ep = [t["epoch"] for t in traces]; v = [t["train_loss_d8"] for t in traces]
        ax.plot(ep, v, color=colors[label], label=labels[label], lw=1.5)
    ax.legend(); _save(fig, "train_d8_loss_vs_epoch.png")

    # 5. Effective D8 contribution
    fig, ax = _base_plot("Effective D8 Contribution (%)",
                          "Effective D8 Contribution [λ·L_D8/(L_Dice+L_CE)] vs Epoch")
    for label, traces in traces_by_tag.items():
        if label == "baseline":
            continue
        ep  = [t["epoch"] for t in traces if t["eff_d8_contribution"] is not None]
        eff = [t["eff_d8_contribution"] * 100 for t in traces if t["eff_d8_contribution"] is not None]
        ax.plot(ep, eff, color=colors[label], label=labels[label], lw=1.5)
    ax.axhline(0.1, color="orange", ls="--", lw=0.8, label="0.1% threshold")
    ax.axhline(1.0, color="red",    ls="--", lw=0.8, label="1.0% threshold")
    ax.legend(); _save(fig, "eff_d8_contribution_vs_epoch.png")

    # 6. Predicted water ratio
    if val_vp > 0:
        fig, ax = _base_plot("Predicted Water Ratio (%)", "Predicted Water Ratio vs Epoch (Val)")
        for label, traces in traces_by_tag.items():
            ep = [t["epoch"] for t in traces if t["val_pred_px"] is not None]
            v  = [t["val_pred_px"] / val_vp * 100 for t in traces if t["val_pred_px"] is not None]
            ax.plot(ep, v, color=colors[label], label=labels[label], lw=1.5)
        ax.legend(); _save(fig, "pred_water_ratio_vs_epoch.png")

    # 7. Topo violation fraction
    fig, ax = _base_plot("Topo Violation Fraction (%)", "Val Topo Violation Fraction vs Epoch")
    for label, traces in traces_by_tag.items():
        ep = [t["epoch"] for t in traces if t["val_topo_vf"] is not None]
        v  = [t["val_topo_vf"] * 100 for t in traces if t["val_topo_vf"] is not None]
        ax.plot(ep, v, color=colors[label], label=labels[label], lw=1.5)
    ax.legend(); _save(fig, "val_topo_violation_vs_epoch.png")

    return created

# ── Markdown report ────────────────────────────────────────────────────────────
def build_markdown(traces_by_tag, eval_by_tag, d8_act, dem_geom,
                   matched, interp, rec_next, val_vp, gt_wr, figs) -> str:
    L = []

    L += [
        "# SegMAN-S N=100 D8 Seed0 — Post-hoc Diagnostic Report",
        "",
        "**Date**: 2026-06-29  |  **Branch**: `experiments/segman-cvpr2025`  |  **Commit**: `8209243`",
        "**Runs**: Dice+CE baseline, D8 real DEM, D8 shuffled DEM (all seed0, N=100)",
        "**No training was launched. All data from existing epoch CSV and summary JSON files.**",
        "",
    ]

    # Part C: matched-epoch table
    L += [
        "## Part C — Matched-Epoch Analysis (Early-Stopping Confound)",
        "",
        "Val mIoU at same epoch for all three runs:",
        "",
        "| Epoch | Baseline | D8 real | D8 shuffled | Real−Shuffled | Real−Baseline | Shuf−Baseline |",
        "|-------|----------|---------|-------------|---------------|---------------|---------------|",
    ]
    notes = {22: "← shuf best", 31: "← real best", 34: "← base best"}
    for ep, row in matched.items():
        L.append(
            f"| ep{ep} {notes.get(ep,'')} | {fmt(row.get('baseline_val_miou'))} | "
            f"{fmt(row.get('d8_real_val_miou'))} | {fmt(row.get('d8_shuf_val_miou'))} | "
            f"{fmt_delta(row.get('real_minus_shuf'))} | {fmt_delta(row.get('real_minus_base'))} | "
            f"{fmt_delta(row.get('shuf_minus_base'))} |"
        )
    L += [
        "",
        "Val precision / recall / predicted-water-ratio at matched epochs:",
        "",
        "| Epoch | Run | Precision | Recall | Pred water % | Topo VF |",
        "|-------|-----|-----------|--------|--------------|---------|",
    ]
    run_labels = {"baseline": "Baseline", "d8_real": "D8 real", "d8_shuf": "D8 shuf"}
    for ep, row in matched.items():
        for run in ["baseline", "d8_real", "d8_shuf"]:
            pp  = fmt(row.get(f"{run}_precision"))
            rc  = fmt(row.get(f"{run}_recall"))
            ppx = row.get(f"{run}_pred_px")
            pw  = pct(ppx / val_vp if ppx and val_vp else None)
            tv  = fmt(row.get(f"{run}_topo_vf"), 6)
            L.append(f"| ep{ep} | {run_labels[run]} | {pp} | {rc} | {pw} | {tv} |")
    L += [""]

    # Part D: D8 activation
    L += [
        "## Part D — D8 Activation Diagnostic",
        "",
        "| Metric | D8 real | D8 shuffled |",
        "|--------|---------|-------------|",
    ]
    r_d8 = d8_act.get("d8_real", {}); s_d8 = d8_act.get("d8_shuf", {})
    rows_d8 = [
        ("Max eff. D8 contribution",   pct(r_d8.get("max_eff_d8")),        pct(s_d8.get("max_eff_d8"))),
        ("Epoch of max eff. D8",       str(r_d8.get("max_eff_d8_epoch")),  str(s_d8.get("max_eff_d8_epoch"))),
        ("Mean eff. D8 (all epochs)",  pct(r_d8.get("mean_eff_d8")),       pct(s_d8.get("mean_eff_d8"))),
        ("Eff. D8 at best epoch",      pct(r_d8.get("eff_d8_at_best_ep")),pct(s_d8.get("eff_d8_at_best_ep"))),
        ("Ever above 0.1%",            str(r_d8.get("ever_above_0p1pct")), str(s_d8.get("ever_above_0p1pct"))),
        ("Ever above 1.0%",            str(r_d8.get("ever_above_1pct")),   str(s_d8.get("ever_above_1pct"))),
        ("Ever above 3.0%",            str(r_d8.get("ever_above_3pct")),   str(s_d8.get("ever_above_3pct"))),
        ("Max raw D8 loss",            fmt(r_d8.get("max_raw_d8_loss"), 8),fmt(s_d8.get("max_raw_d8_loss"), 8)),
        ("Max lambda applied",         fmt(r_d8.get("max_lambda")),         fmt(s_d8.get("max_lambda"))),
    ]
    for m, rv, sv in rows_d8:
        L.append(f"| {m} | {rv} | {sv} |")
    L += [""]

    # Part E: full metrics
    L += ["## Part E — Full Evaluation Metrics by Split", ""]
    metric_keys = [
        ("mean_iou",              "mIoU",                  5),
        ("iou_water",             "IoU water",             5),
        ("iou_background",        "IoU background",        5),
        ("macro_f1",              "Macro F1",              5),
        ("f1_water",              "F1 water",              5),
        ("f1_background",         "F1 background",         5),
        ("accuracy",              "Pixel accuracy",        5),
        ("balanced_accuracy",     "Balanced accuracy",     5),
        ("precision_water",       "Precision water",       5),
        ("recall_water",          "Recall water",          5),
        ("specificity",           "Specificity (TNR)",     5),
        ("fpr",                   "FPR",                   5),
        ("fnr",                   "FNR",                   5),
        ("gt_water_ratio",        "GT water ratio",        4),
        ("pred_water_ratio",      "Pred water ratio",      4),
        ("water_ratio_error",     "Water ratio error",     4),
        ("pred_gt_ratio",         "Pred/GT ratio",         4),
        ("topo_violation_fraction","Topo violation frac",  6),
    ]
    for sp in SPLITS:
        L += [f"### {sp.capitalize()}", ""]
        L += [
            "| Metric | Baseline | D8 real | D8 shuffled | Real−Base | Shuf−Base | Real−Shuf |",
            "|--------|----------|---------|-------------|-----------|-----------|-----------|",
        ]
        for key, label, p in metric_keys:
            bv = eval_by_tag.get("baseline",{}).get(sp, {}).get(key)
            rv = eval_by_tag.get("d8_real", {}).get(sp, {}).get(key)
            sv = eval_by_tag.get("d8_shuf", {}).get(sp, {}).get(key)
            rb = fmt_delta((rv - bv) if rv is not None and bv is not None else None, p)
            sb = fmt_delta((sv - bv) if sv is not None and bv is not None else None, p)
            rs = fmt_delta((rv - sv) if rv is not None and sv is not None else None, p)
            L.append(f"| {label} | {fmt(bv,p)} | {fmt(rv,p)} | {fmt(sv,p)} | {rb} | {sb} | {rs} |")
        L += ["", "**Confusion matrix:**", ""]
        L += ["| | Baseline | D8 real | D8 shuffled |", "|---|---|---|---|"]
        for key, lab in [("tp","TP"),("fp","FP"),("fn","FN"),("tn","TN"),
                          ("support_water","GT water px"),("water_pred_pixels","Pred water px")]:
            bv = eval_by_tag.get("baseline",{}).get(sp,{}).get(key)
            rv = eval_by_tag.get("d8_real", {}).get(sp,{}).get(key)
            sv = eval_by_tag.get("d8_shuf", {}).get(sp,{}).get(key)
            L.append(f"| {lab} | {_int(bv)} | {_int(rv)} | {_int(sv)} |")
        L += [""]

    # Part F: DEM geometry
    L += ["## Part F — DEM Geometry / D8 Weight Distribution", ""]
    for sp_name, sp_key in [("Val (86 tiles)", "valid"), ("Train N=100 seed0 (100 tiles)", "train")]:
        g = dem_geom.get(sp_key, {})
        L += [f"### {sp_name}", ""]
        if "error" in g:
            L += [f"Unavailable: {g['error']}", ""]
            continue
        L += [
            "| Metric | Value |",
            "|--------|-------|",
            f"| Tiles (OK/total) | {g.get('n_tiles_ok')}/{g.get('n_tiles')} |",
            f"| Total valid pixels | {_int(g.get('total_pixels'))} |",
            f"| Fraction w=0 (flat — zero activation) | {pct(g.get('frac_w0'))} |",
            f"| Fraction 0<w≤0.1 (very low activation) | {pct(g.get('frac_0_to_01'))} |",
            f"| Fraction w>0.1 (moderate activation) | {pct(g.get('frac_w_gt01'))} |",
            f"| Fraction w>0.5 (strong activation) | {pct(g.get('frac_w_gt05'))} |",
            f"| Fraction w=1 (fully saturated) | {pct(g.get('frac_w_eq1'))} |",
            f"| Mean D8 drop per tile (m) | {fmt(g.get('mean_drop_tiles'), 4)} |",
            f"| Mean D8 weight per tile | {fmt(g.get('mean_w_tiles'), 4)} |",
            "",
            "s0=1.0m means full weight only when slope ≥ 1 m per 10m pixel (≥10% grade). "
            "GLO-30 DEM is originally 30m and resampled to 512×512; if resampled with bilinear "
            "interpolation, effective relief per 10m pixel may be much smaller than 1m.",
            "",
        ]

    # Part G
    L += ["## Part G — Interpretation", ""] + interp

    # Part H
    L += ["## Part H — Recommended Next Experiment", ""] + rec_next

    # Figures
    if figs:
        L += ["## Figures", ""]
        for fp in figs:
            fn  = Path(fp).name
            rel = f"../reports/figures/segman_n100_d8_seed0/{fn}"
            L.append(f"- [{fn}]({rel})")
        L += [""]

    L += [
        "---",
        "",
        "*Generated by `experiments_cvpr/segman/diagnose_d8_seed0_posthoc.py`.*",
        "*No training launched. No model or loss code modified.*",
    ]
    return "\n".join(L) + "\n"

# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> int:
    print("=" * 70)
    print("SegMAN N=100 D8 Seed0 — Post-hoc Diagnostic")
    print("=" * 70)

    # Part A: preflight
    print("\n[Part A] Preflight")
    for label, tag in TAGS.items():
        cp = RUNS_ROOT / tag / "metrics" / "training_epoch_metrics.csv"
        sp = RUNS_ROOT / tag / "metrics" / f"{tag}_summary.json"
        bp = RUNS_ROOT / tag / "checkpoints" / "best_checkpoint.pt"
        print(f"  {label:10s}: CSV={cp.exists()}  summary={sp.exists()}  ckpt={bp.exists()}")
    print("  No training will be launched.")

    # Load raw data
    all_rows    = {lb: load_epoch_csv(tag) for lb, tag in TAGS.items()}
    all_summary = {lb: load_summary(tag)   for lb, tag in TAGS.items()}

    # Part B: per-epoch traces
    print("\n[Part B] Per-epoch traces")
    traces = {lb: compute_epoch_traces(lb, rows) for lb, rows in all_rows.items()}
    for lb, tr in traces.items():
        valid_miou = [t["val_miou"] for t in tr if t["val_miou"]]
        print(f"  {lb:10s}: {len(tr)} epochs  "
              f"best_val_miou={fmt(max(valid_miou) if valid_miou else None)}")

    val_vp = 0; gt_wr = None
    base_s = all_summary.get("baseline")
    if base_s:
        vev = (base_s.get("evaluations") or {}).get("valid") or {}
        val_vp = int(vev.get("valid_pixel_count") or 0)
        sw = safe_float(vev.get("support_water"))
        if sw and val_vp:
            gt_wr = sw / val_vp
    print(f"  val valid_px={val_vp:,}  gt_water_ratio={fmt(gt_wr)}")

    # Part C: matched-epoch analysis
    print("\n[Part C] Matched-epoch analysis")
    matched = {}
    for ep in MATCHED_EPOCHS:
        row = {}
        for lb, tr in traces.items():
            t = get_epoch_row(tr, ep)
            if t:
                row[f"{lb}_val_miou"]  = t.get("val_miou")
                row[f"{lb}_precision"] = t.get("val_precision")
                row[f"{lb}_recall"]    = t.get("val_recall")
                row[f"{lb}_topo_vf"]   = t.get("val_topo_vf")
                row[f"{lb}_pred_px"]   = t.get("val_pred_px")
        bm = row.get("baseline_val_miou")
        rm = row.get("d8_real_val_miou")
        sm = row.get("d8_shuf_val_miou")
        row["real_minus_shuf"] = (rm - sm) if rm and sm else None
        row["real_minus_base"] = (rm - bm) if rm and bm else None
        row["shuf_minus_base"] = (sm - bm) if sm and bm else None
        matched[ep] = row
        print(f"  ep{ep:2d}: base={fmt(bm)}  real={fmt(rm)}  shuf={fmt(sm)}  "
              f"R-S={fmt_delta(row['real_minus_shuf'])}  R-B={fmt_delta(row['real_minus_base'])}")

    # Part D: D8 activation
    print("\n[Part D] D8 activation")
    d8_act = analyze_d8_activation(traces)
    for lb in ["d8_real", "d8_shuf"]:
        a = d8_act.get(lb, {})
        s = all_summary.get(lb)
        if s and "max_eff_d8" in a:
            bep = s.get("best_epoch")
            if bep:
                t = get_epoch_row(traces.get(lb, []), bep)
                if t:
                    a["eff_d8_at_best_ep"] = t.get("eff_d8_contribution")
        print(f"  {lb:10s}: max_eff={pct(a.get('max_eff_d8'))} at ep{a.get('max_eff_d8_epoch')}  "
              f">0.1%={a.get('ever_above_0p1pct')}  >1%={a.get('ever_above_1pct')}")

    # Part E: full eval metrics
    print("\n[Part E] Full eval metrics")
    evals = {}
    for lb, s in all_summary.items():
        evals[lb] = {}
        if not s:
            continue
        for sp in SPLITS:
            evals[lb][sp] = compute_eval_metrics(s, sp)
    for lb in TAGS:
        v = evals.get(lb, {}).get("valid", {})
        print(f"  {lb:10s}: mIoU={fmt(v.get('mean_iou'))}  "
              f"prec={fmt(v.get('precision_water'))}  "
              f"rec={fmt(v.get('recall_water'))}  "
              f"bal_acc={fmt(v.get('balanced_accuracy'))}  "
              f"FNR={pct(v.get('fnr'))}  pred_water={pct(v.get('pred_water_ratio'))}")

    # Part F: DEM geometry
    print("\n[Part F] DEM geometry")
    dem_geom: dict = {}
    if HAS_RIO:
        print("  Computing val geometry (86 tiles)...")
        dem_geom["valid"] = compute_dem_geometry("valid", VAL_MANIFEST)
        dv = dem_geom["valid"]
        if "error" not in dv:
            print(f"    OK: frac_w0={pct(dv.get('frac_w0'))}  frac_w>0.1={pct(dv.get('frac_w_gt01'))}  "
                  f"mean_drop={fmt(dv.get('mean_drop_tiles'), 4)}m")
        else:
            print(f"    ERROR: {dv['error']}")
        print("  Computing train geometry (100 tiles)...")
        dem_geom["train"] = compute_dem_geometry("train", TRAIN_MANIFEST)
        dt = dem_geom["train"]
        if "error" not in dt:
            print(f"    OK: frac_w0={pct(dt.get('frac_w0'))}  frac_w>0.1={pct(dt.get('frac_w_gt01'))}  "
                  f"mean_drop={fmt(dt.get('mean_drop_tiles'), 4)}m")
        else:
            print(f"    ERROR: {dt['error']}")
    else:
        print("  rasterio unavailable — skipping")
        dem_geom = {"valid": {"error": "rasterio not available"},
                    "train": {"error": "rasterio not available"}}

    # Part G: interpretation
    print("\n[Part G] Interpretation")
    interp = generate_interpretation(traces, evals, d8_act, dem_geom, val_vp)

    # Part H: recommendation
    max_eff   = d8_act.get("d8_real", {}).get("max_eff_d8") or 0
    frac_gt01 = dem_geom.get("valid", {}).get("frac_w_gt01")
    rec_next  = []

    if max_eff < 0.001:
        scale = int(0.01 / max_eff) if max_eff > 0 else ">>100"
        rec_next += [
            "**Recommendation: B — Rescale lambda before multi-seed.**",
            "",
            f"Peak effective D8 contribution = {pct(max_eff)} (target: >1%). "
            f"Lambda=1.0 is too weak by factor ~{scale}.",
            "",
            "Proposed sequence:",
            f"1. Create new config: `lambda_topo: {min(int(scale), 500) if isinstance(scale, int) else 100}` "
            f"  (single knob only; tau=0.05 and s0=1.0 unchanged).",
            "2. Rerun D8-real + D8-shuffled at seed0 only (2 runs).",
            "3. Verify eff. contribution reaches 1–5% before proceeding.",
            "4. If confirmed: lock config -> seeds 1/2 for both D8 and baseline.",
            "",
            "**Also run baseline seeds 1/2** before any final comparison — "
            "you currently have no baseline variance to calibrate deltas against.",
        ]
    elif frac_gt01 is not None and frac_gt01 < 0.05:
        rec_next += [
            "**Recommendation: Reduce s0 before lambda tuning.**",
            "",
            f"Only {pct(frac_gt01)} of val pixels have w>0.1 with s0=1.0m. "
            "DEM is too flat for this normalisation. Try s0=0.1m (1% slope gives full weight).",
        ]
    else:
        rec_next += [
            "**Recommendation: A — Multi-seed with current config.**",
            "D8 is adequately powered. Proceed to seeds 1/2.",
        ]

    rec_next += [
        "",
        "**Command to run next (DO NOT run now — for reference only):**",
        "```powershell",
        "# After updating lambda in config:",
        "# .\\scripts\\launch_segman_n100_d8_seed0_chain.ps1",
        "```",
    ]

    # Plots
    print("\n[Plots]")
    fig_dir  = REPO_ROOT / "reports" / "figures" / "segman_n100_d8_seed0"
    figs     = make_plots(traces, val_vp, fig_dir)
    print(f"  Created {len(figs)} figures in {fig_dir}")

    # Part I: outputs
    print("\n[Part I] Writing outputs")
    rep_dir = REPO_ROOT / "reports"
    doc_dir = REPO_ROOT / "docs"
    rep_dir.mkdir(parents=True, exist_ok=True)
    doc_dir.mkdir(parents=True, exist_ok=True)

    # Epoch traces CSV
    all_traces = [t for tr in traces.values() for t in tr]
    ep_csv = rep_dir / "segman_n100_d8_seed0_posthoc_epoch_traces.csv"
    if all_traces:
        keys = list(all_traces[0].keys())
        with ep_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv_mod.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader(); w.writerows(all_traces)
    print(f"  Wrote: {ep_csv}")

    # Metrics CSV
    met_rows = []
    for lb in TAGS:
        for sp in SPLITS:
            row = {"run": lb, **evals.get(lb, {}).get(sp, {})}
            met_rows.append(row)
    met_csv = rep_dir / "segman_n100_d8_seed0_posthoc_metrics.csv"
    if met_rows:
        keys2 = list(met_rows[0].keys())
        with met_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv_mod.DictWriter(f, fieldnames=keys2, extrasaction="ignore")
            w.writeheader(); w.writerows(met_rows)
    print(f"  Wrote: {met_csv}")

    # Diagnostics JSON
    payload = {
        "matched_epoch_table":   {str(k): v for k, v in matched.items()},
        "d8_activation":         {k: {k2: v2 for k2, v2 in v.items() if k2 != "all_eff_d8"}
                                   for k, v in d8_act.items()},
        "dem_geometry":          dem_geom,
        "eval_by_tag":           evals,
        "val_valid_pixel_count": val_vp,
        "gt_water_ratio_val":    gt_wr,
        "figures_created":       figs,
    }
    diag_json = rep_dir / "segman_n100_d8_seed0_posthoc_diagnostics.json"
    diag_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"  Wrote: {diag_json}")

    # Markdown
    md_str  = build_markdown(traces, evals, d8_act, dem_geom, matched,
                              interp, rec_next, val_vp, gt_wr, figs)
    md_path = doc_dir / "segman_n100_d8_seed0_posthoc_diagnostics.md"
    md_path.write_text(md_str, encoding="utf-8")
    print(f"  Wrote: {md_path}")

    # Part J: final console summary
    print("\n" + "=" * 70)
    print("FINAL REPORT — SUMMARY")
    print("=" * 70)
    print("Branch: experiments/segman-cvpr2025  |  Commit: 8209243")
    print("No training launched. No model/loss code modified.")

    print("\n[Matched-epoch val mIoU]")
    for ep, row in matched.items():
        print(f"  ep{ep:2d}: base={fmt(row.get('baseline_val_miou'))}  "
              f"real={fmt(row.get('d8_real_val_miou'))}  shuf={fmt(row.get('d8_shuf_val_miou'))}  "
              f"R-S={fmt_delta(row.get('real_minus_shuf'))}  R-B={fmt_delta(row.get('real_minus_base'))}")

    print("\n[D8 activation]")
    for lb in ["d8_real", "d8_shuf"]:
        a = d8_act.get(lb, {})
        print(f"  {lb:10s}: max_eff={pct(a.get('max_eff_d8'))} (ep{a.get('max_eff_d8_epoch')})  "
              f">0.1%={a.get('ever_above_0p1pct')}  >1%={a.get('ever_above_1pct')}  "
              f"at_best_ep={pct(a.get('eff_d8_at_best_ep'))}")

    print("\n[DEM geometry — val tiles]")
    dv = dem_geom.get("valid", {})
    if "error" not in dv:
        print(f"  frac_w=0  : {pct(dv.get('frac_w0'))}  (structurally flat)")
        print(f"  frac_w>0.1: {pct(dv.get('frac_w_gt01'))}")
        print(f"  frac_w>0.5: {pct(dv.get('frac_w_gt05'))}")
        print(f"  frac_w=1  : {pct(dv.get('frac_w_eq1'))}")
        print(f"  mean_drop : {fmt(dv.get('mean_drop_tiles'), 4)} m per tile")
    else:
        print(f"  {dv['error']}")

    print("\n[Full val metrics]")
    for lb in TAGS:
        v = evals.get(lb, {}).get("valid", {})
        print(f"  {lb:10s}: mIoU={fmt(v.get('mean_iou'))}  "
              f"prec={fmt(v.get('precision_water'))}  rec={fmt(v.get('recall_water'))}  "
              f"bal_acc={fmt(v.get('balanced_accuracy'))}  FNR={pct(v.get('fnr'))}  "
              f"pred/GT={fmt(v.get('pred_gt_ratio'))}")

    print("\n[Verdict: D8 underpowered or satisfied?]")
    if max_eff < 0.001:
        print("  UNDERPOWERED — contribution never reached 0.1% threshold.")
    elif max_eff < 0.01:
        print("  BORDERLINE — max contribution between 0.1% and 1%.")
    else:
        print("  ADEQUATELY POWERED — max contribution above 1%.")

    print("\n[Early-stopping confound?]")
    r22m = matched.get(22, {}).get("d8_real_val_miou")
    s22m = matched.get(22, {}).get("d8_shuf_val_miou")
    r31m = matched.get(31, {}).get("d8_real_val_miou")
    s31m = matched.get(31, {}).get("d8_shuf_val_miou")
    if r22m and s22m and r31m and s31m:
        g22 = r22m - s22m; g31 = r31m - s31m
        print(f"  Gap at ep22 (shuf best): {fmt_delta(g22)}")
        print(f"  Gap at ep31 (real best): {fmt_delta(g31)}")
        print(f"  Gap {'GROWS' if g31 > g22 * 1.1 else 'SHRINKS' if g31 < g22 * 0.7 else 'STABLE'} "
              f"— early-stopping {'is NOT the main' if g31 > g22 * 1.1 else 'IS a significant'} confound.")

    print("\n[Recommended next step]")
    for line in rec_next:
        if line and not line.startswith("```") and not line.endswith("```"):
            print(f"  {line}")

    print("\n[Files written]")
    for p in [ep_csv, met_csv, diag_json, md_path] + [Path(f) for f in figs]:
        print(f"  {p}")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
