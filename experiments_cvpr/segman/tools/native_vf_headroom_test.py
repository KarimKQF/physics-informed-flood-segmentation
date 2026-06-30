"""
Native D8 Violation-Fraction Headroom + Useful Violation Rate  (inference-only)
==============================================================================
Two free, inference-only diagnostics that close the "T is a partial statistic"
loophole for the D8 / topographic-ordering constraint class, in the constraint's
OWN units (violation fraction), relative to the reference labels.

SegMAN is run once (frozen). The DEM is used ONLY post-hoc to score violations.
DEM is NEVER a model input.

(1) NATIVE VF HEADROOM   H_R = VF(p_hat_0) - VF(y)
    VF(.) = D8 downstream violation fraction (same formula as D8DownstreamLoss /
    diagnose_gt_topographic_consistency): a pixel i that is water while its D8
    steepest-descent downstream neighbor d(i) is dry, slope-weighted by
    w=min(1,drop/s0). Computed on:
        - y          : the reference labels                  -> VF(y)
        - y_hat_0    : the binarized baseline prediction     -> VF(p_hat_0)
    If VF(p_hat_0) <= VF(y) (H_R <= 0) the baseline already violates the
    constraint LESS than the labels -> no native headroom for any constraint
    monotone in this VF.

(2) USEFUL VIOLATION RATE
    Among the prediction's D8-active pixels (the set the loss would push DOWN:
    predicted-water i upstream of a drier downstream d(i)), what does acting on
    them do to label-agreement?
        useful   : GT(i) = dry   -> suppression removes a false positive  (good)
        harmful  : GT(i) = water -> suppression removes a true positive   (bad, recall loss)
        endorsed : GT(i)=water AND GT(d(i))=dry -> GT ITSELF violates D8 here,
                   so the loss penalizes a label-correct configuration.
    Reported on the binary violation set (y_hat water_i & dry_d) and on the
    hinge-active set (p_i - p_d > tau), matching the actual loss gradient.

LIMITATION: relative to reference labels, not the true physical flood state.
Closes the VF-monotone local class; not orthogonal statistics (flow connectivity).

Outputs -> outputs/native_vf_headroom/ . Read-only w.r.t. SegMAN weights.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR  = Path(__file__).resolve().parent
SEGMAN_ROOT = SCRIPT_DIR.parent
REPO_ROOT   = SEGMAN_ROOT.parents[1]
for _p in (str(SEGMAN_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from step6c_v3_train import TopographyDataModule              # noqa: E402
from model.segman_model import build_segman                   # noqa: E402
from diagnose_gt_topographic_consistency import (             # noqa: E402
    IGNORE_INDEX, WATER_CLASS, DRY_CLASS, _compute_d8,
)

EPS = 1e-9


# ── input assembler (same normalisation as training/inference) ────────────────
class InputAssembler:
    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        da = config["data"]["init_args"]
        self.modalities = list(da["modalities"])
        self.mean = {m: torch.tensor(da["means"][m], dtype=torch.float32, device=device).view(1, -1, 1, 1) for m in self.modalities}
        self.std  = {m: torch.tensor(da["stds"][m],  dtype=torch.float32, device=device).view(1, -1, 1, 1) for m in self.modalities}

    def __call__(self, image: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([(image[m].float() - self.mean[m]) / self.std[m] for m in self.modalities], dim=1)


def _get_topo(batch):
    topo = batch.get("topography")
    if topo is None:
        return None
    if topo.ndim == 4 and topo.shape[1] == 1:
        topo = topo[:, 0]
    return topo.float()


def _get_target(batch):
    mask = batch["mask"]
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]
    return mask.long()


def load_config(config_path: Path) -> dict[str, Any]:
    import yaml
    with config_path.open("r", encoding="utf-8-sig") as f:
        config = yaml.safe_load(f)
    return config


# ── per-tile D8 statistics (exact loss units) ─────────────────────────────────
def tile_d8_stats(h: np.ndarray, y: np.ndarray, p: np.ndarray,
                  s0: float, tau: float, thr: float) -> dict[str, float]:
    """All raw sums + per-tile fractions for one tile.

    h: DEM [H,W] (m, may have NaN) ; y: label [H,W] (int, IGNORE/0/1) ;
    p: water probability [H,W] in [0,1].
    """
    valid = (y != IGNORE_INDEX) & np.isfinite(h)
    d_row, d_col, drop = _compute_d8(np.where(np.isfinite(h), h, np.nan).astype(np.float32))
    w = np.clip(drop / s0, 0.0, 1.0)

    valid_d = valid[d_row, d_col]
    active = valid & valid_d & (drop > 0.0)          # i is upstream of d(i)

    gt_water = (y == WATER_CLASS); gt_dry = (y == DRY_CLASS)
    gt_dry_d = gt_dry[d_row, d_col]
    gt_viol = active & gt_water & gt_dry_d            # native GT violation

    yhat_water = valid & (p > thr); yhat_dry = valid & (p <= thr)
    yhat_dry_d = yhat_dry[d_row, d_col]
    pred_viol = active & yhat_water & yhat_dry_d      # native prediction violation

    p_d = p[d_row, d_col]
    hinge_active = active & ((p - p_d) > tau)         # set the loss gradient acts on

    def _sum(mask_):
        return float((w * mask_).sum()), int(mask_.sum())

    w_active, uw_active = _sum(active)
    w_gtv,   uw_gtv     = _sum(gt_viol)
    w_pv,    uw_pv      = _sum(pred_viol)

    # useful-violation accounting on the binary prediction-violation set
    n_pv      = uw_pv
    n_useful  = int((pred_viol & gt_dry).sum())       # center GT dry  -> good to suppress
    n_harmful = int((pred_viol & gt_water).sum())     # center GT water-> bad to suppress
    n_endors  = int((pred_viol & gt_water & gt_dry_d).sum())  # GT also violates

    # same on the hinge-active set (matches loss gradient)
    n_ha        = int(hinge_active.sum())
    n_ha_useful = int((hinge_active & gt_dry).sum())
    n_ha_harm   = int((hinge_active & gt_water).sum())
    n_ha_endors = int((hinge_active & gt_water & gt_dry_d).sum())

    return {
        # raw sums for pooled aggregation
        "w_active": w_active, "uw_active": uw_active,
        "w_gtv": w_gtv, "uw_gtv": uw_gtv, "w_pv": w_pv, "uw_pv": uw_pv,
        "n_pv": n_pv, "n_useful": n_useful, "n_harmful": n_harmful, "n_endors": n_endors,
        "n_ha": n_ha, "n_ha_useful": n_ha_useful, "n_ha_harm": n_ha_harm, "n_ha_endors": n_ha_endors,
        # per-tile fractions
        "vf_gt_w":   w_gtv / (w_active + EPS), "vf_gt_uw":  uw_gtv / (uw_active + EPS),
        "vf_pred_w": w_pv  / (w_active + EPS), "vf_pred_uw": uw_pv / (uw_active + EPS),
        "n_valid": int(valid.sum()),
    }


def collect_split(model, assembler, config, dm_split, device, s0, tau, thr, max_tiles):
    dm = TopographyDataModule(config, batch_size=1)
    dm.split = dm_split
    dm.setup("test")
    loader = dm.test_dataloader()
    tiles: list[dict[str, float]] = []
    n = 0
    model.eval()
    with torch.no_grad():
        for raw in loader:
            if max_tiles is not None and n >= max_tiles:
                break
            topo = _get_topo(raw)
            if topo is None:
                continue
            image = {k: v.to(device) for k, v in raw["image"].items()}
            logits = model(assembler(image))
            p2d = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()
            y2d = _get_target(raw)[0].cpu().numpy()
            h2d = topo[0].cpu().numpy()
            st = tile_d8_stats(h2d, y2d, p2d, s0, tau, thr)
            if st["uw_active"] < 1:
                continue
            st["tile"] = n
            tiles.append(st)
            n += 1
    return tiles


def _boot_ci(diffs: np.ndarray, n_boot: int, seed: int):
    if diffs.size == 0:
        return [float("nan")] * 3
    rng = np.random.default_rng(seed)
    means = [diffs[rng.integers(0, diffs.size, diffs.size)].mean() for _ in range(n_boot)]
    a = np.array(means)
    return [float(np.percentile(a, 2.5)), float(diffs.mean()), float(np.percentile(a, 97.5))]


def aggregate(tiles: list[dict], n_boot: int, seed: int) -> dict[str, Any]:
    def S(k): return float(sum(t[k] for t in tiles))
    w_active, uw_active = S("w_active"), S("uw_active")
    pooled = {
        "vf_gt_weighted":     S("w_gtv") / (w_active + EPS),
        "vf_gt_unweighted":   S("uw_gtv") / (uw_active + EPS),
        "vf_pred_weighted":   S("w_pv") / (w_active + EPS),
        "vf_pred_unweighted": S("uw_pv") / (uw_active + EPS),
    }
    pooled["H_R_weighted"]   = pooled["vf_pred_weighted"]   - pooled["vf_gt_weighted"]
    pooled["H_R_unweighted"] = pooled["vf_pred_unweighted"] - pooled["vf_gt_unweighted"]

    n_pv = S("n_pv")
    uvr = {
        "binary_set": {
            "n_active": int(n_pv),
            "useful_rate":   S("n_useful")  / (n_pv + EPS),
            "harmful_rate":  S("n_harmful") / (n_pv + EPS),
            "label_endorsed_rate": S("n_endors") / (n_pv + EPS),
        },
    }
    n_ha = S("n_ha")
    uvr["hinge_set"] = {
        "n_active": int(n_ha),
        "useful_rate":   S("n_ha_useful") / (n_ha + EPS),
        "harmful_rate":  S("n_ha_harm")   / (n_ha + EPS),
        "label_endorsed_rate": S("n_ha_endors") / (n_ha + EPS),
    }

    # per-tile paired headroom (pred - gt), bootstrap by tile
    dvf_w  = np.array([t["vf_pred_w"]  - t["vf_gt_w"]  for t in tiles])
    dvf_uw = np.array([t["vf_pred_uw"] - t["vf_gt_uw"] for t in tiles])
    paired = {
        "n_tiles": len(tiles),
        "H_R_weighted_pertile_ci":   _boot_ci(dvf_w,  n_boot, seed),
        "H_R_unweighted_pertile_ci": _boot_ci(dvf_uw, n_boot, seed + 1),
        "pct_tiles_pred_cleaner_weighted":   100.0 * float(np.mean(dvf_w  <= 0)),
        "pct_tiles_pred_cleaner_unweighted": 100.0 * float(np.mean(dvf_uw <= 0)),
    }
    return {"pooled": pooled, "useful_violation": uvr, "paired": paired}


def main() -> int:
    ap = argparse.ArgumentParser(description="Native D8 VF headroom + useful violation rate (inference-only).")
    ap.add_argument("--config", type=Path,
                    default=REPO_ROOT / "configs/segman/multiseed_n100/n100_seed0_dice_ce.yaml")
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--eval-splits", nargs="+", default=["val", "test", "bolivia"])
    ap.add_argument("--s0", type=float, default=1.0, help="D8 slope-weight scale (m); matches loss default.")
    ap.add_argument("--tau", type=float, default=0.05, help="hinge margin; matches loss default.")
    ap.add_argument("--thr", type=float, default=0.5, help="prob threshold for binarised prediction.")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--max-tiles", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs/native_vf_headroom")
    args = ap.parse_args()

    config = load_config(args.config)
    if config.get("dem", {}).get("use_as_model_input", False):
        raise RuntimeError("dem.use_as_model_input=true — DEM must not be a model input.")
    run_tag = config.get("run_tag", args.config.stem)
    run_dir = Path(config["run_dir"])
    ckpt_path = args.ckpt or (run_dir / "checkpoints" / "best_checkpoint.pt")
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}"); return 1

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))
    out_dir = args.out_dir; out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}   s0={args.s0} tau={args.tau} thr={args.thr}   out: {out_dir}")

    assembler = InputAssembler(config, device)
    model_cfg = dict(config["model"])
    model_cfg.setdefault("in_chans", sum(len(config["data"]["init_args"]["means"][m])
                                         for m in config["data"]["init_args"]["modalities"]))
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_segman(model_cfg).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Model: SegMAN-{model_cfg.get('variant','s')} best_ep={ckpt.get('best_epoch','?')} "
          f"best_miou={ckpt.get('best_validation_miou', float('nan'))}")

    splitmap = {"val": "valid", "valid": "valid", "test": "test", "bolivia": "bolivia"}
    results: dict[str, Any] = {}
    t0 = time.time()
    for sp in args.eval_splits:
        print(f"\n[collect] {sp} ...")
        tiles = collect_split(model, assembler, config, splitmap[sp], device,
                              args.s0, args.tau, args.thr, args.max_tiles)
        agg = aggregate(tiles, args.n_boot, args.seed)
        results[sp] = agg
        pl = agg["pooled"]; uv = agg["useful_violation"]; pr = agg["paired"]
        print(f"  tiles={pr['n_tiles']}  VF(y)_w={pl['vf_gt_weighted']:.5f}  "
              f"VF(pred)_w={pl['vf_pred_weighted']:.5f}  H_R_w={pl['H_R_weighted']:+.5f}  "
              f"(pred cleaner on {pr['pct_tiles_pred_cleaner_weighted']:.0f}% tiles)")
        print(f"  useful-viol (binary set n={uv['binary_set']['n_active']}): "
              f"useful={uv['binary_set']['useful_rate']:.3f}  "
              f"harmful={uv['binary_set']['harmful_rate']:.3f}  "
              f"label-endorsed={uv['binary_set']['label_endorsed_rate']:.3f}")
        print(f"  useful-viol (hinge set  n={uv['hinge_set']['n_active']}): "
              f"useful={uv['hinge_set']['useful_rate']:.3f}  "
              f"harmful={uv['hinge_set']['harmful_rate']:.3f}  "
              f"label-endorsed={uv['hinge_set']['label_endorsed_rate']:.3f}")
    print(f"\n[done] {time.time()-t0:.1f}s")

    meta = {
        "run_tag": run_tag, "config": str(args.config), "checkpoint": str(ckpt_path),
        "best_epoch": ckpt.get("best_epoch"), "best_validation_miou": ckpt.get("best_validation_miou"),
        "s0": args.s0, "tau": args.tau, "thr": args.thr, "eval_splits": args.eval_splits,
        "vf_routine": "diagnose_gt_topographic_consistency._compute_d8 (D8 steepest descent, edge pad)",
    }
    payload = {"meta": meta, "results": _clean(results)}
    (out_dir / "native_vf_headroom_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "native_vf_headroom_report.md").write_text(_report_md(meta, results), encoding="utf-8")

    print("\n" + "=" * 70)
    print("  NATIVE VF HEADROOM + USEFUL VIOLATION RATE — SUMMARY")
    print("=" * 70)
    for sp in args.eval_splits:
        pl = results[sp]["pooled"]; uv = results[sp]["useful_violation"]["binary_set"]
        verdict = "no headroom (pred already cleaner-or-equal)" if pl["H_R_weighted"] <= 0 else "some headroom"
        print(f"[{sp:8s}] H_R_w={pl['H_R_weighted']:+.5f} -> {verdict};  "
              f"of active corrections {uv['harmful_rate']*100:.0f}% remove TRUE water "
              f"({uv['label_endorsed_rate']*100:.0f}% GT-endorsed)")
    print(f"\nOutputs in {out_dir}")
    return 0


def _clean(o: Any) -> Any:
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, (np.generic,)):
        return o.item()
    return o


def _f(v, d=5):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "N/A"
    return f"{v:.{d}f}"


def _report_md(meta, results) -> str:
    L = ["# Native D8 Violation-Fraction Headroom + Useful Violation Rate", "",
         "**Inference-only. SegMAN frozen. DEM never a model input — used only post-hoc to score violations.**", "",
         "## What this measures",
         "- **H_R = VF(p_hat_0) − VF(y)** in the D8 loss's own units (slope-weighted downstream violation",
         "  fraction). `H_R ≤ 0` ⇒ the baseline prediction already violates the constraint *less than the",
         "  labels* ⇒ no native headroom for any constraint monotone in this VF.",
         "- **Useful violation rate**: among the prediction's D8-active pixels (predicted water perched above",
         "  a drier downstream neighbor — the set the loss pushes down), the fraction whose suppression is",
         "  *useful* (GT dry, a real false positive) vs *harmful* (GT water, removes a true positive), and the",
         "  *label-endorsed* fraction where the GT itself violates D8 (loss penalizes a label-correct pixel).", "",
         f"- Checkpoint: `{meta['checkpoint']}` (best_ep={meta['best_epoch']}, best_miou={_f(meta['best_validation_miou'],4)})",
         f"- D8 params: s0={meta['s0']}, tau={meta['tau']}, threshold={meta['thr']} (match D8DownstreamLoss defaults).", "",
         "## (1) Native VF headroom  H_R = VF(pred) − VF(GT)", "",
         "| split | VF(y) wtd | VF(pred) wtd | **H_R wtd** | 95% CI (per-tile) | % tiles pred cleaner | VF(y) unwtd | VF(pred) unwtd | H_R unwtd |",
         "|-------|----------:|-------------:|------------:|-------------------|---------------------:|------------:|---------------:|----------:|"]
    for sp, r in results.items():
        pl = r["pooled"]; pr = r["paired"]; ci = pr["H_R_weighted_pertile_ci"]
        L.append(f"| {sp} | {_f(pl['vf_gt_weighted'])} | {_f(pl['vf_pred_weighted'])} | "
                 f"**{_f(pl['H_R_weighted'])}** | [{_f(ci[0])}, {_f(ci[2])}] | "
                 f"{pr['pct_tiles_pred_cleaner_weighted']:.0f}% | {_f(pl['vf_gt_unweighted'])} | "
                 f"{_f(pl['vf_pred_unweighted'])} | {_f(pl['H_R_unweighted'])} |")
    L += ["", "_H_R ≤ 0 ⇒ baseline already at-or-below the labels' violation level → no native headroom._", "",
          "## (2) Useful violation rate (does acting on D8-active pixels help labels?)", "",
          "**Binary violation set** (predicted water_i, predicted dry downstream, drop>0):", "",
          "| split | n active | useful (GT dry, FP→remove) | harmful (GT water, TP→remove) | label-endorsed (GT also violates) |",
          "|-------|---------:|---------------------------:|------------------------------:|----------------------------------:|"]
    for sp, r in results.items():
        u = r["useful_violation"]["binary_set"]
        L.append(f"| {sp} | {u['n_active']} | {_f(u['useful_rate'],3)} | {_f(u['harmful_rate'],3)} | {_f(u['label_endorsed_rate'],3)} |")
    L += ["", "**Hinge-active set** (p_i − p_downstream > tau — the set the loss gradient actually acts on):", "",
          "| split | n active | useful | harmful | label-endorsed |",
          "|-------|---------:|-------:|--------:|---------------:|"]
    for sp, r in results.items():
        u = r["useful_violation"]["hinge_set"]
        L.append(f"| {sp} | {u['n_active']} | {_f(u['useful_rate'],3)} | {_f(u['harmful_rate'],3)} | {_f(u['label_endorsed_rate'],3)} |")
    L += ["", "## Interpretation",
          "- If **H_R ≤ 0** and the **harmful rate dominates** (most D8-active pixels are true water, often",
          "  GT-endorsed), then the constraint's active corrections move predictions *away* from the labels.",
          "  This empirically closes the 'T is a partial statistic' loophole for the D8 / VF-monotone class:",
          "  the spatial corrections the constraint makes are predominantly label-harmful here.",
          "- This does NOT cover constraints built on orthogonal statistics (flow connectivity, accumulation),",
          "  nor the true physical flood state (everything is relative to reference labels y).", "",
          "---", "*Generated by `experiments_cvpr/segman/tools/native_vf_headroom_test.py`.*"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
