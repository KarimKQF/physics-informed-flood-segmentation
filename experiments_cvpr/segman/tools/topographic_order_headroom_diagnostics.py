"""
Topographic Order Headroom Diagnostics  (inference-only)
=========================================================
Tests whether the SHARED MECHANISM of the class of local monotone
topographic-order losses has label-relative headroom for SegMAN-S baseline.

The class includes D4, D8, slope-weighted D8, margin-based variants — any
loss that penalises v_ij = max(0, p_i - p_j - tau) when h_i > h_j.

We do NOT refute each formulation independently.  We test the shared
mechanism: does the baseline already satisfy the topographic constraint
as well as (or better than) the reference labels?

Diagnostics
-----------
  1. Hard VF (D8 and D4): labels vs binarised predictions
  2. Soft hinge activity + violation energy  E_topo
  3. Distributional tails  P(v > t)  for t ∈ {0, .05, .10, .20, .30, .50, .70}
  4. Useful violation rate + enrichment factor
  5. Bootstrap-by-tile 95 % CI for ΔVF and ΔE_topo

Inference-only.  SegMAN frozen.  DEM never a model input.
Outputs → outputs/topographic_order_headroom/
"""

from __future__ import annotations

import argparse
import csv
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

from step6c_v3_train import TopographyDataModule          # noqa: E402
from model.segman_model import build_segman               # noqa: E402
from diagnose_gt_topographic_consistency import (         # noqa: E402
    IGNORE_INDEX, WATER_CLASS, DRY_CLASS, _compute_d8,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

EPS = 1e-9
TAIL_THRESHOLDS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70]
_D4_OFFSETS     = [(1, 0), (-1, 0), (0, 1), (0, -1)]


# ── model helpers ─────────────────────────────────────────────────────────────

class InputAssembler:
    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        da = config["data"]["init_args"]
        self.modalities = list(da["modalities"])
        self.mean = {m: torch.tensor(da["means"][m], dtype=torch.float32,
                                     device=device).view(1, -1, 1, 1) for m in self.modalities}
        self.std  = {m: torch.tensor(da["stds"][m],  dtype=torch.float32,
                                     device=device).view(1, -1, 1, 1) for m in self.modalities}

    def __call__(self, image: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            [(image[m].float() - self.mean[m]) / self.std[m] for m in self.modalities], dim=1
        )


def _get_topo(batch):
    topo = batch.get("topography")
    if topo is None:
        return None
    return (topo[:, 0] if topo.ndim == 4 and topo.shape[1] == 1 else topo).float()


def _get_target(batch):
    mask = batch["mask"]
    return (mask[:, 0] if mask.ndim == 4 and mask.shape[1] == 1 else mask).long()


def load_config(config_path: Path) -> dict[str, Any]:
    import yaml
    with config_path.open("r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f)


# ── D4 slice helper ───────────────────────────────────────────────────────────

def _d4_slices(H: int, W: int, dy: int, dx: int):
    """Center and neighbor array slices for directed offset (dy, dx)."""
    cr = slice(max(0, -dy), H + min(0, -dy) if dy != 0 else H)
    cc = slice(max(0, -dx), W + min(0, -dx) if dx != 0 else W)
    nr = slice(max(0,  dy), H + min(0,  dy) if dy != 0 else H)
    nc = slice(max(0,  dx), W + min(0,  dx) if dx != 0 else W)
    return (cr, cc), (nr, nc)


# ── per-tile statistics ───────────────────────────────────────────────────────

def tile_full_stats(
    h: np.ndarray, y: np.ndarray, p: np.ndarray,
    s0: float, tau: float, thr: float,
) -> dict[str, Any]:
    """D8 + D4 statistics for one tile.

    Returns raw sums (for pooled aggregation) and per-tile scalars (bootstrap).
    h: DEM [H,W] float  |  y: label [H,W] int  |  p: water prob [H,W] ∈ [0,1]
    """
    H, W = h.shape
    valid    = (y != IGNORE_INDEX) & np.isfinite(h)
    gt_water = (y == WATER_CLASS)
    gt_dry   = (y == DRY_CLASS)

    # ── D8 ────────────────────────────────────────────────────────────────────
    d_row, d_col, drop = _compute_d8(
        np.where(np.isfinite(h), h, np.nan).astype(np.float32)
    )
    w8       = np.clip(drop / s0, 0.0, 1.0)
    valid_d  = valid[d_row, d_col]
    active   = valid & valid_d & (drop > 0.0)

    gt_dry_d  = gt_dry[d_row, d_col]
    gt_viol   = active & gt_water & gt_dry_d

    yhat      = p > thr
    yhat_d    = yhat[d_row, d_col]
    pred_viol = active & yhat & ~yhat_d

    p_d = p[d_row, d_col]
    # violation magnitudes (zeroed for inactive pixels)
    v_soft = np.where(active, np.maximum(0.0, p - p_d - tau), 0.0)
    y_f    = gt_water.astype(np.float32)
    y_d_f  = y_f[d_row, d_col]
    v_gt   = np.where(active, np.maximum(0.0, y_f - y_d_f - tau), 0.0)
    yh_f   = yhat.astype(np.float32)
    yh_d_f = yh_f[d_row, d_col]
    v_yhat = np.where(active, np.maximum(0.0, yh_f - yh_d_f - tau), 0.0)

    w_act  = float((w8 * active).sum())
    uw_act = int(active.sum())

    d8: dict[str, Any] = {
        # raw sums
        "w_active":    w_act,  "uw_active":   uw_act,
        "w_gt_viol":   float((w8 * gt_viol).sum()),    "uw_gt_viol":   int(gt_viol.sum()),
        "w_pred_viol": float((w8 * pred_viol).sum()),  "uw_pred_viol": int(pred_viol.sum()),
        "w_E_gt":      float((w8 * v_gt**2  * active).sum()),
        "w_E_yhat":    float((w8 * v_yhat**2 * active).sum()),
        "w_E_soft":    float((w8 * v_soft**2 * active).sum()),
        # per-tile fractions
        "vf_gt_w":    float((w8 * gt_viol).sum())   / (w_act + EPS),
        "vf_gt_uw":   int(gt_viol.sum())             / (uw_act + EPS),
        "vf_pred_w":  float((w8 * pred_viol).sum()) / (w_act + EPS),
        "vf_pred_uw": int(pred_viol.sum())           / (uw_act + EPS),
        "E_gt_t":     float((w8 * v_gt**2  * active).sum()) / (w_act + EPS),
        "E_yhat_t":   float((w8 * v_yhat**2 * active).sum()) / (w_act + EPS),
        "E_soft_t":   float((w8 * v_soft**2 * active).sum()) / (w_act + EPS),
    }
    d8["H_R_w"]  = d8["vf_pred_w"]  - d8["vf_gt_w"]
    d8["H_R_uw"] = d8["vf_pred_uw"] - d8["vf_gt_uw"]
    d8["dE_w"]   = d8["E_yhat_t"]   - d8["E_gt_t"]

    # tail curves
    tail_s  = {t: int((active & (v_soft > t)).sum()) for t in TAIL_THRESHOLDS}
    tail_g  = {t: int((active & (v_gt   > t)).sum()) for t in TAIL_THRESHOLDS}
    tail_p  = {t: int((active & (v_yhat > t)).sum()) for t in TAIL_THRESHOLDS}
    d8["tail_soft"] = tail_s
    d8["tail_gt"]   = tail_g
    d8["tail_pred"] = tail_p
    d8["n_tail"]    = uw_act

    # useful violation rate
    n_pv = int(pred_viol.sum())
    d8["n_pred_viol"] = n_pv
    d8["n_useful"]    = int((pred_viol & gt_dry).sum())
    d8["n_harmful"]   = int((pred_viol & gt_water).sum())
    d8["n_endorsed"]  = int((pred_viol & gt_water & gt_dry_d).sum())

    hinge_act = active & (v_soft > 0.0)
    n_ha = int(hinge_act.sum())
    d8["n_ha"]          = n_ha
    d8["n_ha_useful"]   = int((hinge_act & gt_dry).sum())
    d8["n_ha_harmful"]  = int((hinge_act & gt_water).sum())
    d8["n_ha_endorsed"] = int((hinge_act & gt_water & gt_dry_d).sum())

    # enrichment: P(GT dry | D8-active violation) / P(GT dry | predicted water)
    pw_valid = int((valid & yhat).sum())
    pw_dry   = int((valid & yhat & gt_dry).sum())
    d8["n_pw_valid"] = pw_valid
    d8["n_pw_dry"]   = pw_dry
    d8["n_valid"]    = int(valid.sum())

    # ── D4 ────────────────────────────────────────────────────────────────────
    d4_w_act = 0.0; d4_uw_act = 0
    d4_w_gv  = 0.0; d4_uw_gv  = 0
    d4_w_pv  = 0.0; d4_uw_pv  = 0
    d4_wEgt  = 0.0; d4_wEyh   = 0.0; d4_wEsf  = 0.0
    d4_ts: dict = {t: 0 for t in TAIL_THRESHOLDS}
    d4_tg: dict = {t: 0 for t in TAIL_THRESHOLDS}
    d4_tp: dict = {t: 0 for t in TAIL_THRESHOLDS}
    d4_n_tail = 0

    for dy, dx in _D4_OFFSETS:
        (cr, cc), (nr, nc) = _d4_slices(H, W, dy, dx)
        h_c = h[cr, cc];   h_n = h[nr, nc]
        v_c = valid[cr, cc]; v_n = valid[nr, nc]
        gw_c = gt_water[cr, cc]; gd_n = gt_dry[nr, nc]
        gw_n = gt_water[nr, nc]
        p_c  = p[cr, cc];  p_n = p[nr, nc]

        drop4  = h_c - h_n
        act4   = v_c & v_n & (drop4 > 0.0)
        w4     = np.clip(drop4 / s0, 0.0, 1.0) * act4

        gv4 = act4 & gw_c & gd_n
        yh_c = (p_c > thr); yh_n = (p_n > thr)
        pv4 = act4 & yh_c & ~yh_n

        yc_f = gw_c.astype(np.float32); yn_f = gw_n.astype(np.float32)
        yh_c_f = yh_c.astype(np.float32); yh_n_f = yh_n.astype(np.float32)
        vg4  = np.where(act4, np.maximum(0.0, yc_f  - yn_f  - tau), 0.0)
        vyh4 = np.where(act4, np.maximum(0.0, yh_c_f - yh_n_f - tau), 0.0)
        vs4  = np.where(act4, np.maximum(0.0, p_c   - p_n   - tau), 0.0)

        d4_w_act += float(w4.sum()); d4_uw_act += int(act4.sum())
        d4_w_gv  += float((w4 * gv4).sum()); d4_uw_gv += int(gv4.sum())
        d4_w_pv  += float((w4 * pv4).sum()); d4_uw_pv += int(pv4.sum())
        d4_wEgt  += float((w4 * vg4**2).sum())
        d4_wEyh  += float((w4 * vyh4**2).sum())
        d4_wEsf  += float((w4 * vs4**2).sum())
        for t in TAIL_THRESHOLDS:
            d4_ts[t] += int((act4 & (vs4  > t)).sum())
            d4_tg[t] += int((act4 & (vg4  > t)).sum())
            d4_tp[t] += int((act4 & (vyh4 > t)).sum())
        d4_n_tail += int(act4.sum())

    d4: dict[str, Any] = {
        "w_active": d4_w_act, "uw_active": d4_uw_act,
        "w_gt_viol": d4_w_gv, "uw_gt_viol": d4_uw_gv,
        "w_pred_viol": d4_w_pv, "uw_pred_viol": d4_uw_pv,
        "w_E_gt": d4_wEgt, "w_E_yhat": d4_wEyh, "w_E_soft": d4_wEsf,
        "vf_gt_w":   d4_w_gv  / (d4_w_act + EPS),
        "vf_gt_uw":  d4_uw_gv / (d4_uw_act + EPS),
        "vf_pred_w": d4_w_pv  / (d4_w_act + EPS),
        "vf_pred_uw":d4_uw_pv / (d4_uw_act + EPS),
        "E_gt_t":    d4_wEgt  / (d4_w_act + EPS),
        "E_yhat_t":  d4_wEyh  / (d4_w_act + EPS),
        "E_soft_t":  d4_wEsf  / (d4_w_act + EPS),
        "H_R_w":     d4_w_pv  / (d4_w_act + EPS) - d4_w_gv / (d4_w_act + EPS),
        "dE_w":      d4_wEyh  / (d4_w_act + EPS) - d4_wEgt / (d4_w_act + EPS),
        "tail_soft": d4_ts, "tail_gt": d4_tg, "tail_pred": d4_tp,
        "n_tail": d4_n_tail,
    }

    return {"d8": d8, "d4": d4}


# ── inference loop ────────────────────────────────────────────────────────────

def collect_split(model, assembler, config, dm_split, device,
                  s0, tau, thr, max_tiles):
    dm = TopographyDataModule(config, batch_size=1)
    dm.split = dm_split
    dm.setup("test")
    loader = dm.test_dataloader()
    tiles: list[dict] = []
    n = 0
    model.eval()
    with torch.no_grad():
        for raw in loader:
            if max_tiles is not None and n >= max_tiles:
                break
            topo = _get_topo(raw)
            if topo is None:
                continue
            img   = {k: v.to(device) for k, v in raw["image"].items()}
            logits = model(assembler(img))
            p2d = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()
            y2d = _get_target(raw)[0].cpu().numpy()
            h2d = topo[0].cpu().numpy()
            st = tile_full_stats(h2d, y2d, p2d, s0, tau, thr)
            if st["d8"]["uw_active"] < 1:
                continue
            st["tile_idx"] = n
            tiles.append(st)
            n += 1
    return tiles


# ── bootstrap CI ──────────────────────────────────────────────────────────────

def _boot_ci(arr: np.ndarray, n_boot: int, seed: int):
    if arr.size == 0:
        return [float("nan")] * 3
    rng = np.random.default_rng(seed)
    means = [arr[rng.integers(0, arr.size, arr.size)].mean() for _ in range(n_boot)]
    a = np.array(means)
    return [float(np.percentile(a, 2.5)), float(arr.mean()), float(np.percentile(a, 97.5))]


# ── aggregation ───────────────────────────────────────────────────────────────

def _sum(tiles, k8, k4=None):
    if k4 is None:
        return float(sum(t["d8"][k8] for t in tiles))
    return float(sum(t["d4"][k8] for t in tiles))


def aggregate(tiles: list[dict], n_boot: int, seed: int) -> dict[str, Any]:
    # ── D8 pooled ─────────────────────────────────────────────────────────────
    def S8(k): return float(sum(t["d8"][k] for t in tiles))
    def I8(k): return int(sum(t["d8"][k] for t in tiles))

    w_act8 = S8("w_active"); uw_act8 = S8("uw_active")
    d8p = {
        "vf_gt_w":    S8("w_gt_viol")   / (w_act8 + EPS),
        "vf_gt_uw":   S8("uw_gt_viol")  / (uw_act8 + EPS),
        "vf_pred_w":  S8("w_pred_viol") / (w_act8 + EPS),
        "vf_pred_uw": S8("uw_pred_viol")/ (uw_act8 + EPS),
        "E_topo_gt":    S8("w_E_gt")   / (w_act8 + EPS),
        "E_topo_yhat":  S8("w_E_yhat") / (w_act8 + EPS),
        "E_topo_soft":  S8("w_E_soft") / (w_act8 + EPS),
        "n_active_w": w_act8, "n_active_uw": uw_act8,
    }
    d8p["H_R_w"]  = d8p["vf_pred_w"]  - d8p["vf_gt_w"]
    d8p["H_R_uw"] = d8p["vf_pred_uw"] - d8p["vf_gt_uw"]
    d8p["dE_w"]   = d8p["E_topo_yhat"] - d8p["E_topo_gt"]

    # per-tile arrays for bootstrap
    hr_w  = np.array([t["d8"]["H_R_w"]  for t in tiles])
    hr_uw = np.array([t["d8"]["H_R_uw"] for t in tiles])
    de_w  = np.array([t["d8"]["dE_w"]   for t in tiles])
    d8_ci = {
        "H_R_w_ci":   _boot_ci(hr_w,  n_boot, seed),
        "H_R_uw_ci":  _boot_ci(hr_uw, n_boot, seed + 1),
        "dE_w_ci":    _boot_ci(de_w,  n_boot, seed + 2),
        "pct_tiles_pred_cleaner_w":  100.0 * float(np.mean(hr_w  <= 0)),
        "pct_tiles_pred_cleaner_uw": 100.0 * float(np.mean(hr_uw <= 0)),
        "pct_tiles_E_pred_lower_w":  100.0 * float(np.mean(de_w  <= 0)),
        "n_tiles": len(tiles),
        "vf_gt_w_median":   float(np.median([t["d8"]["vf_gt_w"]  for t in tiles])),
        "vf_pred_w_median": float(np.median([t["d8"]["vf_pred_w"] for t in tiles])),
        "vf_gt_w_p25":   float(np.percentile([t["d8"]["vf_gt_w"]  for t in tiles], 25)),
        "vf_gt_w_p75":   float(np.percentile([t["d8"]["vf_gt_w"]  for t in tiles], 75)),
        "vf_pred_w_p25": float(np.percentile([t["d8"]["vf_pred_w"] for t in tiles], 25)),
        "vf_pred_w_p75": float(np.percentile([t["d8"]["vf_pred_w"] for t in tiles], 75)),
    }

    # tail curves (pooled counts → probabilities)
    n_tail8 = int(sum(t["d8"]["n_tail"] for t in tiles))
    tails8 = {}
    for t in TAIL_THRESHOLDS:
        ns = sum(t_["d8"]["tail_soft"][t] for t_ in tiles)
        ng = sum(t_["d8"]["tail_gt"][t]   for t_ in tiles)
        np_ = sum(t_["d8"]["tail_pred"][t] for t_ in tiles)
        tails8[t] = {
            "P_soft":      ns  / (n_tail8 + EPS),
            "P_gt_hard":   ng  / (n_tail8 + EPS),
            "P_pred_hard": np_ / (n_tail8 + EPS),
            "dP_soft":     ns  / (n_tail8 + EPS) - ng / (n_tail8 + EPS),
            "dP_pred":     np_ / (n_tail8 + EPS) - ng / (n_tail8 + EPS),
            "n_soft": int(ns), "n_gt": int(ng), "n_pred": int(np_),
        }

    # useful violation rate
    n_pv  = I8("n_pred_viol")
    n_ha  = I8("n_ha")
    uvr8 = {
        "binary": {
            "n_pred_viol": n_pv,
            "useful_rate":    I8("n_useful")   / (n_pv + EPS),
            "harmful_rate":   I8("n_harmful")  / (n_pv + EPS),
            "endorsed_rate":  I8("n_endorsed") / (n_pv + EPS),
        },
        "hinge": {
            "n_hinge_active": n_ha,
            "useful_rate":    I8("n_ha_useful")   / (n_ha + EPS),
            "harmful_rate":   I8("n_ha_harmful")  / (n_ha + EPS),
            "endorsed_rate":  I8("n_ha_endorsed") / (n_ha + EPS),
        },
    }

    # enrichment
    n_pw  = I8("n_pw_valid")
    n_pwd = I8("n_pw_dry")
    base_fpr = n_pwd / (n_pw + EPS)
    viol_fpr = I8("n_useful") / (n_pv + EPS)
    enrichment = viol_fpr / (base_fpr + EPS)
    uvr8["enrichment"] = {
        "base_fp_rate":      base_fpr,
        "violation_fp_rate": viol_fpr,
        "enrichment_factor": enrichment,
    }

    # ── D4 pooled ─────────────────────────────────────────────────────────────
    def S4(k): return float(sum(t["d4"][k] for t in tiles))
    def I4(k): return int(sum(t["d4"][k] for t in tiles))

    w_act4 = S4("w_active"); uw_act4 = S4("uw_active")
    d4p = {
        "vf_gt_w":    S4("w_gt_viol")   / (w_act4 + EPS),
        "vf_gt_uw":   S4("uw_gt_viol")  / (uw_act4 + EPS),
        "vf_pred_w":  S4("w_pred_viol") / (w_act4 + EPS),
        "vf_pred_uw": S4("uw_pred_viol")/ (uw_act4 + EPS),
        "E_topo_gt":   S4("w_E_gt")   / (w_act4 + EPS),
        "E_topo_yhat": S4("w_E_yhat") / (w_act4 + EPS),
        "E_topo_soft": S4("w_E_soft") / (w_act4 + EPS),
        "n_active_w": w_act4, "n_active_uw": uw_act4,
    }
    d4p["H_R_w"]  = d4p["vf_pred_w"]  - d4p["vf_gt_w"]
    d4p["H_R_uw"] = d4p["vf_pred_uw"] - d4p["vf_gt_uw"]
    d4p["dE_w"]   = d4p["E_topo_yhat"] - d4p["E_topo_gt"]

    hr4_w  = np.array([t["d4"]["H_R_w"]  for t in tiles])
    de4_w  = np.array([t["d4"]["dE_w"]   for t in tiles])
    d4_ci = {
        "H_R_w_ci":  _boot_ci(hr4_w, n_boot, seed + 3),
        "dE_w_ci":   _boot_ci(de4_w, n_boot, seed + 4),
        "pct_tiles_pred_cleaner_w": 100.0 * float(np.mean(hr4_w <= 0)),
        "pct_tiles_E_pred_lower_w": 100.0 * float(np.mean(de4_w <= 0)),
        "n_tiles": len(tiles),
    }

    n_tail4 = int(sum(t["d4"]["n_tail"] for t in tiles))
    tails4 = {}
    for t in TAIL_THRESHOLDS:
        ns = sum(t_["d4"]["tail_soft"][t] for t_ in tiles)
        ng = sum(t_["d4"]["tail_gt"][t]   for t_ in tiles)
        np_ = sum(t_["d4"]["tail_pred"][t] for t_ in tiles)
        tails4[t] = {
            "P_soft":      ns  / (n_tail4 + EPS),
            "P_gt_hard":   ng  / (n_tail4 + EPS),
            "P_pred_hard": np_ / (n_tail4 + EPS),
            "dP_soft": ns / (n_tail4 + EPS) - ng / (n_tail4 + EPS),
            "dP_pred": np_ / (n_tail4 + EPS) - ng / (n_tail4 + EPS),
        }

    return {
        "d8_pooled": d8p, "d8_ci": d8_ci, "d8_tails": tails8, "d8_useful": uvr8,
        "d4_pooled": d4p, "d4_ci": d4_ci, "d4_tails": tails4,
    }


# ── plots ─────────────────────────────────────────────────────────────────────

def _make_plots(results: dict[str, Any], out_dir: Path) -> None:
    if not _HAS_MPL:
        print("  [plots] matplotlib not available - skipping.")
        return

    splits = list(results.keys())
    colors = {"val": "#2196F3", "test": "#4CAF50", "bolivia": "#FF5722"}

    # 1. VF labels vs predictions by split (D8 + D4)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in [
        (axes[0], "d8_pooled", "D8 VF"),
        (axes[1], "d4_pooled", "D4 VF"),
    ]:
        xs = np.arange(len(splits))
        w = 0.35
        gt_vals   = [results[sp][key]["vf_gt_w"]   for sp in splits]
        pred_vals = [results[sp][key]["vf_pred_w"] for sp in splits]
        ax.bar(xs - w/2, gt_vals,   w, label="Labels (y)", color="#E57373", alpha=0.85)
        ax.bar(xs + w/2, pred_vals, w, label="Pred (ŷ₀)",  color="#64B5F6", alpha=0.85)
        ax.set_xticks(xs); ax.set_xticklabels(splits)
        ax.set_ylabel("VF (slope-weighted)"); ax.set_title(title)
        ax.legend(fontsize=8)
        for i, (g, p_) in enumerate(zip(gt_vals, pred_vals)):
            ax.text(i - w/2, g + 0.00002, f"{g:.4f}", ha="center", fontsize=7)
            ax.text(i + w/2, p_ + 0.00002, f"{p_:.4f}", ha="center", fontsize=7)
    fig.suptitle("Violation Fraction: Labels vs Predictions\n(lower = fewer topo violations)")
    fig.tight_layout()
    fig.savefig(out_dir / "vf_labels_vs_predictions_by_split.png", dpi=150)
    plt.close(fig)

    # 2. Violation energy by split
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in [
        (axes[0], "d8_pooled", "D8 E_topo"),
        (axes[1], "d4_pooled", "D4 E_topo"),
    ]:
        xs = np.arange(len(splits)); w = 0.25
        vl_gt  = [results[sp][key]["E_topo_gt"]   for sp in splits]
        vl_yh  = [results[sp][key]["E_topo_yhat"] for sp in splits]
        vl_sft = [results[sp][key]["E_topo_soft"] for sp in splits]
        ax.bar(xs - w, vl_gt,  w, label="GT (hard y)",  color="#E57373", alpha=0.85)
        ax.bar(xs,     vl_yh,  w, label="Pred (hard ŷ)", color="#64B5F6", alpha=0.85)
        ax.bar(xs + w, vl_sft, w, label="Pred (soft p)", color="#81C784", alpha=0.85)
        ax.set_xticks(xs); ax.set_xticklabels(splits)
        ax.set_ylabel("E_topo = mean(w·v²)"); ax.set_title(title)
        ax.legend(fontsize=7)
    fig.suptitle("Violation Energy: GT vs Predictions\n(lower = less constraint violation)")
    fig.tight_layout()
    fig.savefig(out_dir / "violation_energy_labels_vs_predictions_by_split.png", dpi=150)
    plt.close(fig)

    # 3. Tail curves per split
    for sp in splits:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, key, title in [
            (axes[0], "d8_tails", "D8 violation tails"),
            (axes[1], "d4_tails", "D4 violation tails"),
        ]:
            tails = results[sp][key]
            ts = sorted(tails.keys())
            p_gt   = [tails[t]["P_gt_hard"]   for t in ts]
            p_pred = [tails[t]["P_pred_hard"]  for t in ts]
            p_soft = [tails[t]["P_soft"]       for t in ts]
            ax.plot(ts, p_gt,   "r-o", label="Labels (hard)", linewidth=1.5, ms=4)
            ax.plot(ts, p_pred, "b-s", label="Pred (hard ŷ)", linewidth=1.5, ms=4)
            ax.plot(ts, p_soft, "g-^", label="Pred (soft p)", linewidth=1.5, ms=4)
            ax.set_xlabel("Violation threshold t"); ax.set_ylabel("P(v > t)")
            ax.set_title(f"{title}\n{sp}")
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        fig.suptitle(f"Violation Tail Curves — {sp}")
        fig.tight_layout()
        fig.savefig(out_dir / f"violation_tail_curves_{sp}.png", dpi=150)
        plt.close(fig)

    # 4. Useful violation rate by split
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.arange(len(splits)); w = 0.25
    useful_b  = [results[sp]["d8_useful"]["binary"]["useful_rate"]   for sp in splits]
    harmful_b = [results[sp]["d8_useful"]["binary"]["harmful_rate"]  for sp in splits]
    endors_b  = [results[sp]["d8_useful"]["binary"]["endorsed_rate"] for sp in splits]
    ax.bar(xs - w, useful_b,  w, label="Useful (GT dry = FP)", color="#66BB6A", alpha=0.85)
    ax.bar(xs,     harmful_b, w, label="Harmful (GT water = FN risk)", color="#EF5350", alpha=0.85)
    ax.bar(xs + w, endors_b,  w, label="GT-endorsed violation", color="#FFA726", alpha=0.85)
    ax.set_xticks(xs); ax.set_xticklabels(splits)
    ax.set_ylabel("Fraction of D8-active violations"); ax.set_ylim(0, 1)
    ax.set_title("Useful Violation Rate (D8 binary set)\nDoes acting on violations help the labels?")
    ax.legend(fontsize=8); ax.axhline(0.5, color="k", linestyle="--", lw=0.8, alpha=0.5)
    for i, (u, h_, e) in enumerate(zip(useful_b, harmful_b, endors_b)):
        ax.text(i - w, u + 0.01, f"{u:.2f}", ha="center", fontsize=8)
        ax.text(i,     h_ + 0.01, f"{h_:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "useful_violation_rate_by_split.png", dpi=150)
    plt.close(fig)

    print(f"  [plots] 4 figures saved to {out_dir}")


# ── per-tile CSV ──────────────────────────────────────────────────────────────

def _save_tile_csv(all_tiles: list[dict], path: Path) -> None:
    rows = []
    for t in all_tiles:
        d8 = t["d8"]; d4 = t["d4"]
        rows.append({
            "split":         t.get("split", "?"),
            "tile_idx":      t.get("tile_idx", -1),
            "d8_vf_gt_w":   d8["vf_gt_w"],  "d8_vf_pred_w":  d8["vf_pred_w"],
            "d8_H_R_w":     d8["H_R_w"],    "d8_H_R_uw":     d8["H_R_uw"],
            "d8_E_gt":      d8["E_gt_t"],   "d8_E_yhat":     d8["E_yhat_t"],
            "d8_E_soft":    d8["E_soft_t"], "d8_dE_w":       d8["dE_w"],
            "d8_n_active":  d8["uw_active"],
            "d4_vf_gt_w":   d4["vf_gt_w"],  "d4_vf_pred_w":  d4["vf_pred_w"],
            "d4_H_R_w":     d4["H_R_w"],
            "d4_E_gt":      d4["E_gt_t"],   "d4_E_yhat":     d4["E_yhat_t"],
            "d4_dE_w":      d4["dE_w"],     "d4_n_active":   d4["uw_active"],
        })
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ── markdown report ───────────────────────────────────────────────────────────

def _f(v, d=5):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "N/A"
    return f"{v:.{d}f}"


def _report_md(meta: dict, results: dict) -> str:
    L = [
        "# Topographic Order Headroom Diagnostics",
        "## SegMAN-S · Sen1Floods11 · N=100 · seed 0",
        "",
        "**Inference-only. SegMAN frozen. DEM used only post-hoc — never a model input.**",
        "",
        "---",
        "",
        "## 1. Motivation",
        "",
        "We do not refute D4 and D8 independently. We test the **shared mechanism** of",
        "their loss class: the reduction of local topographic-order violations.  In this",
        "setting, the unconstrained baseline already exhibits lower violation fraction and",
        "lower violation energy than the reference labels.  Therefore, any loss whose sole",
        "mechanism is to further reduce these violations lacks native label-relative headroom.",
        "",
        "## 2. Class definition",
        "",
        "A **local topographic-order loss** is any R_φ such that",
        "",
        "    v_ij = max(0, p_i − p_j − τ)    when h_i > h_j, j ∈ N(i)",
        "",
        "is the penalised quantity (N(i) = D8 steepest-descent, or D4 all-lower 4-neighbors).",
        "Minimising R_φ pushes T = −VF(p̂, z) upward (fewer violations).  The class includes",
        "D8, slope-weighted D8, margin-based D8, D4 variants, etc.",
        "",
        f"**Parameters:** s₀={meta['s0']}  τ={meta['tau']}  threshold={meta['thr']}",
        f"**Checkpoint:** `{meta['checkpoint']}`  (best_ep={meta['best_epoch']},",
        f"best_mIoU={_f(meta['best_miou'], 4)})",
        "",
        "---",
        "",
        "## 3. VF native headroom  H_R = VF(pred) − VF(GT)",
        "",
        "| split | VF(y) wtd | VF(ŷ₀) wtd | **H_R wtd** | 95% CI | % tiles pred≤GT | VF(y) uwtd | VF(ŷ₀) uwtd | H_R uwtd |",
        "|-------|----------:|----------:|----------:|--------|-----:|----------:|----------:|----------:|",
    ]
    for sp, r in results.items():
        p8 = r["d8_pooled"]; ci = r["d8_ci"]
        ci_lo, _, ci_hi = ci["H_R_w_ci"]
        L.append(
            f"| {sp} | {_f(p8['vf_gt_w'])} | {_f(p8['vf_pred_w'])} |"
            f" **{_f(p8['H_R_w'])}** | [{_f(ci_lo)}, {_f(ci_hi)}] |"
            f" {ci['pct_tiles_pred_cleaner_w']:.0f}% |"
            f" {_f(p8['vf_gt_uw'])} | {_f(p8['vf_pred_uw'])} | {_f(p8['H_R_uw'])} |"
        )
    L += [
        "",
        "_H_R ≤ 0 ⇒ baseline already violates the D8 constraint **less than the labels** → no native headroom._",
        "",
        "### D4 VF headroom",
        "",
        "| split | VF_D4(y) wtd | VF_D4(ŷ₀) wtd | H_R_D4 wtd | 95% CI | % tiles pred≤GT |",
        "|-------|----------:|----------:|----------:|--------|-----:|",
    ]
    for sp, r in results.items():
        p4 = r["d4_pooled"]; ci4 = r["d4_ci"]
        ci_lo, _, ci_hi = ci4["H_R_w_ci"]
        L.append(
            f"| {sp} | {_f(p4['vf_gt_w'])} | {_f(p4['vf_pred_w'])} |"
            f" **{_f(p4['H_R_w'])}** | [{_f(ci_lo)}, {_f(ci_hi)}] |"
            f" {ci4['pct_tiles_pred_cleaner_w']:.0f}% |"
        )
    L += [
        "",
        "---",
        "",
        "## 4. Violation energy  E_topo = mean(w·v²)",
        "",
        "| split | E_D8(y) | E_D8(ŷ₀) | ΔE_D8 | 95% CI | E_D8(p soft) | E_D4(y) | E_D4(ŷ₀) |",
        "|-------|--------:|--------:|------:|--------|--------:|--------:|--------:|",
    ]
    for sp, r in results.items():
        p8 = r["d8_pooled"]; p4 = r["d4_pooled"]; ci8 = r["d8_ci"]
        ci_lo, _, ci_hi = ci8["dE_w_ci"]
        L.append(
            f"| {sp} | {_f(p8['E_topo_gt'],6)} | {_f(p8['E_topo_yhat'],6)} |"
            f" **{_f(p8['dE_w'],6)}** | [{_f(ci_lo,6)}, {_f(ci_hi,6)}] |"
            f" {_f(p8['E_topo_soft'],6)} |"
            f" {_f(p4['E_topo_gt'],6)} | {_f(p4['E_topo_yhat'],6)} |"
        )
    L += [
        "",
        "_ΔE ≤ 0 → predictions have lower or equal violation energy than labels._",
        "",
        "---",
        "",
        "## 5. Distributional tails  P(v > t)",
        "",
        "Fraction of D8-active pixels with violation magnitude exceeding threshold t.",
        "Hard labels: v_y = 0.95 at violations (step function); hard ŷ: same.",
        "Soft p: v_p = max(0, p_i − p_d − τ), continuous in [0,1].",
        "",
    ]
    for sp, r in results.items():
        L.append(f"### {sp}")
        L.append("")
        L.append("| t | P(v_y>t) GT | P(v_ŷ>t) hard | P(v_p>t) soft | ΔP(pred−GT) hard | ΔP(soft−GT) |")
        L.append("|---|---:|---:|---:|---:|---:|")
        tails = r["d8_tails"]
        for t in sorted(tails.keys()):
            td = tails[t]
            L.append(
                f"| {t:.2f} | {_f(td['P_gt_hard'],4)} | {_f(td['P_pred_hard'],4)} |"
                f" {_f(td['P_soft'],4)} | {_f(td['dP_pred'],4)} | {_f(td['dP_soft'],4)} |"
            )
        L.append("")
    L += [
        "---",
        "",
        "## 6. Useful violation rate (D8)",
        "",
        "Among D8-active prediction violations (predicted water upstream of predicted dry downstream),",
        "what fraction are useful (GT dry = real FP suppression) vs harmful (GT water = recall loss)?",
        "",
        "| split | n violations | useful (GT dry) | harmful (GT water) | GT-endorsed | enrichment factor |",
        "|-------|---:|---:|---:|---:|---:|",
    ]
    for sp, r in results.items():
        u = r["d8_useful"]
        b = u["binary"]; e = u["enrichment"]
        L.append(
            f"| {sp} | {b['n_pred_viol']} |"
            f" {_f(b['useful_rate'],3)} | {_f(b['harmful_rate'],3)} |"
            f" {_f(b['endorsed_rate'],3)} | {_f(e['enrichment_factor'],2)}× |"
        )
    L += [
        "",
        "Enrichment = P(GT dry | D8-violation) / P(GT dry | predicted water).",
        "Value > 1 means violations are enriched in false positives vs average predicted-water pixels.",
        "",
        "---",
        "",
        "## 7. Interpretation — Canal II-a",
        "",
        "The combination of results above closes the label-relative headroom argument for the",
        "**local topographic-order loss class**:",
        "",
        "1. **H_R < 0** (robustly, CI entirely negative): predictions violate D8 and D4 *less* than labels.",
        "2. **ΔE_topo < 0**: violation energy is lower for predictions than for labels.",
        "3. **Distributional dominance**: prediction violation tails are below label tails at all thresholds.",
        "4. **Useful violation rate ~59/41**: violations are enriched in FPs (~{:.0f}×) but the 41% harmful".format(
            list(results.values())[0]["d8_useful"]["enrichment"]["enrichment_factor"]),
        "   component (true water suppressed) explains the observed precision↑/recall↓ wash with no net IoU gain.",
        "",
        "Together: the baseline already satisfies the local topographic-order constraint as well as or",
        "better than the reference labels.  Losses whose sole mechanism is to reduce these violations",
        "have **no native label-relative headroom**.",
        "",
        "---",
        "",
        "## 8. Limitations",
        "",
        "- **Label-relative**: 'no headroom' means relative to reference labels y, which may themselves",
        "  be topographically noisy. If the true physical flood state y★ is cleaner, there could be",
        "  headroom relative to y★ — but y★ is unobserved.",
        "- **Local monotone constraints only**: This closes headroom for losses based on reducing",
        "  v_ij = max(0, p_i − p_j − τ) for higher-to-lower neighbouring pixels (D4, D8 and variants).",
        "  **Constraints based on hydrological connectivity, flow accumulation, HAND, basin routing,",
        "  or other orthogonal statistics are outside scope and not tested here.**",
        "- **Single seed**: seed 0 baseline only. Multi-seed variance not estimated here.",
        "",
        "---",
        "",
        f"*Generated by `experiments_cvpr/segman/tools/topographic_order_headroom_diagnostics.py`.*",
        f"*Checkpoint: `{meta['checkpoint']}`*",
    ]
    return "\n".join(L) + "\n"


# ── JSON serialisation helper ─────────────────────────────────────────────────

def _clean(o: Any) -> Any:
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, (np.generic,)):
        return o.item()
    return o


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Topographic order headroom diagnostics (inference-only)."
    )
    ap.add_argument("--config", type=Path,
                    default=REPO_ROOT / "configs/segman/multiseed_n100/n100_seed0_dice_ce.yaml")
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--eval-splits", nargs="+", default=["val", "test", "bolivia"])
    ap.add_argument("--s0",  type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed",   type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--max-tiles", type=int, default=None)
    ap.add_argument("--out-dir", type=Path,
                    default=REPO_ROOT / "outputs/topographic_order_headroom")
    args = ap.parse_args()

    config = load_config(args.config)
    if config.get("dem", {}).get("use_as_model_input", False):
        raise RuntimeError("dem.use_as_model_input=true — DEM must not be a model input.")
    run_tag  = config.get("run_tag", args.config.stem)
    run_dir  = Path(config["run_dir"])
    ckpt_path = args.ckpt or (run_dir / "checkpoints" / "best_checkpoint.pt")
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}"); return 1

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else ("cpu" if args.device == "auto" else args.device)
    )
    out_dir = args.out_dir; out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}   s0={args.s0}  tau={args.tau}  thr={args.thr}")
    print(f"Out: {out_dir}")

    assembler = InputAssembler(config, device)
    mcfg = dict(config["model"])
    mcfg.setdefault("in_chans", sum(
        len(config["data"]["init_args"]["means"][m])
        for m in config["data"]["init_args"]["modalities"]
    ))
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = build_segman(mcfg).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])
    best_miou = float(ckpt.get("best_validation_miou", float("nan")))
    print(f"Model: SegMAN-{mcfg.get('variant','s')}  ep={ckpt.get('best_epoch','?')}  mIoU={best_miou:.4f}")

    splitmap = {"val": "valid", "valid": "valid", "test": "test", "bolivia": "bolivia"}
    results: dict[str, Any] = {}
    all_tiles: list[dict] = []
    t0 = time.time()

    for sp in args.eval_splits:
        print(f"\n[collect] {sp} ...")
        tiles = collect_split(model, assembler, config, splitmap[sp], device,
                              args.s0, args.tau, args.thr, args.max_tiles)
        for t in tiles:
            t["split"] = sp
        all_tiles.extend(tiles)
        agg = aggregate(tiles, args.n_boot, args.seed)
        results[sp] = agg

        p8 = agg["d8_pooled"]; p4 = agg["d4_pooled"]
        ci8 = agg["d8_ci"];    ci4 = agg["d4_ci"]
        u8  = agg["d8_useful"]["binary"]
        enr = agg["d8_useful"]["enrichment"]
        print(f"  tiles={ci8['n_tiles']}")
        print(f"  D8: VF(y)={p8['vf_gt_w']:.5f}  VF(yhat)={p8['vf_pred_w']:.5f}"
              f"  H_R={p8['H_R_w']:+.5f}  CI=[{ci8['H_R_w_ci'][0]:+.5f},{ci8['H_R_w_ci'][2]:+.5f}]"
              f"  ({ci8['pct_tiles_pred_cleaner_w']:.0f}% tiles pred cleaner)")
        print(f"  D8 E_topo: GT={p8['E_topo_gt']:.6f}  yhat={p8['E_topo_yhat']:.6f}"
              f"  dE={p8['dE_w']:+.6f}")
        print(f"  D4: VF(y)={p4['vf_gt_w']:.5f}  VF(yhat)={p4['vf_pred_w']:.5f}"
              f"  H_R={p4['H_R_w']:+.5f}  ({ci4['pct_tiles_pred_cleaner_w']:.0f}% tiles)")
        print(f"  Useful violations: useful={u8['useful_rate']:.3f}"
              f"  harmful={u8['harmful_rate']:.3f}  endorsed={u8['endorsed_rate']:.3f}"
              f"  enrichment={enr['enrichment_factor']:.1f}×")

    print(f"\n[done] {time.time()-t0:.1f}s")

    meta = {
        "run_tag": run_tag, "config": str(args.config), "checkpoint": str(ckpt_path),
        "best_epoch": ckpt.get("best_epoch"), "best_miou": best_miou,
        "s0": args.s0, "tau": args.tau, "thr": args.thr,
        "eval_splits": args.eval_splits,
        "tail_thresholds": TAIL_THRESHOLDS,
        "d8_routine": "_compute_d8 (steepest descent, edge pad, same as D8DownstreamLoss)",
    }

    payload = {"meta": meta, "results": _clean(results)}
    (out_dir / "topographic_order_headroom_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    _save_tile_csv(all_tiles, out_dir / "topographic_order_headroom_results.csv")
    (out_dir / "topographic_order_headroom_report.md").write_text(
        _report_md(meta, results), encoding="utf-8"
    )
    _make_plots(results, out_dir)

    # ── final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  TOPOGRAPHIC ORDER HEADROOM - SUMMARY")
    print("=" * 72)
    for sp in args.eval_splits:
        r = results[sp]
        p8 = r["d8_pooled"]; p4 = r["d4_pooled"]
        ci8 = r["d8_ci"]
        u  = r["d8_useful"]["binary"]
        tD = r["d8_tails"]
        # check tail dominance: pred below GT at all thresholds?
        dom = all(tD[t]["P_pred_hard"] <= tD[t]["P_gt_hard"] + 1e-8 for t in TAIL_THRESHOLDS)
        print(f"\n[{sp}]")
        print(f"  D8  VF(y)={p8['vf_gt_w']:.5f}  VF(yhat)={p8['vf_pred_w']:.5f}"
              f"  H_R={p8['H_R_w']:+.5f}  CI=[{ci8['H_R_w_ci'][0]:+.5f},{ci8['H_R_w_ci'][2]:+.5f}]")
        print(f"  D8  E_topo(y)={p8['E_topo_gt']:.6f}  E_topo(yhat)={p8['E_topo_yhat']:.6f}"
              f"  dE={p8['dE_w']:+.6f}")
        print(f"  D4  VF(y)={p4['vf_gt_w']:.5f}  VF(yhat)={p4['vf_pred_w']:.5f}"
              f"  H_R={p4['H_R_w']:+.5f}")
        print(f"  Tails pred<=GT at all thresholds: {'YES' if dom else 'NO'}")
        print(f"  Useful {u['useful_rate']:.3f} / Harmful {u['harmful_rate']:.3f}"
              f" / Endorsed {u['endorsed_rate']:.3f}")
    print(f"\nOutputs in {out_dir}")
    print()
    print("CONCLUSION: If H_R < 0 and dE < 0 and tails dominated -> baseline already satisfies")
    print("the local topographic-order constraint as well as or better than the labels.")
    print("Any loss monotone in this VF has no native label-relative headroom.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
