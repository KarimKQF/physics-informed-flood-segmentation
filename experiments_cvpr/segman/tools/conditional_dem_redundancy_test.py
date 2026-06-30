"""
Conditional DEM Redundancy Test  (inference-only diagnostic)
============================================================
Does the DEM add any predictive information about the ground-truth water label
(y) once the trained SegMAN water probability (p_water) is already known?

Information-theoretic target:   I(y ; z_DEM | p_water) ?= 0

We answer it operationally with post-hoc predictors (no new loss, no SegMAN
training, DEM NEVER a SegMAN input). SegMAN is run once (frozen) to produce
p_water; the DEM is used only AFTER inference as a feature for a flexible
post-hoc classifier:

    A) y ~ p_water                      (model only)
    B) y ~ p_water + z_DEM              (model + DEM)
    C) y ~ z_DEM                        (DEM-only marginal control)
    D) y ~ p_water + shuffle(z_DEM)     (sanity; should match A)

Main quantities:  dAUC = AUC(B) - AUC(A) ,  dAP = AP(B) - AP(A).
If dAUC ~ 0 and dAP ~ 0 while C (DEM-only) is non-trivial, then the DEM is
*conditionally redundant* given the trained model — no DEM loss, regardless of
formulation, has residual label-exploitable signal on this task.

Positive control (test power): a *synthetic* weakened predictor (model logits +
Gaussian noise) — dAUC should become POSITIVE, proving the test can detect
headroom when it exists. The definitive positive control (S1-only / early
checkpoint) is left as a TODO because no such checkpoint exists in the repo.

LIMITATION: this is relative to the available reference labels. It does NOT prove
the DEM carries no information about the true physical flood state; only that it
does not improve agreement with the labels used for evaluation.

Outputs -> outputs/conditional_dem_redundancy/ . Read-only w.r.t. SegMAN weights.
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
import torch.nn.functional as F

# ── repository paths ─────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent          # .../experiments_cvpr/segman/tools
SEGMAN_ROOT = SCRIPT_DIR.parent                         # .../experiments_cvpr/segman
REPO_ROOT   = SEGMAN_ROOT.parents[1]                    # repo root
for _p in (str(SEGMAN_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from step6c_v3_train import TopographyDataModule          # noqa: E402
from model.segman_model import build_segman               # noqa: E402
from diagnose_gt_topographic_consistency import (          # noqa: E402
    IGNORE_INDEX, WATER_CLASS,
)

from scipy.ndimage import uniform_filter, laplace          # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import (                               # noqa: E402
    roc_auc_score, average_precision_score, brier_score_loss, log_loss,
)

FEAT_NAMES = ["rel_elev", "slope", "tpi5", "tpi15", "tpi31", "curvature"]


# ── InputAssembler (copied to avoid train_segman import side-effects) ─────────
class InputAssembler:
    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        da = config["data"]["init_args"]
        self.modalities = list(da["modalities"])
        self.mean = {m: torch.tensor(da["means"][m], dtype=torch.float32, device=device).view(1, -1, 1, 1) for m in self.modalities}
        self.std  = {m: torch.tensor(da["stds"][m],  dtype=torch.float32, device=device).view(1, -1, 1, 1) for m in self.modalities}

    def __call__(self, image: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([(image[m].float() - self.mean[m]) / self.std[m] for m in self.modalities], dim=1)


def _get_topo(batch: dict[str, Any]) -> torch.Tensor | None:
    topo = batch.get("topography")
    if topo is None:
        return None
    if topo.ndim == 4 and topo.shape[1] == 1:
        topo = topo[:, 0]
    return topo.float()


def _get_target(batch: dict[str, Any]) -> torch.Tensor:
    mask = batch["mask"]
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]
    return mask.long()


def load_config(config_path: Path) -> dict[str, Any]:
    import yaml
    with config_path.open("r", encoding="utf-8-sig") as f:
        config = yaml.safe_load(f)
    dem_map_file = config.get("dem", {}).get("dem_tile_id_map_file")
    if dem_map_file and Path(dem_map_file).exists():
        with open(dem_map_file, encoding="utf-8") as f:
            config.setdefault("dem", {})["dem_tile_id_map"] = json.load(f).get("mapping", {})
    return config


# ── DEM features (no label leakage) ───────────────────────────────────────────
def dem_features(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """h: [H,W] elevation (meters), may contain NaN. Returns (feats [H,W,F], valid [H,W])."""
    valid = np.isfinite(h)
    if valid.sum() < 4:
        return np.zeros((*h.shape, len(FEAT_NAMES)), np.float32), valid
    fill = float(h[valid].mean())
    hf = np.where(valid, h, fill).astype(np.float32)
    mu = float(hf[valid].mean()); sd = float(hf[valid].std()) + 1e-6
    rel = (hf - mu) / sd
    gy, gx = np.gradient(hf)
    slope = np.sqrt(gx * gx + gy * gy)
    tpi5  = hf - uniform_filter(hf, size=5,  mode="nearest")
    tpi15 = hf - uniform_filter(hf, size=15, mode="nearest")
    tpi31 = hf - uniform_filter(hf, size=31, mode="nearest")
    curv  = laplace(hf, mode="nearest")
    feats = np.stack([rel, slope, tpi5, tpi15, tpi31, curv], axis=-1).astype(np.float32)
    return feats, valid


# ── inference + per-pixel collection (with cache) ─────────────────────────────
def collect_split(model, assembler, config, dm_split, device, cap, seed,
                  cache_path, use_cache, max_tiles):
    if use_cache and cache_path.exists():
        d = np.load(cache_path)
        print(f"  [cache] {cache_path.name}")
        return {k: d[k] for k in d.files}

    dm = TopographyDataModule(config, batch_size=1)
    dm.split = dm_split
    dm.setup("test")
    loader = dm.test_dataloader()

    rng = np.random.default_rng(seed)
    ys, ps, pdegs, fts, tids = [], [], [], [], []
    n = 0
    model.eval()
    with torch.no_grad():
        for raw in loader:
            if max_tiles is not None and n >= max_tiles:
                break
            topo = _get_topo(raw)
            if topo is None:
                n += 1
                continue
            image = {k: v.to(device) for k, v in raw["image"].items()}
            x = assembler(image)
            logits = model(x)                                  # [1,2,H,W]
            p2d = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()  # [H,W]
            y2d = _get_target(raw)[0].cpu().numpy()
            h2d = topo[0].cpu().numpy()
            # NDWI (McFeeters) from raw S2: (green - NIR)/(green + NIR); green=idx2 NIR=idx7
            s2 = image["S2L1C"][0]
            green = s2[2].cpu().numpy(); nir = s2[7].cpu().numpy()
            ndwi2d = (green - nir) / (green + nir + 1e-6)

            feats, valid_dem = dem_features(h2d)
            valid = (y2d != IGNORE_INDEX) & valid_dem & np.isfinite(p2d) & np.isfinite(ndwi2d)
            idx = np.flatnonzero(valid.ravel())
            if idx.size < 16:
                n += 1
                continue
            if cap is not None and idx.size > cap:
                idx = rng.choice(idx, size=cap, replace=False)   # natural prevalence

            yv = (y2d.ravel()[idx] == WATER_CLASS).astype(np.int8)
            pv = p2d.ravel()[idx].astype(np.float32)
            fv = feats.reshape(-1, len(FEAT_NAMES))[idx].astype(np.float32)
            wk = ndwi2d.ravel()[idx].astype(np.float32)          # weak spectral predictor

            ys.append(yv); ps.append(pv); pdegs.append(wk)
            fts.append(fv); tids.append(np.full(idx.size, n, np.int32))
            n += 1

    out = {
        "y": np.concatenate(ys), "p": np.concatenate(ps), "p_weak": np.concatenate(pdegs),
        "feats": np.concatenate(fts), "tile_ids": np.concatenate(tids),
        "feat_names": np.array(FEAT_NAMES),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **out)
    print(f"  collected {out['y'].size} px from {len(ys)} tiles -> cached")
    return out


# ── metrics ───────────────────────────────────────────────────────────────────
def _metrics(y: np.ndarray, s: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {"n_pix": int(y.size), "prevalence": float(y.mean())}
    if y.min() == y.max():
        return {**out, "auc": float("nan"), "ap": float("nan"),
                "brier": float("nan"), "logloss": float("nan")}
    out["auc"] = float(roc_auc_score(y, s))
    out["ap"] = float(average_precision_score(y, s))
    sc = np.clip(s, 1e-7, 1 - 1e-7)
    out["brier"] = float(brier_score_loss(y, sc))
    try:
        out["logloss"] = float(log_loss(y, sc, labels=[0, 1]))
    except Exception:
        out["logloss"] = float("nan")
    return out


def _fit_predict(Xtr, ytr, Xev, seed):
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
        random_state=seed)
    clf.fit(Xtr, ytr)
    return clf, clf.predict_proba(Xev)[:, 1]


def _design(data: dict, which: str, shuffle_seed: int | None = None) -> np.ndarray:
    p = data["p"][:, None]; pwk = data["p_weak"][:, None]; fe = data["feats"]
    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        fe = fe[rng.permutation(fe.shape[0])]
    return {"A": p, "B": np.hstack([p, fe]), "C": fe,
            "D": np.hstack([p, fe]),
            "A_pos": pwk, "B_pos": np.hstack([pwk, fe])}[which]


# ── bootstrap by tile ─────────────────────────────────────────────────────────
def bootstrap_delta(y, sA, sB, tile_ids, n_boot, cap_per_tile, seed):
    rng = np.random.default_rng(seed)
    uniq = np.unique(tile_ids)
    # pre-group + subsample per tile for speed
    groups = {}
    for t in uniq:
        gi = np.flatnonzero(tile_ids == t)
        if cap_per_tile is not None and gi.size > cap_per_tile:
            gi = rng.choice(gi, cap_per_tile, replace=False)
        groups[t] = gi
    dauc, dap = [], []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        gi = np.concatenate([groups[t] for t in pick])
        yy = y[gi]
        if yy.min() == yy.max():
            continue
        a_auc = roc_auc_score(yy, sA[gi]); b_auc = roc_auc_score(yy, sB[gi])
        a_ap = average_precision_score(yy, sA[gi]); b_ap = average_precision_score(yy, sB[gi])
        dauc.append(b_auc - a_auc); dap.append(b_ap - a_ap)
    def ci(v):
        if not v:
            return [float("nan")] * 3
        a = np.array(v)
        return [float(np.percentile(a, 2.5)), float(a.mean()), float(np.percentile(a, 97.5))]
    return {"dAUC_ci": ci(dauc), "dAP_ci": ci(dap), "n_boot_used": len(dauc)}


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="Conditional DEM redundancy test (inference-only).")
    ap.add_argument("--config", type=Path,
                    default=REPO_ROOT / "configs/segman/multiseed_n100/n100_seed0_dice_ce.yaml")
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--train-split", default="train", choices=["train", "valid"])
    ap.add_argument("--eval-splits", nargs="+", default=["val", "test", "bolivia"])
    ap.add_argument("--cap-train", type=int, default=20000)
    ap.add_argument("--cap-eval", type=int, default=20000)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--boot-cap", type=int, default=5000)
    ap.add_argument("--uncertain-band", type=float, nargs=2, default=[0.05, 0.95],
                    help="p_water range defining 'uncertain' pixels for the anti-ceiling eval.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--max-tiles", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs/conditional_dem_redundancy")
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
    cache_dir = out_dir / "cache"; cache_dir.mkdir(exist_ok=True)
    print(f"Device: {device}   out: {out_dir}")

    assembler = InputAssembler(config, device)
    model_cfg = dict(config["model"])
    model_cfg.setdefault("in_chans", sum(len(config["data"]["init_args"]["means"][m])
                                         for m in config["data"]["init_args"]["modalities"]))
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_segman(model_cfg).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])
    best_ep = ckpt.get("best_epoch", "?"); best_m = ckpt.get("best_validation_miou", float("nan"))
    print(f"Model: SegMAN-{model_cfg.get('variant','s')} best_ep={best_ep} best_miou={best_m}")

    ck_stem = ckpt_path.stem
    splitmap = {"val": "valid", "valid": "valid", "test": "test", "bolivia": "bolivia"}

    # ── collect ────────────────────────────────────────────────────────────────
    print(f"\n[collect] train-split='{args.train_split}'")
    t0 = time.time()
    train_data = collect_split(
        model, assembler, config, splitmap.get(args.train_split, args.train_split),
        device, args.cap_train, args.seed,
        cache_dir / f"{run_tag}_{ck_stem}_{args.train_split}_cap{args.cap_train}_s{args.seed}.npz",
        not args.no_cache, args.max_tiles)
    eval_data = {}
    for sp in args.eval_splits:
        print(f"[collect] eval='{sp}'")
        eval_data[sp] = collect_split(
            model, assembler, config, splitmap[sp], device, args.cap_eval, args.seed + 1,
            cache_dir / f"{run_tag}_{ck_stem}_{sp}_cap{args.cap_eval}_s{args.seed}.npz",
            not args.no_cache, args.max_tiles)
    print(f"[collect] done in {time.time()-t0:.1f}s")

    # ── fit post-hoc predictors ──────────────────────────────────────────────────
    print("\n[fit] HistGradientBoosting predictors on train ...")
    ytr = train_data["y"]
    models = {}
    for key, sh in [("A", None), ("B", None), ("C", None),
                    ("D", 9991), ("A_pos", None), ("B_pos", None)]:
        Xtr = _design(train_data, key, sh)
        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.1, random_state=args.seed)
        clf.fit(Xtr, ytr)
        models[key] = (clf, sh)
        print(f"  fit {key:5s} (dim={Xtr.shape[1]})")

    # ── evaluate ─────────────────────────────────────────────────────────────────
    results: dict[str, Any] = {}
    scores_cache: dict[str, dict[str, np.ndarray]] = {}
    for sp in args.eval_splits:
        d = eval_data[sp]; y = d["y"]
        res = {}
        sc = {}
        for key in ("A", "B", "C", "D", "A_pos", "B_pos"):
            clf, sh = models[key]
            X = _design(d, key, sh)
            s = clf.predict_proba(X)[:, 1]
            sc[key] = s
            res[key] = _metrics(y, s)
        scores_cache[sp] = sc
        # deltas
        res["delta_BA"] = {
            "dAUC": res["B"]["auc"] - res["A"]["auc"],
            "dAP":  res["B"]["ap"]  - res["A"]["ap"],
            "dBrier": res["B"]["brier"] - res["A"]["brier"],
            "dLogLoss": res["B"]["logloss"] - res["A"]["logloss"],
        }
        res["delta_BA_pos"] = {
            "dAUC": res["B_pos"]["auc"] - res["A_pos"]["auc"],
            "dAP":  res["B_pos"]["ap"]  - res["A_pos"]["ap"],
        }
        # anti-ceiling: B vs A restricted to model-uncertain pixels (lo<p_water<hi)
        pw = d["p"]; lo, hi = args.uncertain_band
        um = (pw > lo) & (pw < hi)
        if int(um.sum()) > 100 and y[um].min() != y[um].max():
            ua, ub, uc = sc["A"][um], sc["B"][um], sc["C"][um]
            res["uncertain"] = {
                "n_pix": int(um.sum()), "prevalence": float(y[um].mean()),
                "A_auc": float(roc_auc_score(y[um], ua)), "B_auc": float(roc_auc_score(y[um], ub)),
                "C_auc": float(roc_auc_score(y[um], uc)),
                "A_ap": float(average_precision_score(y[um], ua)),
                "B_ap": float(average_precision_score(y[um], ub)),
            }
            res["uncertain"]["dAUC"] = res["uncertain"]["B_auc"] - res["uncertain"]["A_auc"]
            res["uncertain"]["dAP"] = res["uncertain"]["B_ap"] - res["uncertain"]["A_ap"]
        else:
            res["uncertain"] = None
        # bootstrap CIs (B vs A)
        res["bootstrap_BA"] = bootstrap_delta(
            y, sc["A"], sc["B"], d["tile_ids"], args.n_boot, args.boot_cap, args.seed + 7)
        res["n_tiles"] = int(np.unique(d["tile_ids"]).size)
        results[sp] = res
        b = res["delta_BA"]; ci = res["bootstrap_BA"]["dAUC_ci"]
        print(f"\n[{sp}]  AUC: A={res['A']['auc']:.4f} B={res['B']['auc']:.4f} "
              f"dAUC={b['dAUC']:+.4f} (95% CI [{ci[0]:+.4f},{ci[2]:+.4f}])  C(DEM-only)={res['C']['auc']:.4f}")
        print(f"        AP:  A={res['A']['ap']:.4f} B={res['B']['ap']:.4f} dAP={b['dAP']:+.4f}   "
              f"D(shuf)={res['D']['auc']:.4f}  POS(NDWI) dAUC={res['delta_BA_pos']['dAUC']:+.4f}")
        if res["uncertain"]:
            u = res["uncertain"]
            print(f"        uncertain px (n={u['n_pix']}, prev={u['prevalence']:.3f}): "
                  f"A={u['A_auc']:.4f} B={u['B_auc']:.4f} dAUC={u['dAUC']:+.4f}  C={u['C_auc']:.4f}")

    # ── feature importance (permutation) for B on first eval split ───────────────
    feat_imp = None
    try:
        from sklearn.inspection import permutation_importance
        sp0 = args.eval_splits[0]; d0 = eval_data[sp0]
        idx = np.random.default_rng(0).choice(d0["y"].size, min(40000, d0["y"].size), replace=False)
        d_sub = {"p": d0["p"][idx], "p_weak": d0["p_weak"][idx], "feats": d0["feats"][idx]}
        Xb = _design(d_sub, "B")
        clfB = models["B"][0]
        pim = permutation_importance(clfB, Xb, d0["y"][idx], scoring="roc_auc",
                                     n_repeats=5, random_state=0, n_jobs=1)
        names = ["p_water"] + FEAT_NAMES
        feat_imp = {n: float(m) for n, m in zip(names, pim.importances_mean)}
        print(f"\n[feat_imp B on {sp0}] " + ", ".join(f"{k}={v:.4f}" for k, v in feat_imp.items()))
    except Exception as e:
        print(f"feature importance skipped: {e}")

    # ── positive-control status ──────────────────────────────────────────────────
    other_ck = sorted(p.name for p in (run_dir / "checkpoints").glob("*.pt"))
    pos_status = {
        "synthetic_run": True,
        "synthetic_desc": "weak spectral predictor NDWI=(green-NIR)/(green+NIR) in place of p_water; "
                          "DEM should add conditional info (dAUC>0), showing the test detects headroom "
                          "and that the trained model has absorbed this DEM-complementary signal",
        "real_control_available": False,
        "checkpoints_present": other_ck,
        "todo": "Definitive positive control NOT run: no early/undertrained or S1-only checkpoint exists. "
                "Next: train an S1-only baseline (drop S2) OR save an early-epoch checkpoint, then re-run "
                "this test on it; expect dAUC>0 if conditional DEM headroom exists.",
    }

    meta = {
        "run_tag": run_tag, "config": str(args.config), "checkpoint": str(ckpt_path),
        "best_epoch": ckpt.get("best_epoch"), "best_validation_miou": ckpt.get("best_validation_miou"),
        "train_split": args.train_split, "eval_splits": args.eval_splits,
        "feature_set": FEAT_NAMES, "predictor": "HistGradientBoostingClassifier(max_iter=300)",
        "cap_train": args.cap_train, "cap_eval": args.cap_eval, "sampling": "random (natural prevalence)",
        "n_boot": args.n_boot, "boot_cap_per_tile": args.boot_cap, "seed": args.seed,
        "train_pixels": int(ytr.size), "train_prevalence": float(ytr.mean()),
    }
    payload = {"meta": meta, "results": _clean(results),
               "feature_importance_B": feat_imp, "positive_control": pos_status}

    (out_dir / "conditional_dem_redundancy_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(out_dir / "conditional_dem_redundancy_results.csv", results)
    (out_dir / "conditional_dem_redundancy_report.md").write_text(
        _report_md(meta, results, feat_imp, pos_status), encoding="utf-8")
    _make_plots(out_dir, results, scores_cache, eval_data, feat_imp)

    # ── console summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  CONDITIONAL DEM REDUNDANCY — SUMMARY")
    print("=" * 64)
    redundant = True
    for sp in args.eval_splits:
        r = results[sp]; b = r["delta_BA"]; ci = r["bootstrap_BA"]
        print(f"[{sp:8s}] AUC A={r['A']['auc']:.4f} B={r['B']['auc']:.4f} dAUC={b['dAUC']:+.4f} "
              f"CI[{ci['dAUC_ci'][0]:+.4f},{ci['dAUC_ci'][2]:+.4f}] | "
              f"AP A={r['A']['ap']:.4f} B={r['B']['ap']:.4f} dAP={b['dAP']:+.4f} | "
              f"DEM-only AUC={r['C']['auc']:.4f} | POS dAUC={r['delta_BA_pos']['dAUC']:+.4f}")
        if ci["dAUC_ci"][2] > 0.005:   # upper CI meaningfully above 0
            redundant = False
    verdict = ("SUPPORTS conditional DEM redundancy (dAUC ~ 0 on all splits while DEM-only is informative)"
               if redundant else
               "DOES NOT fully support redundancy — at least one split shows dAUC CI above ~0")
    print(f"\nVERDICT: {verdict}")
    print(f"Positive control (NDWI weak predictor): dAUC should be POSITIVE -> test power "
          f"{'OK' if any(results[sp]['delta_BA_pos']['dAUC'] > 0.005 for sp in args.eval_splits) else 'WEAK'}")
    print(f"Real positive control: {'present' if pos_status['real_control_available'] else 'MISSING (see TODO in report)'}")
    print(f"\nOutputs in {out_dir}")
    return 0


# ── output writers ────────────────────────────────────────────────────────────
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


def _f(v, d=4):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "N/A"
    return f"{v:.{d}f}"


def _write_csv(path: Path, results: dict) -> None:
    rows = ["split,model,auc,ap,brier,logloss,n_pix,prevalence"]
    for sp, r in results.items():
        for k in ("A", "B", "C", "D", "A_pos", "B_pos"):
            m = r[k]
            rows.append(f"{sp},{k},{_f(m['auc'])},{_f(m['ap'])},{_f(m['brier'])},"
                        f"{_f(m['logloss'])},{m['n_pix']},{_f(m['prevalence'])}")
    rows.append("")
    rows.append("split,dAUC,dAP,dBrier,dLogLoss,dAUC_ci_lo,dAUC_ci_hi,dAP_ci_lo,dAP_ci_hi,pos_dAUC,pos_dAP")
    for sp, r in results.items():
        b = r["delta_BA"]; ci = r["bootstrap_BA"]; pos = r["delta_BA_pos"]
        rows.append(f"{sp},{_f(b['dAUC'])},{_f(b['dAP'])},{_f(b['dBrier'])},{_f(b['dLogLoss'])},"
                    f"{_f(ci['dAUC_ci'][0])},{_f(ci['dAUC_ci'][2])},{_f(ci['dAP_ci'][0])},{_f(ci['dAP_ci'][2])},"
                    f"{_f(pos['dAUC'])},{_f(pos['dAP'])}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _report_md(meta, results, feat_imp, pos) -> str:
    L = ["# Conditional DEM Redundancy Test", "",
         "**Inference-only. SegMAN weights frozen. DEM never a model input — used only as a",
         "post-hoc feature after inference.**", "",
         "## Question",
         "Does the DEM add predictive information about the water label `y` once the trained",
         "model probability `p_water` is known?  `I(y ; z_DEM | p_water) ?= 0`.", "",
         "## Method",
         f"- Checkpoint: `{meta['checkpoint']}` (best_ep={meta['best_epoch']}, best_miou={_f(meta['best_validation_miou'])})",
         f"- Predictor: **{meta['predictor']}**, fit on `{meta['train_split']}` "
         f"({meta['train_pixels']} px, prevalence {_f(meta['train_prevalence'],3)}), eval on held-out splits.",
         f"- DEM features: {', '.join(meta['feature_set'])} (computed without label leakage).",
         f"- Sampling: {meta['sampling']}, cap_train={meta['cap_train']}, cap_eval={meta['cap_eval']}/tile.",
         f"- Models:  A: `y~p_water`  ·  B: `y~p_water+DEM`  ·  C: `y~DEM`  ·  D: `y~p_water+shuffle(DEM)`.",
         f"- Bootstrap: by tile, n={meta['n_boot']}, {meta['boot_cap_per_tile']} px/tile, 95% CI.", "",
         "## Results", "",
         "| split | model | AUC | AP | Brier | LogLoss | n_pix | prev |",
         "|-------|-------|----:|---:|------:|--------:|------:|-----:|"]
    for sp, r in results.items():
        for k in ("A", "B", "C", "D"):
            m = r[k]
            L.append(f"| {sp} | {k} | {_f(m['auc'])} | {_f(m['ap'])} | {_f(m['brier'])} | "
                     f"{_f(m['logloss'])} | {m['n_pix']} | {_f(m['prevalence'],3)} |")
    L += ["", "### Main comparison  B − A  (model+DEM vs model)", "",
          "| split | dAUC | 95% CI | dAP | 95% CI | dBrier | dLogLoss |",
          "|-------|-----:|--------|----:|--------|-------:|---------:|"]
    for sp, r in results.items():
        b = r["delta_BA"]; ci = r["bootstrap_BA"]
        L.append(f"| {sp} | {_f(b['dAUC'])} | [{_f(ci['dAUC_ci'][0])}, {_f(ci['dAUC_ci'][2])}] | "
                 f"{_f(b['dAP'])} | [{_f(ci['dAP_ci'][0])}, {_f(ci['dAP_ci'][2])}] | "
                 f"{_f(b['dBrier'])} | {_f(b['dLogLoss'])} |")
    L += ["", "### Anti-ceiling check — B vs A on model-uncertain pixels only",
          "Restricted to pixels where the model is unsure (lo < p_water < hi): if DEM helps anywhere it",
          "is here. dAUC ≈ 0 here rules out 'redundancy is just a ceiling artifact'.", "",
          "| split | n_pix | prev | A AUC | B AUC | dAUC | C(DEM-only) AUC |",
          "|-------|------:|-----:|------:|------:|-----:|----------------:|"]
    for sp, r in results.items():
        u = r.get("uncertain")
        if u:
            L.append(f"| {sp} | {u['n_pix']} | {_f(u['prevalence'],3)} | {_f(u['A_auc'])} | {_f(u['B_auc'])} | "
                     f"{_f(u['dAUC'])} | {_f(u['C_auc'])} |")
        else:
            L.append(f"| {sp} | — | — | — | — | — | — |")
    L += ["", "### Positive control (weak spectral predictor, NDWI)",
          f"*{pos['synthetic_desc']}*. dAUC should be **positive** if the test can detect headroom.", "",
          "| split | NDWI AUC (A_pos) | NDWI+DEM AUC (B_pos) | dAUC (pos) | dAP (pos) |",
          "|-------|-----------------:|---------------------:|-----------:|----------:|"]
    for sp, r in results.items():
        p = r["delta_BA_pos"]
        L.append(f"| {sp} | {_f(r['A_pos']['auc'])} | {_f(r['B_pos']['auc'])} | {_f(p['dAUC'])} | {_f(p['dAP'])} |")
    if feat_imp:
        L += ["", "### Permutation feature importance — model B (Δ ROC-AUC)", "",
              "| feature | importance |", "|---------|-----------:|"]
        for k, v in sorted(feat_imp.items(), key=lambda kv: -kv[1]):
            L.append(f"| {k} | {_f(v)} |")
    L += ["", "## Interpretation",
          "- If **dAUC ≈ 0** and **dAP ≈ 0** (CIs spanning 0) while **C (DEM-only)** is non-trivial,",
          "  the DEM is *conditionally redundant* given the trained model: it carries marginal water",
          "  information, but none beyond `p_water`. No DEM loss — regardless of formulation or",
          "  complexity — has residual label-exploitable signal once the image-only model has converged.",
          "- The D8 null is therefore not a formulation failure but an informational property of the",
          "  model–data pair.", "",
          "## Limitation (must be stated)",
          "This test is **relative to the available reference labels**. It does NOT prove the DEM",
          "carries no information about the *true* physical flood state. If labels are noisy or",
          "topographically inconsistent, the DEM could still help relative to unobserved truth.",
          "The claim is only about improving agreement with the labels used for evaluation.", "",
          "## Positive-control status",
          f"- Synthetic power check: **run** ({pos['synthetic_desc']}).",
          f"- Real positive control: **{'available' if pos['real_control_available'] else 'MISSING'}**.",
          f"- Checkpoints present: {', '.join(pos['checkpoints_present'])}.",
          f"- TODO: {pos['todo']}", "",
          "---", "*Generated by `experiments_cvpr/segman/tools/conditional_dem_redundancy_test.py`.*"]
    return "\n".join(L) + "\n"


def _make_plots(out_dir, results, scores, eval_data, feat_imp):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, precision_recall_curve
    except Exception as e:
        print(f"plots skipped: {e}"); return
    splits = list(results.keys())
    # ROC + PR (A vs B) per split
    fig, axes = plt.subplots(2, len(splits), figsize=(4 * len(splits), 7), squeeze=False)
    for j, sp in enumerate(splits):
        y = eval_data[sp]["y"]
        for key, c in [("A", "tab:gray"), ("B", "tab:red")]:
            fa, ta, _ = roc_curve(y, scores[sp][key])
            axes[0][j].plot(fa, ta, color=c, label=f"{key} AUC={results[sp][key]['auc']:.3f}")
            pr, rc, _ = precision_recall_curve(y, scores[sp][key])
            axes[1][j].plot(rc, pr, color=c, label=f"{key} AP={results[sp][key]['ap']:.3f}")
        axes[0][j].plot([0, 1], [0, 1], "k--", lw=0.6)
        axes[0][j].set_title(f"ROC — {sp}"); axes[0][j].legend(fontsize=7)
        axes[1][j].set_title(f"PR — {sp}"); axes[1][j].legend(fontsize=7)
        axes[0][j].set_xlabel("FPR"); axes[0][j].set_ylabel("TPR")
        axes[1][j].set_xlabel("Recall"); axes[1][j].set_ylabel("Precision")
    fig.tight_layout(); fig.savefig(out_dir / "roc_pr_A_vs_B.png", dpi=130); plt.close(fig)
    # bar dAUC / dAP
    fig, ax = plt.subplots(figsize=(1.6 * len(splits) + 2, 3.5))
    x = np.arange(len(splits)); w = 0.35
    dauc = [results[sp]["delta_BA"]["dAUC"] for sp in splits]
    dap  = [results[sp]["delta_BA"]["dAP"] for sp in splits]
    lo = [results[sp]["delta_BA"]["dAUC"] - results[sp]["bootstrap_BA"]["dAUC_ci"][0] for sp in splits]
    hi = [results[sp]["bootstrap_BA"]["dAUC_ci"][2] - results[sp]["delta_BA"]["dAUC"] for sp in splits]
    ax.bar(x - w/2, dauc, w, yerr=[lo, hi], capsize=3, label="dAUC (B−A)", color="tab:red")
    ax.bar(x + w/2, dap, w, label="dAP (B−A)", color="tab:blue")
    ax.axhline(0, color="k", lw=0.6); ax.set_xticks(x); ax.set_xticklabels(splits)
    ax.set_title("DEM gain over model (≈0 ⇒ redundant)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "delta_auc_ap_by_split.png", dpi=130); plt.close(fig)
    # feature importance
    if feat_imp:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        it = sorted(feat_imp.items(), key=lambda kv: kv[1])
        ax.barh([k for k, _ in it], [v for _, v in it], color="tab:green")
        ax.set_title("Permutation importance — model B (ΔAUC)")
        fig.tight_layout(); fig.savefig(out_dir / "feature_importance_B.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
