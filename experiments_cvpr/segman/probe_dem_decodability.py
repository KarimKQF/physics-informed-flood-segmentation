"""
DEM Decodability Probe — Representation-level Diagnostic
=======================================================
Read-only. Does NOT train, modify, or re-run the segmentation model.

Question
--------
How much *relative elevation* (within-tile topography) is linearly decodable
from a frozen, already-trained SegMAN model's internal features?

We fit a LINEAR probe (closed-form ridge regression = optimal linear decoder)
that maps the encoder feature pyramid -> per-pixel demeaned DEM, then measure on
held-out splits how far the decoded elevation is from the true DEM.

Why "relative / demeaned" elevation
-----------------------------------
Absolute elevation is not identifiable from a local 512x512 patch (a model
cannot know if a scene sits at 10 m or 2000 m). The hydrology that any DEM loss
cares about is *relative* height (water sits low, flow goes downhill). So both
features and target are demeaned per tile: the probe is a strict WITHIN-tile
linear decoder of topographic deviation.

Controls (the whole point — without these the number is uninterpretable)
------------------------------------------------------------------------
  trained  : features from the trained checkpoint
  random   : features from a random-init backbone (same architecture)
  input    : the raw normalized 15-ch input itself (avg-pooled)

Interpretation:
  trained >> input AND trained >> random
      -> segmentation training INDUCED an elevation representation.
  trained ~ input
      -> the model merely passes through input<->elevation correlation;
         it did not build a special elevation representation.

Metrics (per split; pooled over all valid pixels + per-tile distribution)
-------------------------------------------------------------------------
  RMSE [m] , MAE [m]   : how far decoded relative elevation is from true (meters)
  R^2                  : fraction of within-tile elevation variance explained
  Pearson r            : linear correlation decoded vs true relative elevation
  signal std [m]       : per-tile elevation std (context for the RMSE)

Outputs go to reports/ and docs/ only. Never writes to run dirs.

Usage (smoke test):
    python experiments_cvpr/segman/probe_dem_decodability.py \\
        --config configs/segman/multiseed_n100/n100_seed0_dice_ce.yaml \\
        --max-fit-tiles 8 --max-eval-tiles 5 --splits val

Full run:
    python experiments_cvpr/segman/probe_dem_decodability.py \\
        --config configs/segman/multiseed_n100/n100_seed0_dice_ce.yaml \\
        --splits val test bolivia
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

# ── repository paths (mirrors diagnose_elevation_auc_predictions.py) ──────────
SEGMAN_ROOT = Path(__file__).resolve().parent
REPO_ROOT   = SEGMAN_ROOT.parents[1]
for _p in (str(SEGMAN_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from step6c_v3_train import TopographyDataModule          # noqa: E402
from model.segman_model import build_segman               # noqa: E402


# ── InputAssembler (copied from train_segman.py to avoid import side-effects) ──
class InputAssembler:
    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        data_args = config["data"]["init_args"]
        self.modalities = list(data_args["modalities"])
        means = data_args["means"]
        stds  = data_args["stds"]
        self.mean = {m: torch.tensor(means[m], dtype=torch.float32, device=device).view(1, -1, 1, 1) for m in self.modalities}
        self.std  = {m: torch.tensor(stds[m],  dtype=torch.float32, device=device).view(1, -1, 1, 1) for m in self.modalities}

    def __call__(self, image: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = []
        for m in self.modalities:
            x = image[m].float()
            parts.append((x - self.mean[m]) / self.std[m])
        return torch.cat(parts, dim=1)


def _get_topo(batch: dict[str, Any]) -> torch.Tensor | None:
    topo = batch.get("topography")
    if topo is None:
        return None
    if topo.ndim == 4 and topo.shape[1] == 1:
        topo = topo[:, 0]
    return topo.float()


def load_config(config_path: Path) -> dict[str, Any]:
    import yaml
    with config_path.open("r", encoding="utf-8-sig") as f:
        config = yaml.safe_load(f)
    dem_map_file = config.get("dem", {}).get("dem_tile_id_map_file")
    if dem_map_file and Path(dem_map_file).exists():
        with open(dem_map_file, encoding="utf-8") as f:
            config.setdefault("dem", {})["dem_tile_id_map"] = json.load(f).get("mapping", {})
    return config


# ── Feature extractors (all return [B, C, oh, ow]) ────────────────────────────

@torch.no_grad()
def feats_pyramid(model: torch.nn.Module, x: torch.Tensor, out_hw: int) -> torch.Tensor:
    """Encoder 4-stage pyramid, each upsampled to out_hw and concatenated."""
    feats = model.forward_features(x)                      # list of [B, Ci, hi, wi]
    ups = [F.interpolate(f, size=(out_hw, out_hw), mode="bilinear", align_corners=False) for f in feats]
    return torch.cat(ups, dim=1)                           # [B, sum Ci, oh, ow]


@torch.no_grad()
def feats_input(x: torch.Tensor, out_hw: int) -> torch.Tensor:
    """Raw normalized input, average-pooled to out_hw."""
    return F.adaptive_avg_pool2d(x, (out_hw, out_hw))


# ── per-tile standardization helpers (z-score) ────────────────────────────────
def _standardize_rows(X: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Z-score each row (channel) over pixels. X: [D, N]."""
    mu = X.mean(dim=1, keepdim=True)
    sd = X.std(dim=1, keepdim=True).clamp_min(eps)
    return (X - mu) / sd


def _standardize_vec(t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Z-score a vector. t: [N]."""
    return (t - t.mean()) / t.std().clamp_min(eps)


# ── Streaming normal-equations accumulator for a within-tile linear probe ──────
class RidgeAccumulator:
    """Accumulate A = sum X_c X_c^T and b = sum X_c t_c over tiles (demeaned)."""

    def __init__(self, dim: int, device: torch.device) -> None:
        self.dim = dim
        self.A = torch.zeros((dim, dim), dtype=torch.float64, device=device)
        self.b = torch.zeros((dim,),     dtype=torch.float64, device=device)
        self.n_pix = 0
        self.n_tiles = 0

    def add_tile(self, X: torch.Tensor, t: torch.Tensor) -> None:
        """X: [D, N] features; t: [N] target (meters).
        Both standardized per tile (z-score) so the probe measures within-tile
        *shape* correlation, free of cross-tile relief-scale heterogeneity."""
        Xc = _standardize_rows(X).double()                 # [D, N] per-channel z
        tc = _standardize_vec(t).double()                  # [N] z-scored elevation
        self.A += Xc @ Xc.t()
        self.b += Xc @ tc
        self.n_pix += X.shape[1]
        self.n_tiles += 1

    def solve(self, ridge: float) -> torch.Tensor:
        diag = torch.diagonal(self.A).mean().clamp_min(1e-12)
        reg = ridge * diag
        A = self.A + reg * torch.eye(self.dim, dtype=torch.float64, device=self.A.device)
        W = torch.linalg.solve(A, self.b)                  # [D]
        return W


# ── Streaming eval metric accumulator ─────────────────────────────────────────
class EvalAccumulator:
    def __init__(self) -> None:
        self.ss_res = 0.0   # sum (pred - t)^2
        self.ss_tot = 0.0   # sum t^2  (t already demeaned per tile)
        self.abs_sum = 0.0  # sum |pred - t|
        self.sp = 0.0       # sum pred
        self.st = 0.0       # sum t
        self.spp = 0.0      # sum pred^2
        self.stt = 0.0      # sum t^2
        self.spt = 0.0      # sum pred*t
        self.n = 0
        self.per_tile: list[dict[str, float]] = []

    def add_tile(self, pred: np.ndarray, z: np.ndarray, std_m: float) -> None:
        """pred: probe output; z: z-scored true relative elevation (unit var);
        std_m: true per-tile elevation std (meters) for converting to meters."""
        r = pred - z
        self.ss_res += float(np.sum(r * r))
        self.ss_tot += float(np.sum(z * z))     # = n (z unit var) but keep exact
        self.abs_sum += float(np.sum(np.abs(r)))
        self.sp += float(pred.sum()); self.st += float(z.sum())
        self.spp += float(np.sum(pred * pred)); self.stt += float(np.sum(z * z))
        self.spt += float(np.sum(pred * z)); self.n += len(z)
        # per-tile (z-score space)
        n = len(z)
        nrmse = math.sqrt(np.mean(r * r)) if n else float("nan")   # tile-std units
        ss_z = float(np.sum(z * z))
        r2 = 1.0 - float(np.sum(r * r)) / ss_z if ss_z > 0 else float("nan")
        std_p, std_z = float(pred.std()), float(z.std())
        pear = float(np.mean((pred - pred.mean()) * (z - z.mean())) / (std_p * std_z)) if std_p > 1e-12 and std_z > 1e-12 else float("nan")
        self.per_tile.append({"nrmse": nrmse, "rmse_m": nrmse * std_m, "r2": r2,
                              "pearson": pear, "signal_std_m": std_m, "n": n})

    def pooled(self) -> dict[str, float]:
        if self.n == 0:
            return {"n_pix": 0}
        nrmse = math.sqrt(self.ss_res / self.n)            # tile-std units
        nmae = self.abs_sum / self.n
        r2 = 1.0 - self.ss_res / self.ss_tot if self.ss_tot > 0 else float("nan")
        # Pearson over all pixels (pred & z standardized per tile)
        cov = self.spt / self.n - (self.sp / self.n) * (self.st / self.n)
        vp = self.spp / self.n - (self.sp / self.n) ** 2
        vt = self.stt / self.n - (self.st / self.n) ** 2
        pear = cov / math.sqrt(vp * vt) if vp > 1e-12 and vt > 1e-12 else float("nan")
        return {"nrmse": nrmse, "nmae": nmae, "r2": r2, "pearson": pear, "n_pix": self.n, "n_tiles": len(self.per_tile)}

    def per_tile_summary(self) -> dict[str, float]:
        def med(key: str) -> float:
            v = [d[key] for d in self.per_tile if math.isfinite(d[key])]
            return float(np.median(v)) if v else float("nan")
        return {f"median_{k}": med(k) for k in ("nrmse", "rmse_m", "r2", "pearson", "signal_std_m")}


# ── Per-tile feature/target extraction ────────────────────────────────────────

def tile_valid_arrays(
    feat_map: torch.Tensor,   # [1, D, oh, ow]
    topo_pool: torch.Tensor,  # [1, 1, oh, ow] (may contain NaN = invalid)
    min_std_m: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    D = feat_map.shape[1]
    oh = feat_map.shape[2]
    t = topo_pool.view(-1)                                  # [oh*ow]
    valid = torch.isfinite(t)
    if valid.sum().item() < 64:
        return None
    t_v = t[valid]                                          # [N]
    if float(t_v.std().item()) < min_std_m:
        return None
    X = feat_map.view(D, oh * oh)[:, valid]                 # [D, N]
    return X, t_v


# ── Main driver ───────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="DEM decodability linear probe (read-only).")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="Checkpoint. Default {run_dir}/checkpoints/best_checkpoint.pt")
    ap.add_argument("--fit-split", default="train", choices=["train", "valid"],
                    help="Split used to FIT the probe (default: train).")
    ap.add_argument("--splits", nargs="+", default=["val", "test", "bolivia"],
                    help="Splits to evaluate on (held-out from probe fit).")
    ap.add_argument("--out-hw", type=int, default=128, help="Probe working resolution (default 128 = stride 4).")
    ap.add_argument("--ridge", type=float, default=1e-2, help="Ridge strength (fraction of mean diag).")
    ap.add_argument("--min-std-m", type=float, default=1.0, help="Skip tiles with elevation std below this (flat).")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--no-random-control", action="store_true")
    ap.add_argument("--no-input-control", action="store_true")
    ap.add_argument("--max-fit-tiles", type=int, default=None)
    ap.add_argument("--max-eval-tiles", type=int, default=None)
    ap.add_argument("--save-maps", type=int, default=0,
                    help="Render generated-vs-true DEM maps for the first N tiles of the first split.")
    ap.add_argument("--out-suffix", default=None)
    args = ap.parse_args()

    config = load_config(args.config)
    run_tag = config.get("run_tag", args.config.stem)
    suffix = args.out_suffix or run_tag

    if config.get("dem", {}).get("use_as_model_input", False):
        raise RuntimeError("Config has dem.use_as_model_input=true — DEM must not be a model input.")

    run_dir = Path(config["run_dir"])
    ckpt_path = args.ckpt or (run_dir / "checkpoints" / "best_checkpoint.pt")
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        return 1

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                          ("cpu" if args.device == "auto" else args.device))
    print(f"Device: {device}")

    reports_dir = REPO_ROOT / "reports"; reports_dir.mkdir(parents=True, exist_ok=True)
    docs_dir    = REPO_ROOT / "docs";    docs_dir.mkdir(parents=True, exist_ok=True)

    assembler = InputAssembler(config, device)
    model_cfg = dict(config["model"])
    model_cfg.setdefault("in_chans", sum(
        len(config["data"]["init_args"]["means"][m]) for m in config["data"]["init_args"]["modalities"]))

    # trained model
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_segman(model_cfg).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])
    best_ep = ckpt.get("best_epoch", "?"); best_m = ckpt.get("best_validation_miou", float("nan"))
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: SegMAN-{model_cfg.get('variant','s')} ({n_params:.2f}M) best_ep={best_ep} best_miou={best_m}")

    # random-init control model (same architecture, fresh weights)
    rand_model = None
    if not args.no_random_control:
        torch.manual_seed(12345)
        rand_model = build_segman(model_cfg).to(device).eval()
        print("Random-init control backbone built.")

    out_hw = args.out_hw

    # condition keys
    conditions = ["trained"]
    if rand_model is not None:
        conditions.append("random")
    if not args.no_input_control:
        conditions.append("input")

    # feature dims
    enc_dim = sum(VARIANT_DIMS(model_cfg))
    in_dim = model_cfg["in_chans"]
    dims = {"trained": enc_dim, "random": enc_dim, "input": in_dim}

    def extract(cond: str, x: torch.Tensor) -> torch.Tensor:
        if cond == "trained":
            return feats_pyramid(model, x, out_hw)
        if cond == "random":
            return feats_pyramid(rand_model, x, out_hw)
        return feats_input(x, out_hw)

    # ── pass over a split, calling cb(cond, X, t) per (condition, tile) ─────────
    def iterate_split(dm_split: str, max_tiles: int | None):
        dm = TopographyDataModule(config, batch_size=1)
        dm.split = dm_split
        dm.setup("test")
        loader = dm.test_dataloader()
        n = 0
        with torch.no_grad():
            for raw in loader:
                if max_tiles is not None and n >= max_tiles:
                    break
                topo = _get_topo(raw)
                if topo is None:
                    n += 1
                    continue
                image = {k: v.to(device) for k, v in raw["image"].items()}
                topo = topo.to(device)
                x = assembler(image)                                  # [1,15,H,W]
                topo_pool = F.adaptive_avg_pool2d(topo.unsqueeze(1), (out_hw, out_hw))  # [1,1,oh,ow]
                per_cond = {}
                for cond in conditions:
                    fmap = extract(cond, x)                           # [1,D,oh,ow]
                    arrs = tile_valid_arrays(fmap, topo_pool, args.min_std_m)
                    per_cond[cond] = arrs
                yield per_cond, topo_pool
                n += 1

    # ── FIT ────────────────────────────────────────────────────────────────────
    fit_split = args.fit_split  # "train" or "valid"
    print(f"\n=== FIT probe on split='{fit_split}' (ridge={args.ridge}, out_hw={out_hw}) ===")
    t0 = time.time()
    accs = {c: RidgeAccumulator(dims[c], device) for c in conditions}
    nfit = 0
    for per_cond, _topo in iterate_split(fit_split, args.max_fit_tiles):
        for c in conditions:
            arrs = per_cond[c]
            if arrs is not None:
                accs[c].add_tile(arrs[0], arrs[1])
        nfit += 1
        if nfit % 20 == 0:
            print(f"  fit tiles: {nfit}")
    W = {}
    for c in conditions:
        W[c] = accs[c].solve(args.ridge)
        print(f"  [{c}] fit on {accs[c].n_tiles} tiles, {accs[c].n_pix} px, dim={dims[c]}")
    print(f"  fit elapsed {time.time()-t0:.1f}s")

    # ── EVAL ───────────────────────────────────────────────────────────────────
    split_map = {"val": "valid", "valid": "valid", "test": "test", "bolivia": "bolivia"}
    results: dict[str, Any] = {}
    viz_tiles: list[dict[str, Any]] = []
    viz_conds = [c for c in ("trained", "input", "random") if c in conditions]
    for si, split in enumerate(args.splits):
        dm_split = split_map[split]
        print(f"\n=== EVAL split='{split}' ===")
        t0 = time.time()
        evals = {c: EvalAccumulator() for c in conditions}
        nev = 0
        for per_cond, topo_pool in iterate_split(dm_split, args.max_eval_tiles):
            for c in conditions:
                arrs = per_cond[c]
                if arrs is None:
                    continue
                X, t = arrs
                Xc = _standardize_rows(X).double()
                pred = (W[c] @ Xc).cpu().numpy()                     # [N]
                std_m = float(t.std().item())
                z = _standardize_vec(t).cpu().numpy()
                evals[c].add_tile(pred, z, std_m)
            # capture generated-vs-true DEM maps for the first split
            if si == 0 and len(viz_tiles) < args.save_maps and per_cond.get("trained") is not None:
                oh = topo_pool.shape[-1]
                t_flat = topo_pool.view(-1)
                valid = torch.isfinite(t_flat)
                truth = torch.full((oh * oh,), float("nan"))
                truth[valid] = _standardize_vec(t_flat[valid]).cpu()
                grids = {"truth": truth.view(oh, oh).numpy(), "split": split}
                for c in viz_conds:
                    arrs = per_cond[c]
                    if arrs is None:
                        continue
                    X, t = arrs
                    Xc = _standardize_rows(X).double()
                    predc = (W[c] @ Xc).float().cpu()
                    g = torch.full((oh * oh,), float("nan")); g[valid] = predc
                    grids[c] = g.view(oh, oh).numpy()
                    pe = evals[c].per_tile[-1]
                    grids[f"{c}_r"] = pe.get("pearson", float("nan"))
                viz_tiles.append(grids)
            nev += 1
        results[split] = {}
        for c in conditions:
            pooled = evals[c].pooled()
            pertile = evals[c].per_tile_summary()
            results[split][c] = {"pooled": pooled, "per_tile": pertile}
        print(f"  eval tiles: {nev}  elapsed {time.time()-t0:.1f}s")
        # console table (pooled R2/Pearson in z-space; per-tile RMSE in meters)
        print(f"  {'cond':8s} | {'R2':>7s} {'r':>7s} {'nRMSE':>7s} | {'med_r':>6s} {'medRMSE[m]':>10s} {'sig.std[m]':>10s}")
        for c in conditions:
            p = results[split][c]["pooled"]; pt = results[split][c]["per_tile"]
            print(f"  {c:8s} | {p.get('r2', float('nan')):7.3f} {p.get('pearson', float('nan')):7.3f} "
                  f"{p.get('nrmse', float('nan')):7.3f} | {pt.get('median_pearson', float('nan')):6.3f} "
                  f"{pt.get('median_rmse_m', float('nan')):10.2f} {pt.get('median_signal_std_m', float('nan')):10.2f}")

    # ── Save JSON + Markdown ───────────────────────────────────────────────────
    out = {
        "run_tag": run_tag, "config_path": str(args.config), "ckpt_path": str(ckpt_path),
        "fit_split": fit_split, "out_hw": out_hw, "ridge": args.ridge, "min_std_m": args.min_std_m,
        "feature_dims": dims, "conditions": conditions,
        "best_epoch": ckpt.get("best_epoch"), "best_validation_miou": ckpt.get("best_validation_miou"),
        "results": results,
    }
    json_path = reports_dir / f"dem_decodability_{suffix}.json"
    json_path.write_text(json.dumps(_clean(out), indent=2), encoding="utf-8")
    print(f"\nJSON: {json_path}")

    md_path = docs_dir / f"dem_decodability_{suffix}.md"
    md_path.write_text(_markdown(out), encoding="utf-8")
    print(f"MD:   {md_path}")

    if args.save_maps and viz_tiles:
        fig_path = docs_dir / f"dem_decodability_{suffix}_maps.png"
        ok = render_maps(viz_tiles, viz_conds, fig_path, run_tag)
        print(f"FIG:  {fig_path}" if ok else "FIG:  (matplotlib unavailable; saved .npz instead)")
        if not ok:
            np.savez(reports_dir / f"dem_decodability_{suffix}_maps.npz",
                     **{f"tile{i}_{k}": v for i, g in enumerate(viz_tiles)
                        for k, v in g.items() if isinstance(v, np.ndarray)})
    return 0


def render_maps(viz_tiles: list[dict[str, Any]], viz_conds: list[str], path: Path, run_tag: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    cols = ["truth"] + viz_conds
    nrow, ncol = len(viz_tiles), len(cols)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.6 * nrow), squeeze=False)
    titles = {"truth": "TRUE rel. DEM", "trained": "generated (trained)",
              "input": "generated (input)", "random": "generated (random)"}
    for i, g in enumerate(viz_tiles):
        for j, c in enumerate(cols):
            ax = axes[i][j]
            arr = g.get(c)
            if arr is None:
                ax.axis("off"); continue
            ax.imshow(arr, cmap="terrain", vmin=-2.5, vmax=2.5, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(titles.get(c, c), fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{g.get('split','')} tile {i}", fontsize=8)
            elif c in viz_conds:
                ax.set_xlabel(f"r={g.get(f'{c}_r', float('nan')):.2f}", fontsize=8)
    fig.suptitle(f"DEM decodability — {run_tag}\n(per-tile z-scored relative elevation; cmap shared)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def VARIANT_DIMS(model_cfg: dict[str, Any]) -> list[int]:
    from model.segman_model import VARIANTS
    return VARIANTS[str(model_cfg.get("variant", "s"))]["embed_dims"]


def _clean(o: Any) -> Any:
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def _f(v: Any, d: int = 3) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "N/A"
    return f"{v:.{d}f}"


def _markdown(out: dict[str, Any]) -> str:
    L: list[str] = []
    L += [
        f"# DEM Decodability Probe — {out['run_tag']}",
        "",
        "**Read-only. Frozen backbone. No training/loss/config changes. DEM never a model input.**",
        "",
        f"- Config: `{out['config_path']}`",
        f"- Checkpoint: `{out['ckpt_path']}` (best_ep={out.get('best_epoch')}, best_miou={_f(out.get('best_validation_miou'),4)})",
        f"- Probe: within-tile **linear** (closed-form ridge), fit on `{out['fit_split']}`, resolution {out['out_hw']}x{out['out_hw']}, ridge={out['ridge']}",
        f"- Feature dims: " + ", ".join(f"{k}={v}" for k, v in out["feature_dims"].items()),
        "",
        "Target = per-tile **z-scored** DEM (relative topography). Both features and target",
        "are standardized per tile, so this is a strict within-tile *shape* decoder, free of",
        "cross-tile relief-scale confounds. `R^2`/`Pearson` are in z-space (scale-free);",
        "`RMSE [m]` = nRMSE x true per-tile elevation std (shape error expressed in meters).",
        "",
        "## Conditions",
        "- `trained` : features from the trained checkpoint",
        "- `random`  : features from a random-init backbone (same architecture)",
        "- `input`   : the raw normalized 15-ch input itself (avg-pooled)",
        "",
        "**Read it like this:** `trained` >> `input` and `trained` >> `random` means training",
        "induced an elevation representation. `trained` ~ `input` means the model only passes",
        "through input<->elevation correlation.",
        "",
    ]
    for split, byc in out["results"].items():
        L += [f"## Split: `{split}`", "",
              "| cond | R^2 (z, pooled) | Pearson r (pooled) | nRMSE (z) | median per-tile r | median RMSE [m] | median signal std [m] |",
              "|------|----------------:|-------------------:|----------:|------------------:|----------------:|----------------------:|"]
        for c, d in byc.items():
            p = d["pooled"]; pt = d["per_tile"]
            L.append(f"| {c} | {_f(p.get('r2'))} | {_f(p.get('pearson'))} | {_f(p.get('nrmse'))} | "
                     f"{_f(pt.get('median_pearson'))} | {_f(pt.get('median_rmse_m'),2)} | {_f(pt.get('median_signal_std_m'),2)} |")
        L.append("")
    L += ["---", "*Generated by `experiments_cvpr/segman/probe_dem_decodability.py`.*"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
