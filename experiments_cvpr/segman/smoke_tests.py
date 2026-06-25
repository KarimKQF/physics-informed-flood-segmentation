"""SegMAN smoke tests (B, C, DEM-shuffle).

  B. Model forward      -> logits are [B, 2, H, W].
  C. Loss backward      -> for each of the 4 loss modes: finite total loss, no
                           NaN, gradients exist; loss_topo == 0 when topo is
                           disabled (ce / dice_ce).
  DEM-shuffle           -> the shuffled-DEM map really changes the per-sample DEM
                           (reproducible derangement), so the control differs
                           from the real-DEM run.

Smoke test A (dataset) is covered by debug_one_batch.py; smoke test D (tiny
training) is run via train_segman.py with --max-epochs/--max-train-batches.

    python experiments_cvpr/segman/smoke_tests.py --config configs/segman/segman_dice_ce_topo.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

SEGMAN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SEGMAN_ROOT.parents[1]
for p in (str(SEGMAN_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from step6c_v3_train import TopographyDataModule, dem_path_for_sample, load_dem_array, tile_id_from_mask_path  # noqa: E402

from model.segman_model import build_segman  # noqa: E402
from segman_losses.segman_loss import MODES, SegManCombinedLoss  # noqa: E402
from train_segman import InputAssembler, get_target  # noqa: E402


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as fh:
        config = yaml.safe_load(fh)
    dem_map_file = config.get("dem", {}).get("dem_tile_id_map_file")
    if dem_map_file:
        with open(dem_map_file, encoding="utf-8") as f:
            config.setdefault("dem", {})["dem_tile_id_map"] = json.load(f).get("mapping", {})
    return config


def get_one_batch(config, device):
    dm = TopographyDataModule(config, batch_size=int(config["trainer"]["batch_size"]))
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    return {k: (v.to(device) if torch.is_tensor(v) else
                {kk: vv.to(device) for kk, vv in v.items()} if isinstance(v, dict) and k != "filename" else v)
            for k, v in batch.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)
    results: dict[str, object] = {}
    passed = True

    batch = get_one_batch(config, device)
    assembler = InputAssembler(config, device)
    x = assembler(batch["image"])
    target = get_target(batch)
    topo = batch.get("topography")

    model_cfg = dict(config["model"])
    model_cfg.setdefault("in_chans", assembler.in_chans)
    model = build_segman(model_cfg).to(device)

    # ---- B. forward ------------------------------------------------------- #
    model.eval()
    with torch.no_grad():
        logits = model(x)
    B, C, H, W = logits.shape
    fwd_ok = (C == int(config["model"]["num_classes"]) and (H, W) == tuple(target.shape[-2:])
              and bool(torch.isfinite(logits).all()))
    print(f"[B] forward: logits={tuple(logits.shape)} finite={bool(torch.isfinite(logits).all())} -> {'PASS' if fwd_ok else 'FAIL'}")
    results["B_forward"] = {"shape": list(logits.shape), "pass": fwd_ok}
    passed &= fwd_ok

    # ---- C. loss backward for every mode ---------------------------------- #
    loss_cfg = config["loss"]
    topo_cfg = loss_cfg.get("topo", {})
    print("[C] loss backward (per mode):")
    results["C_loss"] = {}
    for mode in MODES:
        model.zero_grad(set_to_none=True)
        model.train()
        crit = SegManCombinedLoss(
            mode=mode, ce_alpha=1.0, lambda_topo=0.5,
            ignore_index=int(loss_cfg.get("ignore_index", -1)),
            water_class=int(loss_cfg.get("water_class", 1)),
            elevation_margin=float(topo_cfg.get("elevation_margin", 0.0)),
            elevation_scale=float(topo_cfg.get("elevation_scale", 1.0)),
            use_elevation_weight=bool(topo_cfg.get("use_elevation_weight", True)),
            neighborhood=str(topo_cfg.get("neighborhood", "4")),
        ).to(device)
        out = model(x)
        losses = crit(out, target, topo)
        total = losses["loss_total"]
        total.backward()
        n_grad = sum(1 for p in model.parameters() if p.grad is not None and torch.isfinite(p.grad).all())
        n_param = sum(1 for p in model.parameters() if p.requires_grad)
        topo_disabled = mode in ("ce", "dice_ce")
        topo_val = float(losses["loss_topo"].detach())
        checks = {
            "total_finite": bool(torch.isfinite(total)),
            "grads_exist_finite": n_grad == n_param and n_grad > 0,
            "topo_zero_when_disabled": (topo_val == 0.0) if topo_disabled else True,
            "dice_zero_when_ce_only": (float(losses["loss_dice"].detach()) == 0.0) if mode == "ce" else True,
        }
        mode_ok = all(checks.values())
        print(f"    {mode:30s} total={float(total):.4f} ce={float(losses['loss_ce']):.4f} "
              f"dice={float(losses['loss_dice']):.4f} topo={topo_val:.6f} "
              f"grads={n_grad}/{n_param} -> {'PASS' if mode_ok else 'FAIL'}")
        results["C_loss"][mode] = {**{k: float(losses[k].detach()) for k in
                                      ("loss_total", "loss_ce", "loss_dice", "loss_topo")},
                                   "checks": checks, "pass": mode_ok}
        passed &= mode_ok

    # ---- DEM shuffle check ------------------------------------------------ #
    print("[DEM-shuffle] real vs shuffled DEM per sample:")
    dm = TopographyDataModule(config, batch_size=1)
    dm.setup("fit")
    ds = dm.train_dataset
    dem_map = config.get("dem", {}).get("dem_tile_id_map") or {}
    diffs, deranged = [], []
    for idx in range(min(8, len(ds.samples))):
        tile = tile_id_from_mask_path(ds.samples[idx]["mask"])
        mapped = dem_map.get(tile, tile)
        real = load_dem_array(dem_path_for_sample(config, split="train", tile_id=tile))
        shuf = load_dem_array(dem_path_for_sample(config, split="train", tile_id=mapped))
        deranged.append(mapped != tile)
        diffs.append(float(np.abs(real - shuf).mean()))
    has_map = len(dem_map) > 0
    if has_map:
        shuffle_ok = all(deranged) and all(d > 0 for d in diffs)
        print(f"    map_size={len(dem_map)} all_deranged={all(deranged)} mean|real-shuf|={np.mean(diffs):.2f} -> {'PASS' if shuffle_ok else 'FAIL'}")
    else:
        shuffle_ok = True
        print(f"    no shuffle map in this config (real-DEM run) -> N/A (PASS)")
    results["DEM_shuffle"] = {"has_map": has_map, "map_size": len(dem_map),
                              "all_deranged": bool(all(deranged)) if has_map else None,
                              "mean_abs_diff": float(np.mean(diffs)) if diffs else None, "pass": shuffle_ok}
    passed &= shuffle_ok

    print("=" * 60)
    print(f"SMOKE TESTS B/C/DEM: {'ALL PASS' if passed else 'FAILURES PRESENT'}")
    out_json = SEGMAN_ROOT / "smoke_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"results -> {out_json}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
