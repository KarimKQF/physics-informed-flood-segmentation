"""
Smoke test for SegMAN N=100 diagnostic configs.

Checks for each of the 3 seed0 configs (one per condition):
  1. Dataset loads and returns the expected N=100 tiles
  2. Input tensor shape is [B, 15, 512, 512]
  3. Mask values are in {-1, 0, 1}
  4. DEM shape is valid [B, 1, 512, 512] when loaded
  5. Forward pass returns [B, 2, H, W]
  6. Loss is finite
  7. loss_topo is 0.0 / None for Dice+CE
  8. loss_topo is finite and > 0 for topo variants (after warmup bypass)
  9. Shuffled DEM map is valid (bijective, no self-maps)
 10. No NaN / Inf in logits or loss components

Does NOT write any artifacts to the N=100 run directories.
Uses --max-epochs 1 --max-train-batches 2 via train_segman.run() with a temp dir.

Usage:
    python experiments_cvpr/segman/smoke_test_n100.py
"""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import torch
import yaml

SEGMAN_ROOT = Path(__file__).resolve().parent
REPO_ROOT   = SEGMAN_ROOT.parents[1]
for p in (str(SEGMAN_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import step6c_lambda05_train as t6c
from step6c_v3_train import TopographyDataModule
from segman_losses.segman_loss import build_loss, lambda_for_epoch
from model.segman_model import build_segman

CFG_DIR = REPO_ROOT / "configs" / "segman" / "multiseed_n100"

SMOKE_CONFIGS = [
    ("dice_ce",                   "n100_seed0_dice_ce.yaml"),
    ("dice_ce_topo",              "n100_seed0_dice_ce_topo.yaml"),
    ("dice_ce_topo_dem_shuffled", "n100_seed0_dice_ce_topo_dem_shuffled.yaml"),
]

PASS = "[PASS]"
FAIL = "[FAIL]"


def load_config(cfg_path: Path) -> dict:
    with cfg_path.open("r", encoding="utf-8-sig") as fh:
        cfg = yaml.safe_load(fh)
    dem_map_file = cfg.get("dem", {}).get("dem_tile_id_map_file")
    if dem_map_file:
        import json
        with open(dem_map_file, encoding="utf-8") as f:
            cfg.setdefault("dem", {})["dem_tile_id_map"] = json.load(f).get("mapping", {})
    return cfg


def check(cond: bool, label: str, errors: list) -> bool:
    if cond:
        print(f"  {PASS}  {label}")
    else:
        print(f"  {FAIL}  {label}")
        errors.append(label)
    return cond


def smoke_one(cond_name: str, cfg_path: Path) -> list[str]:
    print(f"\n{'='*60}")
    print(f"  Smoke: {cfg_path.name}  ({cond_name})")
    print(f"{'='*60}")
    errors: list[str] = []

    cfg = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    # ── Config guards ──────────────────────────────────────────────────────
    check(cfg["dem"]["use_as_model_input"] is False,
          "DEM not used as model input (guardrail)", errors)
    check(cfg["dataset_policy"]["train"] == 100,
          f"dataset_policy.train == 100 (got {cfg['dataset_policy']['train']})", errors)
    check(cfg["loss"]["mode"] == cond_name,
          f"loss.mode == {cond_name!r}", errors)
    check(cfg["model"]["in_chans"] == 15,
          "model.in_chans == 15", errors)

    if cond_name == "dice_ce":
        check(cfg["loss"]["lambda_topo"] == 0.0,
              "lambda_topo == 0.0 for Dice+CE", errors)
    else:
        check(cfg["loss"]["lambda_topo"] == 0.5,
              f"lambda_topo == 0.5 for {cond_name}", errors)

    if cond_name == "dice_ce_topo_dem_shuffled":
        dm_map = cfg["dem"].get("dem_tile_id_map", {})
        check(len(dm_map) == 100, f"DEM shuffle map has 100 entries (got {len(dm_map)})", errors)
        n_self = sum(1 for k, v in dm_map.items() if k == v)
        check(n_self == 0, f"DEM shuffle map has 0 self-maps (got {n_self})", errors)
        vals = set(dm_map.values())
        keys = set(dm_map.keys())
        check(vals == keys, "DEM shuffle map is bijective (same keys and values)", errors)
    else:
        check(cfg["dem"].get("dem_tile_id_map") is None,
              "dem_tile_id_map is null (not shuffled)", errors)

    # ── DataModule ─────────────────────────────────────────────────────────
    try:
        dm = TopographyDataModule(cfg, batch_size=2)
        dm.setup("fit")
        train_loader = dm.train_dataloader()
        n_train = len(train_loader.dataset)
        check(n_train == 100, f"train dataset size == 100 (got {n_train})", errors)
    except Exception as exc:
        errors.append(f"DataModule setup failed: {exc}")
        print(f"  {FAIL}  DataModule setup: {exc}")
        return errors

    # ── Grab one batch ─────────────────────────────────────────────────────
    try:
        raw_batch = next(iter(train_loader))
        batch = t6c.move_batch(raw_batch, device)
    except Exception as exc:
        errors.append(f"Batch load failed: {exc}")
        print(f"  {FAIL}  Batch load: {exc}")
        return errors

    # Check tensor shapes
    imgs_dict = batch.get("image", {})
    s2 = imgs_dict.get("S2L1C")
    s1 = imgs_dict.get("S1GRD")
    check(s2 is not None and s2.ndim == 4, "S2L1C in batch with 4D", errors)
    check(s1 is not None and s1.ndim == 4, "S1GRD in batch with 4D", errors)

    # Input assembler
    from train_segman import InputAssembler, get_target
    assembler = InputAssembler(cfg, device)
    x = assembler(batch["image"])
    B, C, H, W = x.shape
    check(C == 15, f"input channels == 15 (got {C})", errors)
    check(H == 512 and W == 512, f"spatial size 512x512 (got {H}x{W})", errors)
    check(not torch.isnan(x).any().item(), "No NaN in input tensor", errors)
    print(f"  input shape: {list(x.shape)}")

    # Mask
    target = get_target(batch)
    unique_vals = set(target.cpu().numpy().flatten().tolist())
    bad_vals = unique_vals - {-1, 0, 1}
    check(len(bad_vals) == 0, f"Mask values in {{-1,0,1}} (unexpected: {bad_vals})", errors)
    print(f"  mask shape: {list(target.shape)}  unique: {sorted(unique_vals)}")

    # DEM
    topo = batch.get("topography")
    if topo is not None:
        check(topo.ndim == 4 and topo.shape[1] == 1,
              f"DEM shape [B,1,H,W] (got {list(topo.shape)})", errors)
        check(not torch.isnan(topo).any().item(), "No NaN in DEM", errors)
        print(f"  DEM shape:  {list(topo.shape)}")
    else:
        print("  DEM: not returned in batch (ok for dice_ce if dem.use_in_loss=False)")

    # ── Model & Loss ───────────────────────────────────────────────────────
    try:
        model_cfg = dict(cfg["model"])
        model_cfg["in_chans"] = assembler.in_chans
        model = build_segman(model_cfg).to(device)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  model: SegMAN-{model_cfg.get('variant','s')}  params={n_params:.2f}M")
    except Exception as exc:
        errors.append(f"Model build failed: {exc}")
        print(f"  {FAIL}  Model build: {exc}")
        return errors

    try:
        criterion = build_loss(cfg).to(device)
    except Exception as exc:
        errors.append(f"Loss build failed: {exc}")
        print(f"  {FAIL}  Loss build: {exc}")
        return errors

    # Force lambda_topo=0.5 for topo variants (bypass warmup for smoke test)
    if cond_name in ("dice_ce_topo", "dice_ce_topo_dem_shuffled"):
        criterion.set_lambda_topo(0.5)
        print("  lambda_topo forced to 0.5 (smoke test bypass warmup)")
    else:
        criterion.set_lambda_topo(0.0)

    # Forward pass
    model.eval()
    with torch.no_grad():
        try:
            logits = model(x)
        except Exception as exc:
            errors.append(f"Forward pass failed: {exc}")
            print(f"  {FAIL}  Forward pass: {exc}")
            return errors

    Bo, C_out, Ho, Wo = logits.shape
    check(C_out == 2, f"output classes == 2 (got {C_out})", errors)
    check(not torch.isnan(logits).any().item(), "No NaN in logits", errors)
    check(not torch.isinf(logits).any().item(), "No Inf in logits", errors)
    print(f"  logit shape: {list(logits.shape)}")

    # Loss
    try:
        losses = criterion(logits, target, topo)
    except Exception as exc:
        errors.append(f"Loss forward failed: {exc}")
        print(f"  {FAIL}  Loss forward: {exc}")
        return errors

    loss_total = float(losses["loss_total"].detach().cpu())
    loss_topo  = float(losses["loss_topo"].detach().cpu()) if "loss_topo" in losses else 0.0
    check(math.isfinite(loss_total), f"loss_total is finite (got {loss_total:.6f})", errors)
    print(f"  loss_total={loss_total:.6f}  loss_topo={loss_topo:.6f}")

    if cond_name == "dice_ce":
        check(abs(loss_topo) < 1e-8,
              f"loss_topo == 0 for Dice+CE (got {loss_topo:.2e})", errors)
    else:
        check(math.isfinite(loss_topo) and loss_topo >= 0,
              f"loss_topo is finite >= 0 for {cond_name} (got {loss_topo:.6f})", errors)
        check(loss_topo > 0,
              f"loss_topo > 0 (topo term active, got {loss_topo:.6f})", errors)

    return errors


def main() -> int:
    all_errors: dict[str, list[str]] = {}
    for cond_name, cfg_file in SMOKE_CONFIGS:
        cfg_path = CFG_DIR / cfg_file
        errs = smoke_one(cond_name, cfg_path)
        all_errors[cond_name] = errs

    print("\n" + "="*60)
    print("SMOKE TEST SUMMARY")
    print("="*60)
    total_fail = 0
    for cond_name, errs in all_errors.items():
        status = "PASS" if not errs else f"FAIL ({len(errs)} errors)"
        print(f"  {cond_name:<38}  {status}")
        for e in errs:
            print(f"      - {e}")
        total_fail += len(errs)

    if total_fail == 0:
        print("\nAll smoke tests PASSED. Safe to launch N=100 chain.")
        return 0
    else:
        print(f"\n{total_fail} checks FAILED. Fix before launching chain.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
