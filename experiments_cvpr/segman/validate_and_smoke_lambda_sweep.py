"""
Part C + D: Validate and smoke-test the 6 lambda_topo sweep configs.

Validation checks (Part C):
  - seed == 0
  - N_train == 100
  - correct manifest path
  - correct DEM shuffle map for shuffled configs
  - loss mode correct
  - lambda_topo == expected value
  - topo enabled
  - DEM input flag == False
  - shuffled flag correct
  - run_tag distinct and matches expected pattern
  - no existing completed run would be overwritten

Smoke checks (Part D) for lambda 1.0/2.0/4.0 real + lambda 4.0 shuffled:
  - dataset loads, N_train == 100
  - input [B,15,512,512]
  - mask values in {-1,0,1}
  - DEM [B,1,512,512]
  - logits [B,2,512,512]
  - loss_total finite
  - loss_topo finite and > 0 (lambda forced to target value)
  - no NaN/Inf
  - DEM not in model input
  - for shuffled: shuffle map valid, bijective, 0 self-maps

Usage:
    python experiments_cvpr/segman/validate_and_smoke_lambda_sweep.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
import yaml

SEGMAN_ROOT = Path(__file__).resolve().parent
REPO_ROOT   = SEGMAN_ROOT.parents[1]
RUNS_ROOT   = Path("E:/flood_research/experiments/segman/runs")

for p in (str(SEGMAN_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import step6c_lambda05_train as t6c
from step6c_v3_train import TopographyDataModule
from segman_losses.segman_loss import build_loss
from model.segman_model import build_segman
from train_segman import InputAssembler, get_target

CFG_DIR = REPO_ROOT / "configs" / "segman" / "lambda_sweep_n100"
MANIFEST_N100_S0 = REPO_ROOT / "manifests" / "terramind_baseline" / "low_data_multiseed_n100" / "flood_train_low_data_n100_seed0.txt"
SHUFFLE_MAP_S0   = REPO_ROOT / "manifests" / "terramind_baseline" / "low_data_multiseed_n100" / "dem_shuffle_map_n100_seed0.json"

VARIANTS = [
    # (cfg_file, cond, lambda_topo, is_shuffled, run_tag)
    ("n100_seed0_dice_ce_topo_lambda1p0.yaml",              "dice_ce_topo",              1.0, False, "segman_n100_topo_lambda1p0_seed0"),
    ("n100_seed0_dice_ce_topo_dem_shuffled_lambda1p0.yaml", "dice_ce_topo_dem_shuffled", 1.0, True,  "segman_n100_topo_dem_shuffled_lambda1p0_seed0"),
    ("n100_seed0_dice_ce_topo_lambda2p0.yaml",              "dice_ce_topo",              2.0, False, "segman_n100_topo_lambda2p0_seed0"),
    ("n100_seed0_dice_ce_topo_dem_shuffled_lambda2p0.yaml", "dice_ce_topo_dem_shuffled", 2.0, True,  "segman_n100_topo_dem_shuffled_lambda2p0_seed0"),
    ("n100_seed0_dice_ce_topo_lambda4p0.yaml",              "dice_ce_topo",              4.0, False, "segman_n100_topo_lambda4p0_seed0"),
    ("n100_seed0_dice_ce_topo_dem_shuffled_lambda4p0.yaml", "dice_ce_topo_dem_shuffled", 4.0, True,  "segman_n100_topo_dem_shuffled_lambda4p0_seed0"),
]

SMOKE_VARIANTS = {
    "segman_n100_topo_lambda1p0_seed0",
    "segman_n100_topo_lambda2p0_seed0",
    "segman_n100_topo_lambda4p0_seed0",
    "segman_n100_topo_dem_shuffled_lambda4p0_seed0",
}

PASS = "[PASS]"
FAIL = "[FAIL]"


def check(ok: bool, label: str, errors: list) -> bool:
    print(f"  {'OK ' if ok else 'FAIL'} {label}")
    if not ok:
        errors.append(label)
    return ok


def load_config(cfg_path: Path) -> dict:
    with cfg_path.open("r", encoding="utf-8-sig") as fh:
        cfg = yaml.safe_load(fh)
    dem_map_file = cfg.get("dem", {}).get("dem_tile_id_map_file")
    if dem_map_file:
        with open(dem_map_file, encoding="utf-8") as f:
            raw = json.load(f)
        cfg.setdefault("dem", {})["dem_tile_id_map"] = raw.get("mapping", {})
    return cfg


def part_c_validate(cfg_file: str, cond: str, lam: float, is_shuffled: bool, run_tag: str) -> list[str]:
    errors: list[str] = []
    cfg_path = CFG_DIR / cfg_file
    if not cfg_path.exists():
        errors.append(f"Config file missing: {cfg_path}")
        return errors

    cfg = load_config(cfg_path)

    check(int(cfg.get("seed", -1)) == 0,                          "seed == 0", errors)
    check(int(cfg.get("seed_everything", -1)) == 0,               "seed_everything == 0", errors)
    check(cfg["dataset_policy"]["train"] == 100,                   f"N_train == 100", errors)
    check(cfg.get("run_tag") == run_tag,                           f"run_tag == {run_tag}", errors)

    # Manifest (normalize both to Path before comparing — handles / vs \ on Windows)
    ms = cfg["data"]["init_args"].get("train_split", "")
    check(Path(ms).resolve() == MANIFEST_N100_S0.resolve(),
          f"train_split points to N=100 seed0 manifest", errors)

    # Loss
    check(cfg["loss"]["mode"] == cond,                             f"loss.mode == {cond}", errors)
    check(abs(float(cfg["loss"]["lambda_topo"]) - lam) < 1e-9,    f"lambda_topo == {lam}", errors)
    check(cfg["loss"]["lambda_schedule"]["type"] == "warmup_linear",
          "lambda_schedule type == warmup_linear", errors)

    # DEM
    check(cfg["dem"]["use_as_model_input"] is False,               "DEM not model input", errors)
    check(cfg["dem"]["use_in_loss"] is True,                       "DEM used in loss", errors)

    # Shuffled specific
    if is_shuffled:
        dm_file = cfg["dem"].get("dem_tile_id_map_file", "")
        check(Path(dm_file).resolve() == SHUFFLE_MAP_S0.resolve(),
              "shuffled: dem_tile_id_map_file points to N=100 seed0 shuffle map", errors)
        dm_map = cfg["dem"].get("dem_tile_id_map") or {}
        check(len(dm_map) == 100,                                  f"shuffle map has 100 entries (got {len(dm_map)})", errors)
        n_self = sum(1 for k, v in dm_map.items() if k == v)
        check(n_self == 0,                                         f"shuffle map has 0 self-maps (got {n_self})", errors)
        keys, vals = set(dm_map.keys()), set(dm_map.values())
        check(keys == vals,                                        "shuffle map is bijective", errors)
        check(cfg["guardrails"].get("dem_shuffled_control") is True,
              "guardrail dem_shuffled_control == true", errors)
    else:
        dm = cfg["dem"].get("dem_tile_id_map")
        check(dm is None,                                          "real DEM: dem_tile_id_map is null", errors)

    # No existing completed run
    run_dir = Path(cfg["run_dir"])
    summary = run_dir / "metrics" / f"{run_tag}_summary.json"
    if summary.exists():
        try:
            s = json.loads(summary.read_text(encoding="utf-8"))
            check(s.get("status") != "done",                      f"no completed run exists at {run_dir}", errors)
        except Exception:
            pass
    else:
        print(f"  OK  no completed run at {run_dir} (safe to launch)")

    return errors


def part_d_smoke(cfg_file: str, run_tag: str, lam: float, is_shuffled: bool) -> list[str]:
    errors: list[str] = []
    cfg_path = CFG_DIR / cfg_file
    cfg = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    # DataModule
    try:
        dm = TopographyDataModule(cfg, batch_size=2)
        dm.setup("fit")
        loader = dm.train_dataloader()
        check(len(loader.dataset) == 100, f"N_train == 100 (got {len(loader.dataset)})", errors)
    except Exception as exc:
        errors.append(f"DataModule failed: {exc}"); return errors

    # Batch
    try:
        raw_batch = next(iter(loader))
        batch = t6c.move_batch(raw_batch, device)
    except Exception as exc:
        errors.append(f"Batch load failed: {exc}"); return errors

    # Input assembler
    assembler = InputAssembler(cfg, device)
    x = assembler(batch["image"])
    B, C, H, W = x.shape
    check(C == 15,              f"input channels == 15 (got {C})", errors)
    check(H == 512 and W == 512, f"spatial 512x512 (got {H}x{W})", errors)
    check(not torch.isnan(x).any().item(), "no NaN in input", errors)
    print(f"  input shape: {list(x.shape)}")

    # Mask
    target = get_target(batch)
    bad = set(target.cpu().numpy().flatten().tolist()) - {-1, 0, 1}
    check(len(bad) == 0, f"mask values in {{-1,0,1}} (bad: {bad})", errors)

    # DEM
    topo = batch.get("topography")
    check(topo is not None, "DEM returned in batch", errors)
    if topo is not None:
        check(topo.ndim == 4 and topo.shape[1] == 1, f"DEM [B,1,H,W] (got {list(topo.shape)})", errors)
        check(not torch.isnan(topo).any().item(), "no NaN in DEM", errors)
        print(f"  DEM shape: {list(topo.shape)}")

    # Model
    try:
        mc = dict(cfg["model"]); mc["in_chans"] = assembler.in_chans
        model = build_segman(mc).to(device)
        n_p = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  model: SegMAN-s params={n_p:.2f}M")
    except Exception as exc:
        errors.append(f"Model build failed: {exc}"); return errors

    # Loss
    try:
        criterion = build_loss(cfg).to(device)
        criterion.set_lambda_topo(lam)
        print(f"  lambda_topo forced to {lam} (smoke bypass warmup)")
    except Exception as exc:
        errors.append(f"Loss build failed: {exc}"); return errors

    # Forward
    model.eval()
    with torch.no_grad():
        try:
            logits = model(x)
        except Exception as exc:
            errors.append(f"Forward failed: {exc}"); return errors

    check(logits.shape[1] == 2, f"output classes == 2 (got {logits.shape[1]})", errors)
    check(not torch.isnan(logits).any().item(), "no NaN in logits", errors)
    check(not torch.isinf(logits).any().item(), "no Inf in logits", errors)
    print(f"  logit shape: {list(logits.shape)}")

    # Loss forward
    try:
        losses = criterion(logits, target, topo)
    except Exception as exc:
        errors.append(f"Loss forward failed: {exc}"); return errors

    loss_total = float(losses["loss_total"].detach().cpu())
    loss_topo  = float(losses.get("loss_topo", torch.tensor(0.0)).detach().cpu())
    check(math.isfinite(loss_total), f"loss_total finite ({loss_total:.5f})", errors)
    check(math.isfinite(loss_topo) and loss_topo > 0, f"loss_topo finite and > 0 ({loss_topo:.5f})", errors)
    print(f"  loss_total={loss_total:.5f}  loss_topo={loss_topo:.5f}  (lambda={lam})")
    check(not math.isnan(loss_total) and not math.isinf(loss_total), "no NaN/Inf in loss_total", errors)

    return errors


def main() -> int:
    print("=" * 70)
    print("PART C — Config validation")
    print("=" * 70)

    val_results: dict[str, list[str]] = {}
    for cfg_file, cond, lam, is_shuf, tag in VARIANTS:
        print(f"\n  [{tag}]")
        errs = part_c_validate(cfg_file, cond, lam, is_shuf, tag)
        val_results[tag] = errs

    print("\n" + "=" * 70)
    print("PART C — Validation summary")
    print("=" * 70)
    val_fail = 0
    for tag, errs in val_results.items():
        status = "PASS" if not errs else f"FAIL ({len(errs)})"
        print(f"  {tag:<52} {status}")
        for e in errs: print(f"      - {e}")
        val_fail += len(errs)

    if val_fail:
        print(f"\n{val_fail} validation checks FAILED. Fix before smoke tests.")
        return 1

    print("\n" + "=" * 70)
    print("PART D — Smoke tests")
    print("=" * 70)

    smoke_results: dict[str, list[str]] = {}
    for cfg_file, cond, lam, is_shuf, tag in VARIANTS:
        if tag not in SMOKE_VARIANTS:
            print(f"\n  SKIP smoke: {tag} (not in smoke set)")
            continue
        print(f"\n{'='*60}")
        print(f"  Smoke: {tag}  (lambda={lam}, shuffled={is_shuf})")
        print(f"{'='*60}")
        errs = part_d_smoke(cfg_file, tag, lam, is_shuf)
        smoke_results[tag] = errs

    print("\n" + "=" * 70)
    print("PART D — Smoke summary")
    print("=" * 70)
    smoke_fail = 0
    for tag, errs in smoke_results.items():
        status = "PASS" if not errs else f"FAIL ({len(errs)})"
        print(f"  {tag:<52} {status}")
        for e in errs: print(f"      - {e}")
        smoke_fail += len(errs)

    if smoke_fail == 0:
        print("\nAll validation and smoke tests PASSED. Safe to launch lambda sweep.")
        return 0
    else:
        print(f"\n{smoke_fail} smoke checks FAILED.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
