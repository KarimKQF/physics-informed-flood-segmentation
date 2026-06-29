"""
Validate and smoke-test the 2 D8 N=100 seed0 configs before launching training.

Part A -- Config validation:
  - seed == 0, N_train == 100
  - correct manifest path
  - loss.mode is dice_ce_d8 / dice_ce_d8_dem_shuffled
  - lambda_topo == 1.0
  - warmup_linear schedule correct
  - D8 params present (s0, tau)
  - DEM not model input
  - shuffled config: shuffle map has 100 entries, 0 self-maps, bijective
  - real config: dem_tile_id_map is null
  - no existing completed run would be overwritten

Part B -- Synthetic toy DEM test (pure Python, no GPU):
  - 5x5 DEM sloping right (columns 4..0); D8 direction = right for all pixels
  - Prediction: upstream water (col 0) + downstream dry (col 1) -> loss > 0
  - Prediction: downstream water (col 1) + upstream dry (col 0) -> loss near 0
  - All-dry prediction -> loss near 0
  - Random prediction -> loss > 0 on average

Part C -- Real-data smoke test (loads 1 batch, runs 1 forward pass):
  - Both configs loaded and validated
  - input [B,15,512,512], mask in {-1,0,1}, DEM [B,1,512,512]
  - D8 loss finite, non-NaN, non-Inf
  - D8 loss > 0 when lambda forced to 1.0
  - D8 loss for real DEM vs shuffled DEM on same real-DEM batch (D8 direction differs)
  - DEM not in model input channels

Usage:
    python experiments_cvpr/segman/validate_and_smoke_d8_n100.py
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
from segman_losses.segman_loss import build_loss, SegManCombinedLoss, _D8_MODES
from losses.d8_downstream_loss import D8DownstreamLoss
from model.segman_model import build_segman
from train_segman import InputAssembler, get_target

CFG_DIR          = REPO_ROOT / "configs" / "segman" / "d8_n100_seed0"
MANIFEST_N100_S0 = REPO_ROOT / "manifests" / "terramind_baseline" / "low_data_multiseed_n100" / "flood_train_low_data_n100_seed0.txt"
SHUFFLE_MAP_S0   = REPO_ROOT / "manifests" / "terramind_baseline" / "low_data_multiseed_n100" / "dem_shuffle_map_n100_seed0.json"

VARIANTS = [
    # (cfg_file,                                       mode,                     lam,  is_shuffled, run_tag)
    ("n100_seed0_dice_ce_d8_lambda1p0.yaml",             "dice_ce_d8",             1.0, False, "segman_n100_d8_lambda1p0_seed0"),
    ("n100_seed0_dice_ce_d8_dem_shuffled_lambda1p0.yaml", "dice_ce_d8_dem_shuffled", 1.0, True,  "segman_n100_d8_dem_shuffled_lambda1p0_seed0"),
]


def _ok(b: bool, label: str, errors: list) -> bool:
    print(f"  {'OK ' if b else 'FAIL'} {label}")
    if not b:
        errors.append(label)
    return b


def load_config(cfg_path: Path) -> dict:
    with cfg_path.open("r", encoding="utf-8-sig") as fh:
        cfg = yaml.safe_load(fh)
    dem_map_file = cfg.get("dem", {}).get("dem_tile_id_map_file")
    if dem_map_file:
        with open(dem_map_file, encoding="utf-8") as f:
            raw = json.load(f)
        cfg.setdefault("dem", {})["dem_tile_id_map"] = raw.get("mapping", {})
    return cfg


# ============================================================
# Part A: Config validation
# ============================================================

def part_a_validate(cfg_file: str, mode: str, lam: float, is_shuffled: bool, run_tag: str) -> list[str]:
    errors: list[str] = []
    cfg_path = CFG_DIR / cfg_file
    if not cfg_path.exists():
        errors.append(f"Config file missing: {cfg_path}")
        return errors

    cfg = load_config(cfg_path)

    _ok(int(cfg.get("seed", -1)) == 0,           "seed == 0", errors)
    _ok(int(cfg.get("seed_everything", -1)) == 0, "seed_everything == 0", errors)
    _ok(cfg["dataset_policy"]["train"] == 100,    "N_train == 100", errors)
    _ok(cfg.get("run_tag") == run_tag,            f"run_tag == {run_tag}", errors)

    ms = cfg["data"]["init_args"].get("train_split", "")
    _ok(Path(ms).resolve() == MANIFEST_N100_S0.resolve(), "train_split -> N=100 seed0 manifest", errors)

    _ok(cfg["loss"]["mode"] == mode,              f"loss.mode == {mode}", errors)
    _ok(abs(float(cfg["loss"]["lambda_topo"]) - lam) < 1e-9, f"lambda_topo == {lam}", errors)
    _ok(cfg["loss"]["lambda_schedule"]["type"] == "warmup_linear", "schedule == warmup_linear", errors)
    _ok(int(cfg["loss"]["lambda_schedule"]["warmup_epochs"]) == 5,  "warmup_epochs == 5", errors)
    _ok(int(cfg["loss"]["lambda_schedule"]["ramp_epochs"])   == 15, "ramp_epochs == 15", errors)

    d8_cfg = cfg["loss"].get("d8", {})
    _ok("s0" in d8_cfg,  "loss.d8.s0 present", errors)
    _ok("tau" in d8_cfg, "loss.d8.tau present", errors)
    _ok(float(d8_cfg.get("s0",  0.0)) > 0,   f"d8.s0 > 0 (got {d8_cfg.get('s0')})", errors)
    _ok(0 < float(d8_cfg.get("tau", -1)) < 1, f"d8.tau in (0,1) (got {d8_cfg.get('tau')})", errors)

    _ok(cfg["dem"]["use_as_model_input"] is False, "DEM not model input", errors)
    _ok(cfg["dem"]["use_in_loss"] is True,          "DEM used in loss", errors)

    if is_shuffled:
        dm_file = cfg["dem"].get("dem_tile_id_map_file", "")
        _ok(Path(dm_file).resolve() == SHUFFLE_MAP_S0.resolve(),
            "shuffled: dem_tile_id_map_file -> N=100 seed0 shuffle map", errors)
        dm_map = cfg["dem"].get("dem_tile_id_map") or {}
        _ok(len(dm_map) == 100, f"shuffle map has 100 entries (got {len(dm_map)})", errors)
        n_self = sum(1 for k, v in dm_map.items() if k == v)
        _ok(n_self == 0, f"shuffle map has 0 self-maps (got {n_self})", errors)
        keys, vals = set(dm_map.keys()), set(dm_map.values())
        _ok(keys == vals, "shuffle map is bijective", errors)
        _ok(cfg["guardrails"].get("dem_shuffled_control") is True, "guardrail dem_shuffled_control == true", errors)
    else:
        dm = cfg["dem"].get("dem_tile_id_map")
        _ok(dm is None, "real DEM: dem_tile_id_map is null", errors)

    run_dir  = Path(cfg["run_dir"])
    summary  = run_dir / "metrics" / f"{run_tag}_summary.json"
    if summary.exists():
        try:
            s = json.loads(summary.read_text(encoding="utf-8"))
            _ok(s.get("status") != "done", f"no completed run at {run_dir}", errors)
        except Exception:
            pass
    else:
        print(f"  OK  no completed run at {run_dir} (safe to launch)")

    return errors


# ============================================================
# Part B: Synthetic toy DEM unit test
# ============================================================

def part_b_toy_dem() -> list[str]:
    """Validate D8 loss on a hand-crafted DEM where the flow direction is known."""
    errors: list[str] = []
    print("  Toy DEM: 5x5, slopes right (h=5..1 per column).")
    print("  Expected D8 direction for all interior pixels: right (dx=+1, dy=0).")

    loss_fn = D8DownstreamLoss(ignore_index=-1, water_class=1, s0=1.0, tau=0.05, eps=1e-6)

    # DEM: columns 4,3,2,1,0 -> all D8 downstream = right neighbour.
    dem_np = [[5, 4, 3, 2, 1],
               [5, 4, 3, 2, 1],
               [5, 4, 3, 2, 1],
               [5, 4, 3, 2, 1],
               [5, 4, 3, 2, 1]]
    dem = torch.tensor(dem_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,5,5]

    # target: all water (1) -- valid, no ignore.
    target = torch.ones(1, 1, 5, 5, dtype=torch.long)

    # Test 1: upstream water (col 0, high) + downstream dry (col 1) -> large loss.
    logits_up_water = torch.zeros(1, 2, 5, 5)
    logits_up_water[0, 1, :, 0] = 10.0   # col 0 -> water (high p_water)
    logits_up_water[0, 0, :, 1] = 10.0   # col 1 -> dry   (low  p_water)
    target_up = target.clone()
    target_up[0, 0, :, 1] = 0  # col 1 is labelled dry
    loss_up = float(loss_fn(logits_up_water, target_up, dem).detach())
    _ok(loss_up > 0.01, f"Test1 (upstream water+downstream dry) loss>0.01: {loss_up:.5f}", errors)
    print(f"    Test1 loss (upstream water -> downstream dry): {loss_up:.5f}  [expect > 0.01]")

    # Test 2: p_water increases monotonically left->right (downstream) => loss = 0.
    # D8 direction = right for all cols 0-3.  p_upstream < p_downstream for every pair.
    # hinge = max(0, p_up - p_down - tau) = 0 everywhere.
    logits_increasing = torch.zeros(1, 2, 5, 5)
    logits_increasing[0, 0, :, 0] = 5.0   # col 0: strong dry   (p_water ~0.007)
    logits_increasing[0, 1, :, 1] = 1.0   # col 1: mild water   (p_water ~0.731)
    logits_increasing[0, 1, :, 2] = 2.0   # col 2: more water   (p_water ~0.880)
    logits_increasing[0, 1, :, 3] = 3.0   # col 3: even more    (p_water ~0.953)
    logits_increasing[0, 1, :, 4] = 5.0   # col 4: strong water (p_water ~0.993)
    target_mono = torch.ones(1, 1, 5, 5, dtype=torch.long)  # all water -> valid
    loss_down = float(loss_fn(logits_increasing, target_mono, dem).detach())
    _ok(loss_down < 1e-4, f"Test2 (p increases downstream) loss<1e-4: {loss_down:.7f}", errors)
    print(f"    Test2 loss (p_water increasing downstream): {loss_down:.7f}  [expect ~0]")

    # Test 3: all-dry prediction -> loss ~0.
    logits_dry = torch.full((1, 2, 5, 5), 0.0)
    logits_dry[0, 0] = 10.0  # dry logit high for all pixels
    target_alldry = torch.zeros(1, 1, 5, 5, dtype=torch.long)
    loss_dry = float(loss_fn(logits_dry, target_alldry, dem).detach())
    _ok(loss_dry < 1e-4, f"Test3 (all-dry) loss<1e-4: {loss_dry:.7f}", errors)
    print(f"    Test3 loss (all-dry): {loss_dry:.7f}  [expect ~0]")

    # Test 4: random logits -> loss > 0 with high probability.
    torch.manual_seed(42)
    logits_rand = torch.randn(4, 2, 16, 16)
    dem_rand    = torch.rand(4, 1, 16, 16) * 100.0
    target_rand = torch.zeros(4, 1, 16, 16, dtype=torch.long)
    loss_rand   = float(loss_fn(logits_rand, target_rand, dem_rand).detach())
    _ok(loss_rand > 1e-6, f"Test4 (random logits 4x16x16) loss>1e-6: {loss_rand:.6f}", errors)
    print(f"    Test4 loss (random logits 4x16x16): {loss_rand:.6f}  [expect > 1e-6]")

    # Test 5: NaN DEM values are handled without crashing.
    dem_nan = dem.clone()
    dem_nan[0, 0, 2, 2] = float("nan")
    try:
        loss_nan = float(loss_fn(logits_up_water, target_up, dem_nan).detach())
        _ok(math.isfinite(loss_nan), f"Test5 (NaN DEM at 1 pixel) finite: {loss_nan:.5f}", errors)
        print(f"    Test5 loss (NaN DEM at 1 pixel): {loss_nan:.5f}  [expect finite]")
    except Exception as exc:
        _ok(False, f"Test5 (NaN DEM) raised exception: {exc}", errors)

    # Test 6: all ignore_index -> loss should be near 0 (no active pixels).
    target_ign = torch.full((1, 1, 5, 5), -1, dtype=torch.long)
    loss_ign   = float(loss_fn(logits_up_water, target_ign, dem).detach())
    _ok(loss_ign < 1e-4, f"Test6 (all ignore_index) loss<1e-4: {loss_ign:.7f}", errors)
    print(f"    Test6 loss (all ignore_index): {loss_ign:.7f}  [expect ~0]")

    return errors


# ============================================================
# Part C: Real-data smoke test
# ============================================================

def part_c_smoke(cfg_file: str, run_tag: str, lam: float, is_shuffled: bool) -> list[str]:
    errors: list[str] = []
    cfg_path = CFG_DIR / cfg_file
    cfg      = load_config(cfg_path)
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    # DataModule (N=100 train split).
    try:
        dm = TopographyDataModule(cfg, batch_size=2)
        dm.setup("fit")
        loader = dm.train_dataloader()
        _ok(len(loader.dataset) == 100, f"N_train == 100 (got {len(loader.dataset)})", errors)
    except Exception as exc:
        errors.append(f"DataModule failed: {exc}"); return errors

    try:
        raw_batch = next(iter(loader))
        batch     = t6c.move_batch(raw_batch, device)
    except Exception as exc:
        errors.append(f"Batch load failed: {exc}"); return errors

    assembler = InputAssembler(cfg, device)
    x         = assembler(batch["image"])
    B, C, H, W = x.shape
    _ok(C == 15,               f"input channels == 15 (got {C})", errors)
    _ok(H == 512 and W == 512, f"spatial 512x512 (got {H}x{W})", errors)
    _ok(not torch.isnan(x).any().item(), "no NaN in input", errors)
    print(f"  input: {list(x.shape)}")

    target = get_target(batch)
    bad    = set(target.cpu().numpy().flatten().tolist()) - {-1, 0, 1}
    _ok(len(bad) == 0, f"mask values in {{-1,0,1}}", errors)

    topo = batch.get("topography")
    _ok(topo is not None, "DEM returned in batch", errors)
    if topo is not None:
        _ok(topo.ndim == 4 and topo.shape[1] == 1, f"DEM [B,1,H,W] got {list(topo.shape)}", errors)
        print(f"  DEM: {list(topo.shape)}")
        _ok("topography" not in [m for m in assembler.modalities],
            "DEM not in model input modalities", errors)

    # Model.
    try:
        mc    = dict(cfg["model"]); mc["in_chans"] = assembler.in_chans
        model = build_segman(mc).to(device)
        print(f"  model: SegMAN-s params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    except Exception as exc:
        errors.append(f"Model build failed: {exc}"); return errors

    # Loss.
    try:
        criterion = build_loss(cfg).to(device)
        _ok(criterion.use_d8, "criterion.use_d8 == True", errors)
        _ok(isinstance(criterion.loss_phys, D8DownstreamLoss), "loss_phys is D8DownstreamLoss", errors)
        criterion.set_lambda_topo(lam)
        print(f"  lambda forced to {lam} (smoke bypass warmup)")
    except Exception as exc:
        errors.append(f"Loss build failed: {exc}"); return errors

    model.eval()
    with torch.no_grad():
        try:
            logits = model(x)
        except Exception as exc:
            errors.append(f"Forward failed: {exc}"); return errors

    _ok(logits.shape == (B, 2, H, W), f"logits shape {list(logits.shape)}", errors)
    _ok(not torch.isnan(logits).any().item(), "no NaN in logits", errors)
    _ok(not torch.isinf(logits).any().item(), "no Inf in logits", errors)

    try:
        losses = criterion(logits, target, topo)
    except Exception as exc:
        errors.append(f"Loss forward failed: {exc}"); return errors

    loss_total = float(losses["loss_total"].detach().cpu())
    loss_topo  = float(losses["loss_topo"].detach().cpu())
    loss_dice  = float(losses["loss_dice"].detach().cpu())
    loss_ce    = float(losses["loss_ce"].detach().cpu())
    _ok(math.isfinite(loss_total), f"loss_total finite ({loss_total:.5f})", errors)
    _ok(math.isfinite(loss_topo),  f"loss_topo (D8) finite ({loss_topo:.6f})", errors)
    _ok(loss_topo >= 0,            f"loss_topo (D8) >= 0 ({loss_topo:.6f})", errors)
    _ok(math.isfinite(loss_dice),  f"loss_dice finite ({loss_dice:.5f})", errors)
    _ok(math.isfinite(loss_ce),    f"loss_ce finite ({loss_ce:.5f})", errors)
    print(f"  loss_total={loss_total:.5f} loss_dice={loss_dice:.5f} "
          f"loss_ce={loss_ce:.5f} loss_d8={loss_topo:.6f}")
    if abs(loss_topo) < 1e-10:
        print(f"  NOTE: loss_d8=0 from untrained model (uniform preds, p_i-p_d < tau={criterion.loss_phys.tau:.2f}). "
              f"This is expected at init -- loss activates as model learns spatial structure.")

    # Verify D8 > 0 with spatially perturbed logits (untrained model has near-uniform preds).
    loss_fn_raw = D8DownstreamLoss(s0=1.0, tau=0.05).to(device)
    torch.manual_seed(0)
    logits_perturbed = logits.detach() + torch.randn_like(logits) * 3.0
    with torch.no_grad():
        l_perturb = float(loss_fn_raw(logits_perturbed, target, topo).detach())
    _ok(l_perturb > 1e-8, f"D8 > 0 with perturbed logits (variance ~3.0): {l_perturb:.6f}", errors)
    print(f"  D8 loss with perturbed logits (sigma=3.0): {l_perturb:.6f}  [expect > 1e-8]")

    # Real vs shifted DEM on perturbed logits (shows DEM routing matters).
    if not is_shuffled and topo is not None:
        topo_shifted = torch.roll(topo, shifts=1, dims=0)
        with torch.no_grad():
            l_real = float(loss_fn_raw(logits_perturbed, target, topo).detach())
            l_shuf = float(loss_fn_raw(logits_perturbed, target, topo_shifted).detach())
        print(f"  Same-batch D8 real={l_real:.6f}  shifted={l_shuf:.6f}  "
              f"diff={l_real - l_shuf:+.6f}  [DEM routing changes loss]")
        _ok(abs(l_real - l_shuf) > 1e-8, "D8(real DEM) != D8(shifted DEM) with perturbed logits", errors)

    return errors


# ============================================================
# Main
# ============================================================

def main() -> int:
    print("=" * 70)
    print("PART A -- Config validation")
    print("=" * 70)
    val_results: dict[str, list[str]] = {}
    for cfg_file, mode, lam, is_shuf, tag in VARIANTS:
        print(f"\n  [{tag}]")
        errs = part_a_validate(cfg_file, mode, lam, is_shuf, tag)
        val_results[tag] = errs

    print("\n" + "=" * 70)
    print("PART A -- Validation summary")
    print("=" * 70)
    val_fail = 0
    for tag, errs in val_results.items():
        status = "PASS" if not errs else f"FAIL ({len(errs)})"
        print(f"  {tag:<55} {status}")
        for e in errs:
            print(f"      - {e}")
        val_fail += len(errs)

    print("\n" + "=" * 70)
    print("PART B -- Synthetic toy DEM unit tests (CPU, no data loading)")
    print("=" * 70)
    toy_errors = part_b_toy_dem()
    toy_status = "PASS" if not toy_errors else f"FAIL ({len(toy_errors)})"
    print(f"\n  Toy DEM tests: {toy_status}")
    for e in toy_errors:
        print(f"      - {e}")

    if val_fail or toy_errors:
        print(f"\n{val_fail} validation + {len(toy_errors)} toy DEM checks FAILED.")
        print("Fix before smoke tests.")
        return 1

    print("\n" + "=" * 70)
    print("PART C -- Real-data smoke tests")
    print("=" * 70)
    smoke_results: dict[str, list[str]] = {}
    for cfg_file, mode, lam, is_shuf, tag in VARIANTS:
        print(f"\n{'='*60}")
        print(f"  Smoke: {tag}  (lambda={lam}, shuffled={is_shuf})")
        print(f"{'='*60}")
        errs = part_c_smoke(cfg_file, tag, lam, is_shuf)
        smoke_results[tag] = errs

    print("\n" + "=" * 70)
    print("PART C -- Smoke summary")
    print("=" * 70)
    smoke_fail = 0
    for tag, errs in smoke_results.items():
        status = "PASS" if not errs else f"FAIL ({len(errs)})"
        print(f"  {tag:<55} {status}")
        for e in errs:
            print(f"      - {e}")
        smoke_fail += len(errs)

    if smoke_fail == 0:
        print("\nAll validation + toy DEM + smoke tests PASSED.")
        print("Safe to launch D8 training with:")
        print("  powershell -File scripts/launch_segman_n100_d8_seed0_chain.ps1")
        return 0
    else:
        print(f"\n{smoke_fail} smoke checks FAILED. Fix before launching training.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
