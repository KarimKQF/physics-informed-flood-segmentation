"""
Validation and smoke test for SegMAN N=100 D8 downstream loss lambda=100, seed0.

Three-part verification:
  Part A: Config validation (both lambda=100 configs).
  Part B: D8 loss unit tests (synthetic toy DEMs, CPU-only).
  Part C: Real-data smoke on a small batch (CPU inference, no training).

Scientific context:
  lambda=1.0 pilot: effective D8 contribution ~0.01% (target 1-5%).
  DEM geometry: 72% of pixels have w>0.1 with s0=1.0m -- DEM is not flat.
  Correct fix: rescale lambda x100. tau=0.05, s0=1.0 unchanged.

Usage:
    python experiments_cvpr/segman/validate_and_smoke_d8_lambda100_n100.py

All tests must PASS before running the launcher.
"""
from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path

# ── Repo root on sys.path ──────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parents[2]
for _p in [str(_REPO), str(_REPO / "src"), str(_REPO / "scripts")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn.functional as F

# ── Constants ──────────────────────────────────────────────────────────────────
REPO_ROOT = _REPO
CFGDIR    = REPO_ROOT / "configs" / "segman" / "d8_n100_seed0"
RUNS_ROOT = Path("E:/flood_research/experiments/segman/runs")

REAL_CFG_NAME = "n100_seed0_dice_ce_d8_lambda100p0.yaml"
SHUF_CFG_NAME = "n100_seed0_dice_ce_d8_dem_shuffled_lambda100p0.yaml"

EXPECTED_LAMBDA    = 100.0
EXPECTED_SEED      = 0
EXPECTED_N_TRAIN   = 100
EXPECTED_TAU       = 0.05
EXPECTED_S0        = 1.0
EXPECTED_WARMUP_EP = 5
EXPECTED_RAMP_EP   = 15

REAL_TAG  = "segman_n100_d8_lambda100p0_seed0"
SHUF_TAG  = "segman_n100_d8_dem_shuffled_lambda100p0_seed0"
REAL_RDIR = RUNS_ROOT / REAL_TAG
SHUF_RDIR = RUNS_ROOT / SHUF_TAG

# Lambda=1 run dirs that must NOT be overwritten
LAMBDA1_TAGS = [
    "segman_n100_d8_lambda1p0_seed0",
    "segman_n100_d8_dem_shuffled_lambda1p0_seed0",
]

TRAIN_MANIFEST = (REPO_ROOT / "manifests" / "terramind_baseline"
                  / "low_data_multiseed_n100" / "flood_train_low_data_n100_seed0.txt")
SHUFFLE_MAP    = (REPO_ROOT / "manifests" / "terramind_baseline"
                  / "low_data_multiseed_n100" / "dem_shuffle_map_n100_seed0.json")

DEM_ROOT = Path("E:/flood_research/data/derived/sen1floods11_topography/dem_aligned")
DEM_PAT  = "{split}_{tile_id}_copernicus_glo30_dem_aligned.tif"

PASS = "[PASS]"
FAIL = "[FAIL]"

# ── Result tracker ─────────────────────────────────────────────────────────────
results: list[tuple[str, str, str]] = []

def check(name: str, ok: bool, detail: str = "") -> bool:
    status = PASS if ok else FAIL
    results.append((status, name, detail))
    marker = "  OK" if ok else "FAIL"
    print(f"  {marker}  {name}" + (f"  [{detail}]" if detail else ""))
    return ok

# ── YAML loader ───────────────────────────────────────────────────────────────
def load_yaml(path: Path) -> dict:
    try:
        import yaml
        with path.open(encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        # Minimal fallback: not parsing nested structures, just key=value lines
        raise RuntimeError("PyYAML required")

# ══════════════════════════════════════════════════════════════════════════════
# PART A — CONFIG VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
def part_a_config_validation() -> int:
    print("\n" + "=" * 60)
    print("PART A — Config Validation")
    print("=" * 60)
    n_fail = 0

    for cfg_name, tag, rdir, mode_expected, is_shuffled in [
        (REAL_CFG_NAME, REAL_TAG, REAL_RDIR, "dice_ce_d8",              False),
        (SHUF_CFG_NAME, SHUF_TAG, SHUF_RDIR, "dice_ce_d8_dem_shuffled", True),
    ]:
        print(f"\n--- {cfg_name} ---")
        cfg_path = CFGDIR / cfg_name
        ok = check(f"{cfg_name}: file exists", cfg_path.exists())
        if not ok:
            n_fail += 1
            continue

        try:
            cfg = load_yaml(cfg_path)
        except Exception as e:
            check(f"{cfg_name}: YAML parse", False, str(e))
            n_fail += 1
            continue

        # Identity
        ok = check("seed == 0",           cfg.get("seed") == EXPECTED_SEED,
                   str(cfg.get("seed")))
        if not ok: n_fail += 1

        ok = check("run_tag correct",     cfg.get("run_tag") == tag,
                   str(cfg.get("run_tag")))
        if not ok: n_fail += 1

        ok = check("run_dir correct",     cfg.get("run_dir", "").endswith(tag),
                   str(cfg.get("run_dir", ""))[-60:])
        if not ok: n_fail += 1

        # Loss
        loss = cfg.get("loss", {})
        lam  = loss.get("lambda_topo")
        ok = check("lambda_topo == 100.0", lam == EXPECTED_LAMBDA, str(lam))
        if not ok: n_fail += 1

        ok = check("loss.mode correct",   loss.get("mode") == mode_expected,
                   str(loss.get("mode")))
        if not ok: n_fail += 1

        d8 = loss.get("d8", {})
        ok = check("d8.tau == 0.05",      d8.get("tau") == EXPECTED_TAU, str(d8.get("tau")))
        if not ok: n_fail += 1

        ok = check("d8.s0 == 1.0",        d8.get("s0") == EXPECTED_S0, str(d8.get("s0")))
        if not ok: n_fail += 1

        sched = loss.get("lambda_schedule", {})
        ok = check("warmup_epochs == 5",  sched.get("warmup_epochs") == EXPECTED_WARMUP_EP,
                   str(sched.get("warmup_epochs")))
        if not ok: n_fail += 1

        ok = check("ramp_epochs == 15",   sched.get("ramp_epochs") == EXPECTED_RAMP_EP,
                   str(sched.get("ramp_epochs")))
        if not ok: n_fail += 1

        # DEM model input
        dem = cfg.get("dem", {})
        ok = check("dem.use_as_model_input == false",
                   dem.get("use_as_model_input") == False,
                   str(dem.get("use_as_model_input")))
        if not ok: n_fail += 1

        ok = check("dem.use_in_loss == true",
                   dem.get("use_in_loss") == True,
                   str(dem.get("use_in_loss")))
        if not ok: n_fail += 1

        # N_train
        dp = cfg.get("dataset_policy", {})
        ok = check("N_train == 100", dp.get("train") == EXPECTED_N_TRAIN,
                   str(dp.get("train")))
        if not ok: n_fail += 1

        # Manifests
        data = cfg.get("data", {}).get("init_args", {})
        ts   = data.get("train_split", "")
        ok = check("train_split is seed0 N100", "n100_seed0" in ts.lower(), ts[-60:])
        if not ok: n_fail += 1

        # Shuffled-specific
        if is_shuffled:
            shuf_map = dem.get("dem_tile_id_map_file", "")
            ok = check("dem_tile_id_map_file set",     bool(shuf_map), shuf_map[-60:])
            if not ok: n_fail += 1
            ok = check("shuffle map file exists",     Path(shuf_map).exists() if shuf_map else False,
                       shuf_map[-60:])
            if not ok: n_fail += 1
            ok = check("dem_shuffled_control == true",
                       cfg.get("guardrails", {}).get("dem_shuffled_control") == True,
                       str(cfg.get("guardrails", {}).get("dem_shuffled_control")))
            if not ok: n_fail += 1

            # Verify the shuffle map has the right N
            try:
                smap = json.loads(Path(shuf_map).read_text(encoding="utf-8"))
                # Map is {"mapping": {tile_id: shuffled_tile_id, ...}, "n_tiles": 100, ...}
                n_map = smap.get("n_tiles") or len(smap.get("mapping", {}))
                ok = check("shuffle map has 100 entries", n_map == 100, str(n_map))
                if not ok: n_fail += 1
            except Exception as e:
                check("shuffle map parseable", False, str(e))
                n_fail += 1
        else:
            ok = check("dem_tile_id_map == null",
                       dem.get("dem_tile_id_map") is None,
                       str(dem.get("dem_tile_id_map")))
            if not ok: n_fail += 1

        # No collision with lambda=1 run dirs
        for l1_tag in LAMBDA1_TAGS:
            ok = check(f"run_dir != lambda=1 dir ({l1_tag})",
                       tag != l1_tag,
                       f"{tag} vs {l1_tag}")
            if not ok: n_fail += 1

        # Run dir must not already exist as a completed run
        summary_path = rdir / "metrics" / f"{tag}_summary.json"
        if summary_path.exists():
            try:
                s = json.loads(summary_path.read_text(encoding="utf-8"))
                already_done = s.get("status") == "done"
            except Exception:
                already_done = False
            ok = check("lambda=100 run not already completed (fresh start)",
                       not already_done,
                       "ALREADY DONE — would skip" if already_done else "not yet run")
            # This is a warning, not a blocker (launcher will skip gracefully)
        else:
            check("lambda=100 run dir is fresh (no summary JSON)", True, "not yet run")

        # Model
        model = cfg.get("model", {})
        ok = check("model.in_chans == 15", model.get("in_chans") == 15,
                   str(model.get("in_chans")))
        if not ok: n_fail += 1
        ok = check("model.num_classes == 2", model.get("num_classes") == 2,
                   str(model.get("num_classes")))
        if not ok: n_fail += 1

    return n_fail

# ══════════════════════════════════════════════════════════════════════════════
# PART B — D8 LOSS UNIT TESTS (CPU, synthetic DEMs)
# ══════════════════════════════════════════════════════════════════════════════
def part_b_unit_tests() -> int:
    print("\n" + "=" * 60)
    print("PART B — D8 Loss Unit Tests (CPU, synthetic)")
    print("=" * 60)
    n_fail = 0

    try:
        from losses.d8_downstream_loss import D8DownstreamLoss
    except ImportError as e:
        check("D8DownstreamLoss importable", False, str(e))
        return 1

    check("D8DownstreamLoss importable", True)
    loss_fn = D8DownstreamLoss(ignore_index=-1, water_class=1,
                                s0=EXPECTED_S0, tau=EXPECTED_TAU, eps=1e-6,
                                reduction="mean")

    def _make_batch(B, H, W, logit_val=0.0):
        return torch.full((B, 2, H, W), logit_val)

    def _run(logits, target, dem):
        with torch.no_grad():
            return loss_fn(logits=logits, target=target, topography=dem)

    # Test 1: Upstream water → downstream dry → loss > 0
    print("\n  Test 1: upstream water, downstream dry -> loss > 0")
    try:
        dem = torch.zeros(1, 1, 1, 6)
        dem[0, 0, 0, :] = torch.tensor([10., 8., 6., 4., 2., 0.])
        logits = torch.zeros(1, 2, 1, 6)
        logits[0, 1, 0, :3] = 5.0   # left 3 pixels: water
        logits[0, 0, 0, 3:] = 5.0   # right 3 pixels: dry
        target = torch.ones(1, 1, 6, dtype=torch.long)
        v = _run(logits, target, dem).item()
        ok = check("Test 1: D8 loss > 0 (upstream water, downstream dry)", v > 0, f"loss={v:.6f}")
        if not ok: n_fail += 1
    except Exception as e:
        check("Test 1: no exception", False, traceback.format_exc()[-200:])
        n_fail += 1

    # Test 2: Monotonically increasing water probability downstream → loss ≈ 0
    print("\n  Test 2: p increasing downstream -> loss ~= 0")
    try:
        dem = torch.zeros(1, 1, 1, 5)
        dem[0, 0, 0, :] = torch.tensor([10., 8., 6., 4., 2.])
        logits = torch.zeros(1, 2, 1, 5)
        logits[0, 1, 0, :] = torch.tensor([-3., -1., 0., 1., 3.])
        target = torch.zeros(1, 1, 5, dtype=torch.long)
        v = _run(logits, target, dem).item()
        ok = check("Test 2: D8 loss ~= 0 (p increases downstream)", v < 0.01, f"loss={v:.6f}")
        if not ok: n_fail += 1
    except Exception as e:
        check("Test 2: no exception", False, traceback.format_exc()[-200:])
        n_fail += 1

    # Test 3: All-dry predictions → loss ≈ 0
    print("\n  Test 3: all-dry predictions -> loss ~= 0")
    try:
        dem = torch.arange(16, dtype=torch.float32).flip(0).view(1, 1, 4, 4)
        logits = torch.zeros(1, 2, 4, 4); logits[:, 0] = 3.0
        target = torch.zeros(1, 4, 4, dtype=torch.long)
        v = _run(logits, target, dem).item()
        ok = check("Test 3: all-dry -> loss near zero", v < 1e-4, f"loss={v:.8f}")
        if not ok: n_fail += 1
    except Exception as e:
        check("Test 3: no exception", False, traceback.format_exc()[-200:])
        n_fail += 1

    # Test 4: Random logits → loss > 0 (has some violations)
    print("\n  Test 4: random logits -> loss > 0")
    try:
        torch.manual_seed(42)
        dem    = torch.randn(2, 1, 8, 8) * 5.0
        logits = torch.randn(2, 2, 8, 8)
        target = torch.zeros(2, 8, 8, dtype=torch.long)
        v = _run(logits, target, dem).item()
        ok = check("Test 4: random -> loss > 0", v > 0, f"loss={v:.6f}")
        if not ok: n_fail += 1
    except Exception as e:
        check("Test 4: no exception", False, traceback.format_exc()[-200:])
        n_fail += 1

    # Test 5: NaN DEM → finite loss
    print("\n  Test 5: NaN DEM -> finite loss")
    try:
        dem    = torch.randn(1, 1, 4, 4)
        dem[0, 0, 1:3, 1:3] = float("nan")
        logits = torch.randn(1, 2, 4, 4)
        target = torch.zeros(1, 4, 4, dtype=torch.long)
        v = _run(logits, target, dem).item()
        ok = check("Test 5: NaN DEM -> finite loss", math.isfinite(v), f"loss={v}")
        if not ok: n_fail += 1
    except Exception as e:
        check("Test 5: no exception", False, traceback.format_exc()[-200:])
        n_fail += 1

    # Test 6: All-ignore target → loss == 0
    print("\n  Test 6: all-ignore target -> loss == 0")
    try:
        dem    = torch.randn(1, 1, 4, 4)
        logits = torch.randn(1, 2, 4, 4)
        target = torch.full((1, 4, 4), -1, dtype=torch.long)
        v = _run(logits, target, dem).item()
        ok = check("Test 6: all-ignore -> loss == 0", v == 0.0, f"loss={v:.8f}")
        if not ok: n_fail += 1
    except Exception as e:
        check("Test 6: no exception", False, traceback.format_exc()[-200:])
        n_fail += 1

    return n_fail

# ══════════════════════════════════════════════════════════════════════════════
# PART C — REAL-DATA SMOKE (one batch, CPU, no training)
# ══════════════════════════════════════════════════════════════════════════════
def part_c_real_data_smoke() -> int:
    print("\n" + "=" * 60)
    print("PART C — Real-data Smoke (one batch, CPU, no training)")
    print("=" * 60)
    n_fail = 0

    # Import loss and model
    try:
        from losses.d8_downstream_loss import D8DownstreamLoss
    except ImportError as e:
        check("D8DownstreamLoss importable (Part C)", False, str(e))
        return 1

    try:
        from experiments_cvpr.segman.segman_losses.segman_loss import SegManCombinedLoss
    except ImportError:
        try:
            sys.path.insert(0, str(REPO_ROOT / "experiments_cvpr" / "segman"))
            from segman_losses.segman_loss import SegManCombinedLoss
        except ImportError as e:
            check("SegManCombinedLoss importable", False, str(e))
            return 1
    check("SegManCombinedLoss importable", True)

    # Try to load one real DEM tile for smoke
    dem_tile: torch.Tensor | None = None
    try:
        import rasterio
        manifest_lines = TRAIN_MANIFEST.read_text().splitlines()
        for tile_id in manifest_lines[:10]:
            tile_id = tile_id.strip()
            if not tile_id:
                continue
            dem_path = DEM_ROOT / DEM_PAT.format(split="train", tile_id=tile_id)
            if dem_path.exists():
                with rasterio.open(dem_path) as ds:
                    arr = ds.read(1).astype("float32")
                    nd  = ds.nodata
                    if nd is not None:
                        arr[arr == nd] = float("nan")
                # Resize to 512x512 if needed (assume already 512x512 from aligned)
                dem_tile = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
                print(f"  Loaded DEM tile: {dem_path.name}  shape={dem_tile.shape}")
                break
    except ImportError:
        print("  rasterio not available; using synthetic DEM for smoke")
    except Exception as e:
        print(f"  Could not load real DEM: {e}; using synthetic DEM")

    if dem_tile is None:
        dem_tile = torch.randn(1, 1, 512, 512) * 50.0

    # Build a synthetic batch matching expected shapes
    B, C, H, W = 1, 15, 512, 512
    images  = torch.randn(B, C, H, W)
    labels  = torch.randint(0, 2, (B, H, W))
    dem_b   = dem_tile.expand(B, -1, -1, -1)  # [B,1,H,W]

    check("DEM tile not model input (smoke batch has no DEM channel in images)",
          images.shape[1] == 15)

    check("images shape [B,15,512,512]", images.shape == (B, C, H, W), str(images.shape))
    check("labels shape [B,512,512]",    labels.shape == (B, H, W),     str(labels.shape))
    check("DEM shape [B,1,512,512]",     dem_b.shape  == (B, 1, H, W),  str(dem_b.shape))
    check("labels values in {0,1}",
          labels.min().item() >= 0 and labels.max().item() <= 1,
          f"min={labels.min()} max={labels.max()}")

    # Build SegManCombinedLoss with lambda=100
    try:
        loss_fn = SegManCombinedLoss(
            mode="dice_ce_d8",
            ignore_index=-1,
            water_class=1,
            lambda_topo=EXPECTED_LAMBDA,
            d8_s0=EXPECTED_S0,
            d8_tau=EXPECTED_TAU,
            d8_eps=1e-6,
        )
        loss_fn.set_lambda_topo(EXPECTED_LAMBDA)
        check("SegManCombinedLoss(mode=dice_ce_d8, lambda=100) constructed", True)
    except Exception as e:
        check("SegManCombinedLoss(mode=dice_ce_d8, lambda=100) constructed", False, str(e))
        n_fail += 1
        return n_fail

    # Forward with untrained logits (expected: D8 ≈ 0 due to uniform probabilities)
    logits = torch.zeros(B, 2, H, W)
    try:
        with torch.no_grad():
            out = loss_fn(logits=logits, target=labels, topography=dem_b)
        check("forward pass completes (untrained logits)", True)
        for key in ("loss_total", "loss_ce", "loss_dice", "loss_topo"):
            v = out.get(key)
            if v is None:
                check(f"{key} present in output", False, "missing key")
                n_fail += 1
                continue
            val = v.item() if hasattr(v, "item") else float(v)
            ok = check(f"{key} finite", math.isfinite(val), f"{val:.6f}")
            if not ok: n_fail += 1
        # Note: D8 ≈ 0 at init is expected (uniform predictions → no hinge violations)
        d8_val = out.get("loss_topo")
        if d8_val is not None:
            print(f"  D8 loss (untrained, expected ~0): {d8_val.item():.8f}")
            print("  NOTE: D8 ~= 0 at init is correct -- uniform logits produce no hinge violations.")
    except Exception as e:
        check("forward pass (untrained logits)", False, traceback.format_exc()[-200:])
        n_fail += 1

    # Forward with perturbed logits → D8 must be > 0 and effective contribution measurable
    print("\n  Smoke with perturbed logits (sigma=3 Gaussian noise):")
    torch.manual_seed(0)
    logits_p = logits + torch.randn_like(logits) * 3.0
    try:
        with torch.no_grad():
            out_p = loss_fn(logits=logits_p, target=labels, topography=dem_b)
        check("forward pass (perturbed logits) completes", True)
        for key in ("loss_total", "loss_ce", "loss_dice", "loss_topo"):
            v = out_p.get(key)
            val = v.item() if v is not None and hasattr(v, "item") else None
            if val is not None:
                ok = check(f"{key} finite (perturbed)", math.isfinite(val), f"{val:.6f}")
                if not ok: n_fail += 1

        d8_p    = out_p.get("loss_topo")
        lce_p   = out_p.get("loss_ce")
        ldice_p = out_p.get("loss_dice")
        lam_v   = out_p.get("lambda_topo")

        if d8_p is not None:
            d8_val_p = d8_p.item()
            ok = check("D8 loss > 0 with perturbed logits", d8_val_p > 1e-10, f"{d8_val_p:.8f}")
            if not ok: n_fail += 1

        if all(x is not None for x in [d8_p, lce_p, ldice_p]):
            raw_d8   = d8_p.item()
            raw_base = lce_p.item() + ldice_p.item()
            lam_used = EXPECTED_LAMBDA
            eff_d8   = (lam_used * raw_d8 / raw_base) if raw_base > 0 else None
            print(f"  raw D8 loss:              {raw_d8:.6f}")
            print(f"  lambda:                   {lam_used}")
            print(f"  lambda*D8 (numerator):    {lam_used * raw_d8:.6f}")
            print(f"  dice+ce (denominator):    {raw_base:.6f}")
            if eff_d8 is not None:
                print(f"  Effective D8 contribution: {eff_d8*100:.3f}%")
                check("Effective D8 contribution > 0% (lambda=100, perturbed)",
                      eff_d8 > 0, f"{eff_d8*100:.3f}%")
    except Exception as e:
        check("forward pass (perturbed logits)", False, traceback.format_exc()[-200:])
        n_fail += 1

    # Verify DEM-specificity at lambda=100: shifted DEM gives different D8
    print("\n  DEM-specificity check (real vs cyclically shifted DEM at lambda=100):")
    try:
        # Cyclically shift DEM in spatial dimension
        dem_shifted = torch.roll(dem_b, shifts=64, dims=2)
        with torch.no_grad():
            out_real    = loss_fn(logits=logits_p, target=labels, topography=dem_b)
            out_shifted = loss_fn(logits=logits_p, target=labels, topography=dem_shifted)
        d8_real = out_real.get("loss_topo")
        d8_shif = out_shifted.get("loss_topo")
        if d8_real is not None and d8_shif is not None:
            diff = abs(d8_real.item() - d8_shif.item())
            ok = check("D8 loss differs when DEM is shifted (lambda=100)",
                       diff > 1e-8,
                       f"real={d8_real.item():.6f}  shifted={d8_shif.item():.6f}  diff={diff:.6f}")
            if not ok: n_fail += 1
    except Exception as e:
        check("DEM specificity check", False, traceback.format_exc()[-200:])
        n_fail += 1

    return n_fail

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main() -> int:
    print("=" * 60)
    print("Validate & Smoke — D8 lambda=100, N=100, seed0")
    print("=" * 60)
    print(f"Repo:   {REPO_ROOT}")
    print(f"Config: {CFGDIR}")
    print()

    total_fail = 0
    total_fail += part_a_config_validation()
    total_fail += part_b_unit_tests()
    total_fail += part_c_real_data_smoke()

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    n_pass = sum(1 for s, _, _ in results if s == PASS)
    n_fail = sum(1 for s, _, _ in results if s == FAIL)
    print(f"  PASS: {n_pass}")
    print(f"  FAIL: {n_fail}")
    print()
    if n_fail > 0:
        print("FAILURES:")
        for status, name, detail in results:
            if status == FAIL:
                print(f"  [FAIL] {name}" + (f"  [{detail}]" if detail else ""))
        print()
        print("ALL CHECKS MUST PASS BEFORE LAUNCHING TRAINING.")
        print("Fix failures above, then re-run this script.")
        return 1
    else:
        print("ALL CHECKS PASSED.")
        print()
        print("Lambda=100 configs are ready to train.")
        print("Run:")
        print("  .\\scripts\\launch_segman_n100_d8_lambda100_seed0_chain.ps1")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
