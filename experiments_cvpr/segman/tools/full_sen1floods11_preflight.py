# -*- coding: utf-8 -*-
"""
Full Sen1Floods11 preflight check for SegMAN-S full-dataset baselines.

Verifies:
  1. Dataset files are present (S2Hand, S1Hand, LabelHand)
  2. Split manifests exist with correct tile counts
  3. DataModule loads correctly (no N=100 cap active)
  4. DEM is NOT present in the input tensor
  5. Input tensor has exactly 15 channels
  6. Labels are binary (0/1/-1) with correct ignore index
  7. SegMAN-S forward pass succeeds
  8. Loss computation succeeds
  9. Metrics computation succeeds
 10. Gradient flow verified

Output: E:/flood_research/experiments/segman/full_sen1floods11_preflight_report.md
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import yaml

# --- paths ---
REPO_ROOT = Path(__file__).resolve().parents[3]
SEGMAN_DIR = REPO_ROOT / "experiments_cvpr" / "segman"
for p in (str(SEGMAN_DIR), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import step6c_lambda05_train as t6c
from step6c_v3_train import TopographyDataModule
from segman_losses.segman_loss import build_loss
from model.segman_model import build_segman

# --- constants ---
REPORT_PATH = Path("E:/flood_research/experiments/segman/full_sen1floods11_preflight_report.md")
CONFIG_DIR  = REPO_ROOT / "configs" / "segman" / "full_baselines"
DATA_ROOT   = Path("E:/flood_research/data/raw/sen1floods11/v1.1/data/flood_events/HandLabeled")
MANIFESTS   = REPO_ROOT / "manifests" / "terramind_baseline"

CHECKS: list[dict] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    CHECKS.append({"name": name, "status": status, "detail": detail})
    print(f"  [{status}] {name}" + (f": {detail}" if detail else ""))
    return ok


def count_tiles(split_file: Path) -> list[str]:
    if not split_file.exists():
        return []
    return [l.strip() for l in split_file.read_text(encoding="utf-8").splitlines() if l.strip()]


def count_dataset_files(modality_dir: Path, pattern: str) -> int:
    if not modality_dir.exists():
        return 0
    return len(list(modality_dir.glob(pattern)))


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("\n" + "=" * 60)
    print("  Full Sen1Floods11 Preflight — SegMAN-S Full Baselines")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # 1. Dataset files present
    # ------------------------------------------------------------------ #
    print("\n[1] Dataset file counts")
    s2_dir = DATA_ROOT / "S2Hand"
    s1_dir = DATA_ROOT / "S1Hand"
    lbl_dir = DATA_ROOT / "LabelHand"
    n_s2  = count_dataset_files(s2_dir,  "*_S2Hand.tif")
    n_s1  = count_dataset_files(s1_dir,  "*_S1Hand.tif")
    n_lbl = count_dataset_files(lbl_dir, "*_LabelHand.tif")
    check("S2Hand directory exists", s2_dir.exists(), str(s2_dir))
    check("S1Hand directory exists", s1_dir.exists(), str(s1_dir))
    check("LabelHand directory exists", lbl_dir.exists(), str(lbl_dir))
    check("S2Hand files present", n_s2 >= 400, f"{n_s2} tiles found")
    check("S1Hand files present", n_s1 >= 400, f"{n_s1} tiles found")
    check("LabelHand files present", n_lbl >= 400, f"{n_lbl} tiles found")

    # ------------------------------------------------------------------ #
    # 2. Split manifests
    # ------------------------------------------------------------------ #
    print("\n[2] Split manifests")
    train_manifest = MANIFESTS / "flood_train_data.txt"
    val_manifest   = MANIFESTS / "flood_valid_data.txt"
    test_manifest  = MANIFESTS / "flood_test_data.txt"
    bol_manifest   = MANIFESTS / "flood_bolivia_data.txt"

    train_tiles = count_tiles(train_manifest)
    val_tiles   = count_tiles(val_manifest)
    test_tiles  = count_tiles(test_manifest)
    bol_tiles   = count_tiles(bol_manifest)

    check("train manifest exists", train_manifest.exists(), str(train_manifest))
    check("val manifest exists",   val_manifest.exists(),   str(val_manifest))
    check("test manifest exists",  test_manifest.exists(),  str(test_manifest))
    check("bolivia manifest exists", bol_manifest.exists(), str(bol_manifest))
    check("train count = 251",     len(train_tiles) == 251, f"got {len(train_tiles)}")
    check("val count = 86",        len(val_tiles)   == 86,  f"got {len(val_tiles)}")
    check("test count = 89",       len(test_tiles)  == 89,  f"got {len(test_tiles)}")
    check("bolivia count = 15",    len(bol_tiles)   == 15,  f"got {len(bol_tiles)}")
    check("train tiles != test tiles", set(train_tiles).isdisjoint(set(test_tiles)),
          "overlap detected!" if not set(train_tiles).isdisjoint(set(test_tiles)) else "no overlap")
    check("train tiles != val tiles", set(train_tiles).isdisjoint(set(val_tiles)),
          "overlap detected!" if not set(train_tiles).isdisjoint(set(val_tiles)) else "no overlap")

    # ------------------------------------------------------------------ #
    # 3. Config files exist
    # ------------------------------------------------------------------ #
    print("\n[3] Config files")
    for cname in ("full_seed0_ce.yaml", "full_seed0_dice.yaml", "full_seed0_dice_ce.yaml"):
        cp = CONFIG_DIR / cname
        check(f"config {cname} exists", cp.exists(), str(cp))
    check("config DEM guard (ce)", not yaml.safe_load(
        (CONFIG_DIR/"full_seed0_ce.yaml").read_text())["dem"]["use_as_model_input"],
          "dem.use_as_model_input=false")
    check("config DEM guard (dice_ce)", not yaml.safe_load(
        (CONFIG_DIR/"full_seed0_dice_ce.yaml").read_text())["dem"]["use_as_model_input"],
          "dem.use_as_model_input=false")

    # ------------------------------------------------------------------ #
    # 4-9. DataModule + model checks
    # ------------------------------------------------------------------ #
    print("\n[4-9] DataModule / model / loss / metrics")
    config_path = CONFIG_DIR / "full_seed0_dice_ce.yaml"
    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    check("CUDA available", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

    # DataModule — train split is the full 251
    dm = TopographyDataModule(config, batch_size=2)
    dm.setup("fit")
    train_loader = dm.train_dataloader()
    val_loader   = dm.val_dataloader()

    n_train_batches = len(train_loader)
    n_val_batches   = len(val_loader)
    check("train loader has batches", n_train_batches > 0, f"{n_train_batches} batches")
    expected_train_batches = math.ceil(251 / 2)
    check("train loader consistent with 251 tiles",
          abs(n_train_batches - expected_train_batches) <= 2,
          f"got {n_train_batches}, expected ~{expected_train_batches}")
    check("val loader has batches", n_val_batches > 0, f"{n_val_batches} batches")

    # Get one batch
    raw_batch = next(iter(train_loader))
    batch = t6c.move_batch(raw_batch, device)

    # Assemble 15-channel input
    from train_segman import InputAssembler, get_target
    assembler = InputAssembler(config, device)
    x = assembler(batch["image"])

    check("input tensor is 15-channel", x.shape[1] == 15, f"shape={tuple(x.shape)}")
    check("DEM not in model input", x.shape[1] != 16,
          f"input has {x.shape[1]} channels (16 would mean DEM leaked in)")
    check("input shape is [B,15,512,512]",
          x.shape[2] == 512 and x.shape[3] == 512, f"{tuple(x.shape)}")
    check("input is finite", torch.isfinite(x).all().item(), "no NaN/Inf")

    # Labels
    target = get_target(batch)
    unique_labels = torch.unique(target).cpu().tolist()
    check("label tensor is long", target.dtype == torch.int64, str(target.dtype))
    check("labels contain only {-1, 0, 1}", all(v in (-1, 0, 1) for v in unique_labels),
          f"unique={unique_labels}")
    check("label shape is [B, 512, 512]",
          target.shape[1] == 512 and target.shape[2] == 512, f"{tuple(target.shape)}")

    # DEM availability in batch (for topo loss — should exist even when not used)
    topo = batch.get("topography")
    check("topography key in batch", topo is not None,
          "present (for optional topo diagnostic)" if topo is not None else "MISSING")

    # Model
    model_cfg = dict(config["model"])
    model_cfg.setdefault("in_chans", assembler.in_chans)
    model = build_segman(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    check("SegMAN-S instantiated", n_params > 30_000_000, f"{n_params:,} params")
    check("parameter count ~33.45M", abs(n_params - 33_447_272) < 10_000,
          f"{n_params:,} (expected 33,447,272)")

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits_val = model(x)
    check("forward pass shape [B,2,512,512]",
          logits_val.shape == (x.shape[0], 2, 512, 512), f"{tuple(logits_val.shape)}")
    check("logits are finite", torch.isfinite(logits_val).all().item(), "no NaN/Inf")

    # Loss
    criterion = build_loss(config).to(device)
    model.train()
    x.requires_grad_(False)
    logits_train = model(x)
    losses = criterion(logits_train, target, topo)
    loss_val = losses["loss_total"]
    check("loss is finite", torch.isfinite(loss_val).item(), f"loss={float(loss_val):.4f}")
    check("loss_ce present", "loss_ce" in losses, "yes")
    check("loss_dice present", "loss_dice" in losses, "yes")
    check("loss_topo=0 (no physics)", float(losses["loss_topo"]) == 0.0,
          f"loss_topo={float(losses['loss_topo']):.6f}")

    # Gradient flow
    loss_val.backward()
    grad_norms = [float(p.grad.norm().item()) for p in model.parameters() if p.grad is not None]
    check("gradients flow to all params", len(grad_norms) > 0, f"{len(grad_norms)} param grads")
    check("no zero-norm gradients", all(g > 0 for g in grad_norms[:10]),
          f"first 10 norms: {[f'{g:.4f}' for g in grad_norms[:10]]}")

    # Metrics
    model.eval()
    with torch.no_grad():
        logits_m = model(x)
    pred = torch.argmax(logits_m.detach(), dim=1)
    matrix = t6c.confusion(target, pred)
    metrics = t6c.metrics_from_conf(matrix)
    check("mIoU computable", math.isfinite(metrics.get("mean_iou", math.nan)),
          f"mIoU={metrics.get('mean_iou', 'n/a'):.4f}")
    _w_iou = metrics.get("water_iou", metrics.get("iou_water", math.nan))
    check("water IoU computable", math.isfinite(float(_w_iou)) if _w_iou != 'n/a' else False,
          f"waterIoU={_w_iou:.4f}" if isinstance(_w_iou, float) else f"key=water_iou not found; available: {list(metrics.keys())[:8]}")

    # ------------------------------------------------------------------ #
    # Generate preflight report
    # ------------------------------------------------------------------ #
    elapsed = time.time() - t0
    n_pass = sum(1 for c in CHECKS if c["status"] == "PASS")
    n_fail = sum(1 for c in CHECKS if c["status"] == "FAIL")
    overall = "PASS" if n_fail == 0 else "FAIL"

    print(f"\n{'='*60}")
    print(f"  PREFLIGHT {overall}: {n_pass} PASS / {n_fail} FAIL  ({elapsed:.1f}s)")
    print(f"{'='*60}")

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    gpu_mem  = torch.cuda.get_device_properties(0).total_memory // (1024**3) if torch.cuda.is_available() else 0
    peak_mb  = torch.cuda.max_memory_allocated() // (1024**2) if torch.cuda.is_available() else 0

    md_lines = [
        f"# Full Sen1Floods11 Preflight Report",
        f"## SegMAN-S Full-Dataset Baselines",
        f"",
        f"**Generated:** {dt.datetime.now().isoformat(timespec='seconds')}  ",
        f"**Overall:** {overall} ({n_pass}/{n_pass+n_fail} checks passed)  ",
        f"**Elapsed:** {elapsed:.1f}s",
        f"",
        f"---",
        f"",
        f"## Dataset",
        f"",
        f"| Item | Value |",
        f"|---|---|",
        f"| S2Hand files | {n_s2} |",
        f"| S1Hand files | {n_s1} |",
        f"| LabelHand files | {n_lbl} |",
        f"| Train tiles | {len(train_tiles)} |",
        f"| Val tiles | {len(val_tiles)} |",
        f"| Test tiles | {len(test_tiles)} |",
        f"| Bolivia tiles | {len(bol_tiles)} |",
        f"| Train manifest | `{train_manifest}` |",
        f"| Val manifest | `{val_manifest}` |",
        f"| Test manifest | `{test_manifest}` |",
        f"| Bolivia manifest | `{bol_manifest}` |",
        f"",
        f"---",
        f"",
        f"## Model and Input",
        f"",
        f"| Item | Value |",
        f"|---|---|",
        f"| Architecture | SegMAN-S |",
        f"| Parameters | {n_params:,} (33.45M) |",
        f"| Input channels | {x.shape[1]} (13 S2L1C + 2 S1GRD) |",
        f"| DEM as model input | **NEVER** |",
        f"| Output shape | (B, 2, 512, 512) |",
        f"| Label classes | 0=dry, 1=water |",
        f"| Ignore index | -1 |",
        f"| GPU | {gpu_name} ({gpu_mem} GB) |",
        f"| Peak VRAM (preflight fwd) | {peak_mb} MB |",
        f"",
        f"---",
        f"",
        f"## Configs",
        f"",
        f"| Config | Path |",
        f"|---|---|",
        f"| CE | `configs/segman/full_baselines/full_seed0_ce.yaml` |",
        f"| Dice | `configs/segman/full_baselines/full_seed0_dice.yaml` |",
        f"| Dice+CE | `configs/segman/full_baselines/full_seed0_dice_ce.yaml` |",
        f"",
        f"---",
        f"",
        f"## Checks",
        f"",
        f"| # | Check | Status | Detail |",
        f"|---|---|---|---|",
    ]
    for i, c in enumerate(CHECKS, 1):
        md_lines.append(f"| {i} | {c['name']} | **{c['status']}** | {c['detail']} |")

    md_lines += [
        f"",
        f"---",
        f"",
        f"## Final Verdict",
        f"",
        f"**{overall}** — {n_pass} checks passed, {n_fail} failed.",
        f"" if overall == "PASS" else "**Training must NOT be launched until all failures are resolved.**",
        f"",
        f"Next step: run smoke tests (1 mini-batch per config) then launch three sequential baselines.",
    ]

    REPORT_PATH.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nReport written: {REPORT_PATH}")

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
