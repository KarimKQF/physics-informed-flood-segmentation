"""
STEP 6C v2 — D4 A/B microtest (NO full training; controlled 3-epoch isolation).

Goal: determine whether the manual-D4 / RNG path of the v2 runner is the cause of the
no-water (gradient-dead all-background) collapse, by comparing it against the STEP
5S-A-style albumentations.D4 path under PURE DICE (lambda_topo = 0).

Everything else is held identical between the two arms:
  - original TerraMind pretrained init (same build_task, same seed -> same weights)
  - same train/valid split, same batch_size=2, grad_accum=4, FP32, BN-eval policy
  - same AdamW lr=2e-5 wd=1e-4, NO scheduler step (LR fixed) in both arms
  - same exact Dice loss (CombinedDicePhysicsLoss with lambda_topo=0 == smp DiceLoss)
  - same metric code

Only difference:
  A "manual"        : dataloader=ToTensorV2 only; D4 applied manually to image+mask+DEM
                      in the loop (the current v2 path).
  B "albumentations": dataloader=[albumentations.D4, ToTensorV2] (the 5S-A path);
                      no manual D4; DEM loaded raw (unused at lambda=0).

DEM is NEVER a model input. Raw data untouched. Existing runs untouched.

Output JSON:
  E:/flood_research/experiments/terramind_baseline/runs/step6c_v2_ab_microtest/metrics/d4_ab_microtest.json
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for _p in (str(SRC_ROOT), str(SCRIPTS_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import yaml  # noqa: E402
import step6c_lambda05_train as t6c  # noqa: E402
from losses.combined_loss import CombinedDicePhysicsLoss  # noqa: E402

CONFIG = REPO_ROOT / "configs" / "step6c_v2_terramind_l_upernet_dice_topographic_lambda01.yaml"
OUT_JSON = Path("E:/flood_research/experiments/terramind_baseline/runs/step6c_v2_ab_microtest/metrics/d4_ab_microtest.json")
EPOCHS = 3
SEED = 42


def jsafe(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): jsafe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsafe(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        v = float(v)
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def set_all_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def eval_dice(task, criterion, loader, config, device) -> dict[str, Any]:
    task.eval()
    t6c.set_bn_eval(task)
    matrix = [[0, 0], [0, 0]]
    dice_sum = 0.0
    batches = 0
    pw_sum = 0.0
    pw_count = 0
    with torch.no_grad():
        for raw_batch in loader:
            dem_cpu = t6c.load_dem_batch(config, raw_batch, split="valid")
            batch = t6c.move_batch(raw_batch, device)
            dem = dem_cpu.to(device)
            losses, logits, target = t6c.compute_loss(task, criterion, batch, dem)
            dice_sum += float(losses["loss_dice"].detach().cpu())
            batches += 1
            pred = torch.argmax(logits.detach(), dim=1)
            matrix = t6c.add_conf(matrix, t6c.confusion(target, pred))
            pw = torch.softmax(logits, dim=1)[:, 1]
            valid = target != -1
            pw_sum += float(pw[valid].sum().detach().cpu())
            pw_count += int(valid.sum().item())
    m = t6c.metrics_from_conf(matrix)
    m["loss_dice"] = dice_sum / batches if batches else math.nan
    water_pred = int(m["tp"]) + int(m["fp"])
    total_valid = int(m["tn"]) + int(m["fp"]) + int(m["fn"]) + int(m["tp"])
    m["val_water_pred_pixels"] = water_pred
    m["val_total_valid_pixels"] = total_valid
    m["pred_water_fraction"] = (water_pred / total_valid) if total_valid else math.nan
    m["mean_p_water"] = (pw_sum / pw_count) if pw_count else math.nan
    return m


def run_path(label: str, d4_mode: str, device: torch.device) -> dict[str, Any]:
    set_all_seeds(SEED)
    with CONFIG.open("r", encoding="utf-8-sig") as fh:
        config = yaml.safe_load(fh)

    bs = int(config["trainer"]["batch_size"])
    grad_accum = int(config["trainer"]["gradient_accumulation_steps"])

    if d4_mode == "manual":
        dm = t6c.build_datamodule(config, batch_size=bs, train_aug=True)   # ToTensorV2 only -> manual D4
    elif d4_mode == "albumentations":
        dm = t6c.build_datamodule(config, batch_size=bs, train_aug=False)  # [A.D4(), ToTensorV2()]
    else:
        raise ValueError(d4_mode)
    dm.setup("fit")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    task = t6c.build_task(config).to(device)
    t6c.set_bn_eval(task)
    criterion = CombinedDicePhysicsLoss(lambda_topo=0.0, ignore_index=-1, water_class=1).to(device)
    params = [p for p in task.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(config["optimizer"]["init_args"]["lr"]),
                                  weight_decay=float(config["optimizer"]["init_args"]["weight_decay"]))
    d4_rng = random.Random(SEED)

    epochs_out: list[dict[str, Any]] = []
    collapsed = False
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        task.train()
        t6c.set_bn_eval(task)
        optimizer.zero_grad(set_to_none=True)
        matrix = [[0, 0], [0, 0]]
        dice_sum = 0.0
        batches = 0
        epoch_max_grad_norm = 0.0

        for batch_idx, raw_batch in enumerate(train_loader, start=1):
            if d4_mode == "manual":
                dem_cpu = t6c.load_dem_batch(config, raw_batch, split="train")
                t6c.apply_d4_to_batch(raw_batch, dem_cpu, d4_rng)
            else:
                dem_cpu = t6c.load_dem_batch(config, raw_batch, split="train")  # raw, unused at lambda=0
            batch = t6c.move_batch(raw_batch, device)
            dem = dem_cpu.to(device)
            losses, logits, target = t6c.compute_loss(task, criterion, batch, dem)
            loss_total = losses["loss_total"]
            (loss_total / grad_accum).backward()
            if batch_idx % grad_accum == 0 or batch_idx == len(train_loader):
                gsq = 0.0
                for p in params:
                    if p.grad is not None:
                        gsq += float(p.grad.detach().pow(2).sum().item())
                epoch_max_grad_norm = max(epoch_max_grad_norm, math.sqrt(gsq))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                t6c.set_bn_eval(task)
            pred = torch.argmax(logits.detach(), dim=1)
            matrix = t6c.add_conf(matrix, t6c.confusion(target, pred))
            dice_sum += float(losses["loss_dice"].detach().cpu())
            batches += 1

        train_m = t6c.metrics_from_conf(matrix)
        val_m = eval_dice(task, criterion, val_loader, config, device)
        grads_dead = epoch_max_grad_norm < 1e-6
        row = {
            "epoch": epoch,
            "train_loss_dice": dice_sum / batches if batches else math.nan,
            "train_miou": train_m["mean_iou"],
            "val_loss_dice": val_m["loss_dice"],
            "val_miou": val_m["mean_iou"],
            "val_iou_water": val_m["iou_water"],
            "val_f1_water": val_m["f1_water"],
            "val_water_pred_pixels": val_m["val_water_pred_pixels"],
            "val_total_valid_pixels": val_m["val_total_valid_pixels"],
            "pred_water_fraction": val_m["pred_water_fraction"],
            "mean_p_water": val_m["mean_p_water"],
            "epoch_max_grad_norm": epoch_max_grad_norm,
            "gradients_dead": grads_dead,
            "elapsed_seconds": round(time.time() - t0, 1),
        }
        epochs_out.append(row)
        print(f"[{label}] epoch={epoch} train_dice={row['train_loss_dice']:.6f} "
              f"val_miou={row['val_miou']:.6f} val_iou_water={row['val_iou_water']:.6f} "
              f"val_water_pred_px={row['val_water_pred_pixels']} "
              f"mean_p_water={row['mean_p_water']:.6f} grad_norm={epoch_max_grad_norm:.3e} "
              f"dead={grads_dead}", flush=True)
        if row["val_water_pred_pixels"] == 0 and grads_dead:
            collapsed = True
            print(f"[{label}] COLLAPSE (gradient-dead all-background) at epoch={epoch}", flush=True)
            break

    # final collapse verdict: dead+zero water OR zero water on every epoch
    zero_every = all(r["val_water_pred_pixels"] == 0 for r in epochs_out)
    verdict = collapsed or zero_every
    result = {"label": label, "d4_mode": d4_mode, "epochs": epochs_out, "collapsed": bool(verdict)}

    del task, criterion, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report: dict[str, Any] = {
        "step": "6C-v2-D4-AB-microtest",
        "device": str(device),
        "epochs": EPOCHS,
        "seed": SEED,
        "lambda_topo": 0.0,
        "note": "Pure-Dice (lambda=0) isolation of manual D4 vs albumentations D4.",
    }
    report["A_manual_d4"] = run_path("A_manual_d4", "manual", device)
    report["B_albumentations_d4"] = run_path("B_albumentations_d4", "albumentations", device)

    report["conclusion"] = {
        "manual_d4_collapsed": report["A_manual_d4"]["collapsed"],
        "albumentations_d4_collapsed": report["B_albumentations_d4"]["collapsed"],
        "manual_d4_is_cause": (report["A_manual_d4"]["collapsed"] and not report["B_albumentations_d4"]["collapsed"]),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(jsafe(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\n=== CONCLUSION ===")
    print(json.dumps(jsafe(report["conclusion"]), indent=2))
    print(f"\nWROTE: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
