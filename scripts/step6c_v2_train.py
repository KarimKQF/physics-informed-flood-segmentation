"""
STEP 6C v2 training runner (config-driven).

Identical recipe to STEP 5S-A / STEP 6C (TerraMind-L + UPerNet, corrected indices,
Dice, FP32, bs2/accum4, manual D4 to image+mask+DEM) but with two additions that
fix the STEP 6C lambda=0.5 all-background collapse:

  1. Exact Dice parity: the segmentation term is smp.losses.DiceLoss("multiclass",
     ignore_index=-1) via CombinedDicePhysicsLoss, bit-exact with STEP 5S-A.
  2. Epoch-wise lambda schedule (constant OR warmup_linear), so the topographic
     prior can be kept at 0 during the fragile early epochs and ramped in later.

All paths are read from the config (run_dir, run_tag), so each config writes to its
own run directory and never overwrites the failed lambda=0.5 run.

Guardrails: DEM is loss-only (never a model input), no DARN, no STURM, raw data
unchanged. Refuses to overwrite existing run artifacts unless --resume.

DO NOT launch full training without explicit instruction. Smoke tests import the
helpers here but never call main().

Usage:
  python step6c_v2_train.py --config <config.yaml> [--resume <ckpt>] [--log-file <path>]
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for _p in (str(SRC_ROOT), str(SCRIPTS_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse the EXACT, proven data / metric / DEM / D4 helpers from the lambda=0.5 runner
# so the only intentional differences are the lambda schedule, the parity-fixed Dice,
# and config-driven paths. These helpers are pure functions (no module global state).
import albumentations as A  # noqa: E402

import step6c_lambda05_train as t6c  # noqa: E402
from losses.combined_loss import CombinedDicePhysicsLoss  # noqa: E402


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# D4 augmentation
# ---------------------------------------------------------------------------
# The manual torch rot90/flip D4 path (t6c.apply_d4_to_batch) was proven to cause
# the all-background gradient-dead collapse (see STEP_6C_V2_D4_AB_MICROTEST_REPORT.md:
# manual D4 -> collapse, albumentations.D4 -> healthy, with everything else identical).
# We therefore use albumentations' battle-tested D4 via ReplayCompose so the EXACT same
# geometric operation is applied to every image modality, the mask, and the DEM. This
# keeps image/mask/DEM perfectly aligned and matches the STEP 5S-A augmentation family.
_D4_REPLAY = A.ReplayCompose([A.D4(p=1.0)], is_check_shapes=False)


def apply_albu_d4_to_batch(batch: dict[str, Any], dem: torch.Tensor) -> None:
    """In-place: one random albumentations D4 op per sample, shared across image
    modalities + mask + DEM. Operates on CPU tensors (call before move_batch).

    DEM is augmented here ONLY so it stays aligned for the topographic loss; it is
    never fed to the model.
    """
    mods = list(batch["image"].keys())
    bsize = len(batch["mask"])
    for i in range(bsize):
        primary = batch["image"][mods[0]][i].permute(1, 2, 0).contiguous().numpy()  # HWC
        mask_np = batch["mask"][i].contiguous().numpy()
        out = _D4_REPLAY(image=primary, mask=mask_np)
        replay = out["replay"]
        batch["image"][mods[0]][i] = torch.from_numpy(
            np.ascontiguousarray(out["image"])).permute(2, 0, 1).contiguous().to(batch["image"][mods[0]].dtype)
        batch["mask"][i] = torch.from_numpy(np.ascontiguousarray(out["mask"])).to(batch["mask"].dtype)
        for m in mods[1:]:
            arr = batch["image"][m][i].permute(1, 2, 0).contiguous().numpy()
            r = A.ReplayCompose.replay(replay, image=arr)
            batch["image"][m][i] = torch.from_numpy(
                np.ascontiguousarray(r["image"])).permute(2, 0, 1).contiguous().to(batch["image"][m].dtype)
        dr = A.ReplayCompose.replay(replay, image=dem[i].contiguous().numpy())
        dem[i] = torch.from_numpy(np.ascontiguousarray(dr["image"])).to(dem.dtype)


def apply_train_d4(batch: dict[str, Any], dem: torch.Tensor, mode: str, rng) -> None:
    """Dispatch the train-time D4 augmentation by mode."""
    if mode == "albu_d4_replay":
        apply_albu_d4_to_batch(batch, dem)
    elif mode == "manual":  # legacy path that caused the collapse; kept for reproducibility
        t6c.apply_d4_to_batch(batch, dem, rng)
    elif mode in (None, "none"):
        return
    else:
        raise ValueError(f"Unknown data.d4_mode: {mode!r}")


# ---------------------------------------------------------------------------
# Lambda schedule
# ---------------------------------------------------------------------------

def lambda_for_epoch(physics: dict[str, Any], epoch: int) -> float:
    """Return the topographic weight for a given 1-indexed epoch.

    Supported schedule types (config["physics_loss"]["lambda_schedule"]["type"]):
      - "constant" (or missing): lambda_topo every epoch.
      - "warmup_linear":
            epoch <= warmup_epochs                       -> 0.0
            warmup_epochs < epoch < warmup+ramp_epochs   -> linear 0 -> lambda_topo
            epoch >= warmup_epochs + ramp_epochs         -> lambda_topo
    """
    base = float(physics["lambda_topo"])
    sched = physics.get("lambda_schedule") or {}
    stype = str(sched.get("type", "constant")).lower()

    if stype == "constant":
        return base
    if stype == "warmup_linear":
        warm = int(sched.get("warmup_epochs", 0))
        ramp = int(sched.get("ramp_epochs", 0))
        if epoch <= warm:
            return 0.0
        if ramp <= 0 or epoch >= warm + ramp:
            return base
        frac = (epoch - warm) / ramp
        return base * frac
    raise ValueError(f"Unsupported lambda_schedule.type: {stype!r}")


# ---------------------------------------------------------------------------
# Config-driven path resolution
# ---------------------------------------------------------------------------

class RunPaths:
    def __init__(self, config: dict[str, Any]) -> None:
        self.run_dir = Path(config["run_dir"])
        tag = config.get("run_tag") or self.run_dir.name
        self.tag = tag
        self.logs = self.run_dir / "logs"
        self.checkpoints = self.run_dir / "checkpoints"
        self.metrics = self.run_dir / "metrics"
        self.epoch_csv = self.metrics / "training_epoch_metrics.csv"
        self.summary_json = self.metrics / f"{tag}_summary.json"
        self.training_state = self.metrics / "training_state.json"
        self.final_json = self.metrics / f"{tag}_final_metrics.json"
        self.final_csv = self.metrics / f"{tag}_final_metrics.csv"
        self.best_ckpt = self.checkpoints / "best_checkpoint.pt"
        self.last_ckpt = self.checkpoints / "last_checkpoint.pt"
        self.default_log = self.logs / f"{tag}_training.log"

    def ensure_dirs(self) -> None:
        for sub in ["logs", "checkpoints", "metrics", "configs", "scripts",
                    "predictions/valid", "predictions/test", "predictions/bolivia"]:
            (self.run_dir / sub).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Loss / model
# ---------------------------------------------------------------------------

def build_loss(config: dict[str, Any]) -> CombinedDicePhysicsLoss:
    physics = config["physics_loss"]
    # lambda_topo passed here is the MAX/base; the schedule sets the per-epoch value.
    return CombinedDicePhysicsLoss(
        lambda_topo=float(physics["lambda_topo"]),
        ignore_index=int(physics["ignore_index"]),
        water_class=int(physics["water_class"]),
        dice_smooth=float(physics.get("dice_smooth", 0.0)),  # 0.0 => exact 5S-A parity
        elevation_margin=float(physics["elevation_margin"]),
        elevation_scale=float(physics["elevation_scale"]),
        use_elevation_weight=bool(physics["use_elevation_weight"]),
        neighborhood=str(physics["neighborhood"]),
    )


def maybe_init_from_checkpoint(task: Any, config: dict[str, Any], device: torch.device) -> str | None:
    """Optionally warm-start model weights (only) from another run's checkpoint.

    Used by the SECONDARY fine-tune experiment (init from STEP 5S-A best). The
    primary fair-comparison configs do NOT set this and start from the original
    TerraMind pretrained checkpoint like 5S-A.
    """
    init = config.get("initialization") or {}
    ckpt_path = init.get("from_checkpoint")
    if not ckpt_path:
        return None
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"initialization.from_checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    missing, unexpected = task.load_state_dict(state, strict=True)
    logging.info("Warm-started model weights from %s (missing=%s unexpected=%s)",
                 ckpt_path, list(missing), list(unexpected))
    return str(ckpt_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--log-file", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    is_resume = args.resume is not None

    with args.config.open("r", encoding="utf-8-sig") as fh:
        config = yaml.safe_load(fh)

    paths = RunPaths(config)
    paths.ensure_dirs()

    log_file = args.log_file if args.log_file else paths.default_log
    # logging
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); root.addHandler(sh)
    fh = logging.FileHandler(log_file, encoding="utf-8"); fh.setFormatter(fmt); root.addHandler(fh)

    physics = config["physics_loss"]
    sched = physics.get("lambda_schedule") or {"type": "constant"}
    d4_mode = (config.get("data", {}) or {}).get("d4_mode", "albu_d4_replay")

    logging.info("=" * 78)
    logging.info("STEP 6C v2: TerraMind-L + UPerNet Dice + TopographicInconsistencyLoss")
    logging.info("run_tag=%s  run_dir=%s", paths.tag, paths.run_dir)
    logging.info("lambda_topo(base)=%.4f  schedule=%s  d4_mode=%s", float(physics["lambda_topo"]), sched, d4_mode)
    logging.info("Guardrails: DEM loss-only, no DEM model input, no DARN, no STURM, raw data unchanged")
    logging.info("=" * 78)

    # Guardrails
    if config.get("dem", {}).get("use_as_model_input", False):
        raise ValueError("DEM as model input is forbidden.")
    if not is_resume and any(p.exists() for p in [paths.best_ckpt, paths.last_ckpt, paths.epoch_csv,
                                                  paths.training_state, paths.final_json]):
        raise RuntimeError(f"Refusing to overwrite existing run artifacts in {paths.run_dir}")

    seed = int(config.get("seed_everything", 42))
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    shutil.copy2(args.config, paths.run_dir / "configs" / args.config.name)
    shutil.copy2(Path(__file__), paths.run_dir / "scripts" / Path(__file__).name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Device: %s", device)
    if device.type == "cuda":
        logging.info("GPU: %s VRAM=%.0f MB", torch.cuda.get_device_name(0),
                     torch.cuda.get_device_properties(0).total_memory / 1024 ** 2)

    batch_size = int(config["trainer"]["batch_size"])
    grad_accum = int(config["trainer"]["gradient_accumulation_steps"])
    max_epochs = int(config["trainer"]["max_epochs"])
    es_patience = int(config["trainer"]["early_stopping_patience"])
    es_min_epochs = int(config["trainer"]["early_stopping_min_epochs"])

    dm = t6c.build_datamodule(config, batch_size=batch_size, train_aug=True)
    dm.setup("fit")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    task = t6c.build_task(config).to(device)
    warm_start = maybe_init_from_checkpoint(task, config, device)
    initial_bn = t6c.set_bn_eval(task)
    criterion = build_loss(config).to(device)

    params = [p for p in task.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=float(config["optimizer"]["init_args"]["lr"]),
        weight_decay=float(config["optimizer"]["init_args"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    start_epoch, best_miou, best_epoch, no_improve = 1, -math.inf, 0, 0
    if is_resume:
        start_epoch, best_miou, best_epoch, no_improve = t6c.load_checkpoint(
            args.resume, task, optimizer, scheduler, device)

    summary: dict[str, Any] = {
        "step": "6C-v2",
        "run_tag": paths.tag,
        "status": "running",
        "run_dir": str(paths.run_dir),
        "config_path": str(args.config),
        "lambda_topo_base": float(physics["lambda_topo"]),
        "lambda_schedule": sched,
        "d4_mode": d4_mode,
        "warm_start_checkpoint": warm_start,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "effective_batch_size": batch_size * grad_accum,
        "precision": config["trainer"]["precision"],
        "dem_as_model_input": False,
        "dem_in_loss_only": True,
        "darn_started": False,
        "sturm_training_started": False,
        "raw_data_modified": False,
        "batchnorm_eval_modules_initial": initial_bn,
        "started_at": now_utc(),
        "is_resume": is_resume,
    }
    t6c.write_json(paths.summary_json, summary)

    d4_rng = random.Random(seed)
    train_start = time.time()

    try:
        for epoch in range(start_epoch, max_epochs + 1):
            epoch_lambda = lambda_for_epoch(physics, epoch)
            criterion.set_lambda_topo(epoch_lambda)

            epoch_start = time.time()
            task.train()
            bn_count = t6c.set_bn_eval(task)
            optimizer.zero_grad(set_to_none=True)
            matrix = [[0, 0], [0, 0]]
            train_totals = {"loss_total": 0.0, "loss_dice": 0.0, "loss_topo": 0.0}
            batches = 0
            epoch_max_grad_norm = 0.0  # for the gradient-dead collapse guard

            for batch_idx, raw_batch in enumerate(train_loader, start=1):
                dem_cpu = t6c.load_dem_batch(config, raw_batch, split="train")
                apply_train_d4(raw_batch, dem_cpu, d4_mode, d4_rng)
                batch = t6c.move_batch(raw_batch, device)
                dem = dem_cpu.to(device, non_blocking=True)

                losses, logits, target = t6c.compute_loss(task, criterion, batch, dem)
                loss_total = losses["loss_total"]
                if not torch.isfinite(loss_total).item():
                    raise RuntimeError(f"Non-finite total loss at epoch={epoch} batch={batch_idx}")
                if not torch.isfinite(logits).all().item():
                    raise RuntimeError(f"Non-finite logits at epoch={epoch} batch={batch_idx}")

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
                for key in train_totals:
                    train_totals[key] += float(losses[key].detach().cpu())
                batches += 1

            train_m = t6c.metrics_from_conf(matrix)
            for key, value in train_totals.items():
                train_m[key] = value / batches if batches else math.nan
            train_m["lambda_topo_loss"] = epoch_lambda * train_m["loss_topo"]

            val_m = t6c.evaluate_split(task=task, criterion=criterion, loader=val_loader,
                                       config=config, split="valid", device=device)
            cur_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_m["mean_iou"])

            improved = val_m["mean_iou"] > best_miou
            if improved:
                best_miou = float(val_m["mean_iou"]); best_epoch = epoch; no_improve = 0
                t6c.save_ckpt(paths.best_ckpt, task, optimizer, scheduler, epoch,
                              best_miou, best_epoch, no_improve, config)
            else:
                no_improve += 1
            t6c.save_ckpt(paths.last_ckpt, task, optimizer, scheduler, epoch,
                          best_miou, best_epoch, no_improve, config)

            t6c.write_json(paths.training_state, {
                "epoch": epoch, "best_epoch": best_epoch, "best_validation_miou": best_miou,
                "no_improve": no_improve, "current_lr": optimizer.param_groups[0]["lr"],
                "lambda_topo_epoch": epoch_lambda, "updated_at": now_utc(),
            })

            row = {
                "epoch": epoch,
                "lambda_topo_epoch": epoch_lambda,
                "train_loss_total": train_m["loss_total"],
                "train_loss_dice": train_m["loss_dice"],
                "train_loss_topo": train_m["loss_topo"],
                "train_lambda_topo_loss": train_m["lambda_topo_loss"],
                "train_miou": train_m["mean_iou"],
                "val_loss_total": val_m["loss_total"],
                "val_loss_dice": val_m["loss_dice"],
                "val_loss_topo": val_m["loss_topo"],
                "val_miou": val_m["mean_iou"],
                "val_iou_water": val_m["iou_water"],
                "val_f1_water": val_m["f1_water"],
                "val_topo_score": val_m["topographic_inconsistency_score"],
                "val_topo_violation_fraction": val_m["topo_violation_fraction"],
                "learning_rate": cur_lr,
                "best_epoch": best_epoch,
                "best_validation_miou": best_miou,
                "no_improve": no_improve,
                "batchnorm_eval_modules": bn_count,
                "epoch_max_grad_norm": epoch_max_grad_norm,
                "elapsed_seconds": round(time.time() - epoch_start, 3),
                "improved": improved,
            }
            t6c.write_csv_append(paths.epoch_csv, [row], list(row.keys()))
            logging.info(
                "epoch=%d lambda_topo=%.5f train_loss_total=%.6f train_loss_dice=%.6f train_loss_topo=%.8f "
                "lambda_x_train_loss_topo=%.8f val_loss_total=%.6f val_loss_dice=%.6f val_loss_topo=%.8f "
                "val_miou=%.6f val_iou_water=%.6f val_f1_water=%.6f val_topo_score=%.8f "
                "val_topo_violation_fraction=%.8f lr=%.2e best_epoch=%d no_improve=%d bn_eval=%d "
                "epoch_max_grad_norm=%.6e",
                epoch, epoch_lambda, row["train_loss_total"], row["train_loss_dice"], row["train_loss_topo"],
                row["train_lambda_topo_loss"], row["val_loss_total"], row["val_loss_dice"], row["val_loss_topo"],
                row["val_miou"], row["val_iou_water"], row["val_f1_water"], row["val_topo_score"],
                row["val_topo_violation_fraction"], cur_lr, best_epoch, no_improve, bn_count,
                epoch_max_grad_norm,
            )

            summary.update({"last_epoch": epoch, "best_epoch": best_epoch,
                            "best_validation_miou": best_miou,
                            "training_elapsed_seconds": round(time.time() - train_start, 3)})
            t6c.write_json(paths.summary_json, summary)

            # ---- early-collapse watch + gradient-dead all-background stop guard ----
            val_water_pred_pixels = int(val_m["tp"]) + int(val_m["fp"])
            if val_m["iou_water"] == 0.0:
                logging.warning(
                    "EARLY-COLLAPSE-WATCH epoch=%d lambda_topo=%.5f val_iou_water=0 "
                    "val_water_pred_pixels=%d epoch_max_grad_norm=%.6e "
                    "(STEP 5S-A normally recovers by epoch 2; logging this, not hiding it)",
                    epoch, epoch_lambda, val_water_pred_pixels, epoch_max_grad_norm,
                )
            # A true gradient-dead all-background collapse: model predicts ZERO water on
            # the whole val set AND training gradients have vanished (~0). A healthy slow
            # start has large grads (~2-3), so this will not false-positive on warmup.
            if val_water_pred_pixels == 0 and epoch_max_grad_norm < 1e-6:
                logging.error(
                    "GRADIENT-DEAD ALL-BACKGROUND COLLAPSE detected at epoch=%d: "
                    "val_water_pred_pixels=0 AND epoch_max_grad_norm=%.6e < 1e-6. "
                    "This is the unrecoverable absorbing state from the lambda=0.5 run. "
                    "Stopping and reporting (last_checkpoint.pt preserved).",
                    epoch, epoch_max_grad_norm,
                )
                summary.update({"status": "collapsed_gradient_dead", "collapse_epoch": epoch,
                                "epoch_max_grad_norm": epoch_max_grad_norm,
                                "val_water_pred_pixels": 0, "blocked_at": now_utc()})
                t6c.write_json(paths.summary_json, summary)
                return 4

            if epoch >= es_min_epochs and no_improve >= es_patience:
                logging.info("Early stopping at epoch=%d (no improvement %d epochs after %d)",
                             epoch, es_patience, best_epoch)
                break

        # final evals on best checkpoint
        ckpt = torch.load(paths.best_ckpt, map_location=device, weights_only=False)
        task.load_state_dict(ckpt["model_state_dict"])
        evals: dict[str, Any] = {}
        rows: list[dict[str, Any]] = []
        for split in ["valid", "test", "bolivia"]:
            dmx = t6c.build_datamodule(config, split=split, batch_size=batch_size)
            dmx.setup("test")
            m = t6c.evaluate_split(task=task, criterion=criterion, loader=dmx.test_dataloader(),
                                   config=config, split=split, device=device)
            evals[split] = m
            rows.append({"split": split, **m})
            logging.info("final_%s mIoU=%.6f iou_water=%.6f f1_water=%.6f", split,
                         m["mean_iou"], m["iou_water"], m["f1_water"])
        t6c.write_json(paths.final_json, {"step": "6C-v2", "run_tag": paths.tag,
                                          "generated_at": now_utc(), "evaluations": evals})
        t6c.write_csv_overwrite(paths.final_csv, rows, list(rows[0].keys()))
        summary.update({"status": "done", "training_completed": True, "completed_at": now_utc(),
                        "evaluations": evals})
        t6c.write_json(paths.summary_json, summary)
        logging.info("STEP 6C v2 (%s) complete. Human validation required.", paths.tag)

    except torch.cuda.OutOfMemoryError:
        logging.exception("CUDA OOM.")
        summary.update({"status": "oom", "blocked_at": now_utc()})
        t6c.write_json(paths.summary_json, summary)
        return 3
    except Exception as exc:
        logging.exception("STEP 6C v2 training failed.")
        summary.update({"status": "blocked", "blocked_at": now_utc(), "error": repr(exc)})
        t6c.write_json(paths.summary_json, summary)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
