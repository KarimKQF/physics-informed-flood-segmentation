"""SegMAN (CVPR 2025) training runner for Sen1Floods11 flood/water segmentation.

Physics-informed-loss study: the SegMAN architecture is held fixed and only the
loss changes across four config-selected variants (ce / dice_ce / dice_ce_topo /
dice_ce_topo_dem_shuffled).

Reuses the existing repository machinery:
  * data + DEM + reproducible DEM-shuffle  : scripts/step6c_v3_train.TopographyDataModule
  * confusion-matrix metrics + topo counts : scripts/step6c_lambda05_train (t6c)
  * topographic loss                       : src/losses/physics_topographic_loss
SegMAN-specific additions are the 15-channel input assembly + normalization, the
SegMAN model, and the config-driven loss selector.

DEM is never a model input -- it is used only inside the topographic loss
(batch["topography"]). Run:

    python experiments_cvpr/segman/train_segman.py --config configs/segman/segman_ce.yaml
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
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

# --- repository paths ------------------------------------------------------- #
SEGMAN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SEGMAN_ROOT.parents[1]
for p in (str(SEGMAN_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import step6c_lambda05_train as t6c  # noqa: E402  (metric + checkpoint helpers)
from step6c_v3_train import TopographyDataModule  # noqa: E402  (data + DEM + shuffle)

from segman_losses.segman_loss import build_loss, lambda_for_epoch  # noqa: E402
from model.segman_model import build_segman  # noqa: E402


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(t6c.json_safe(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a" if exists else "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()), extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def configure_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding="utf-8")):
        handler.setFormatter(fmt)
        root.addHandler(handler)


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# 15-channel input assembly + normalization                                   #
# --------------------------------------------------------------------------- #
class InputAssembler:
    """Concatenate the per-modality image dict into a single normalized tensor.

    DEM is *not* included here -- it stays in batch["topography"] for the loss.
    """

    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        data_args = config["data"]["init_args"]
        self.modalities = list(data_args["modalities"])
        means = data_args["means"]
        stds = data_args["stds"]
        self.mean = {
            m: torch.tensor(means[m], dtype=torch.float32, device=device).view(1, -1, 1, 1)
            for m in self.modalities
        }
        self.std = {
            m: torch.tensor(stds[m], dtype=torch.float32, device=device).view(1, -1, 1, 1)
            for m in self.modalities
        }
        self.in_chans = sum(len(means[m]) for m in self.modalities)

    def __call__(self, image: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = []
        for m in self.modalities:
            x = image[m].float()
            parts.append((x - self.mean[m]) / self.std[m])
        return torch.cat(parts, dim=1)


def get_target(batch: dict[str, Any]) -> torch.Tensor:
    mask = batch["mask"]
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]
    return mask.long()


# --------------------------------------------------------------------------- #
# evaluation                                                                   #
# --------------------------------------------------------------------------- #
def evaluate_split(model, criterion, assembler, loader, config, device) -> dict[str, Any]:
    model.eval()
    totals = {"loss_total": 0.0, "loss_ce": 0.0, "loss_dice": 0.0, "loss_topo": 0.0}
    matrix = [[0, 0], [0, 0]]
    topo_counts = {"topo_descending_pair_count": 0, "topo_violation_pair_count": 0,
                   "topo_violation_fraction": math.nan}
    loss_cfg = config["loss"]
    batches = 0
    with torch.no_grad():
        for raw_batch in loader:
            batch = t6c.move_batch(raw_batch, device)
            x = assembler(batch["image"])
            target = get_target(batch)
            topo = batch.get("topography")
            logits = model(x)
            losses = criterion(logits, target, topo)
            for k in totals:
                totals[k] += float(losses[k].detach().cpu())
            pred = torch.argmax(logits.detach(), dim=1)
            matrix = t6c.add_conf(matrix, t6c.confusion(target, pred))
            if topo is not None:
                topo_counts = t6c.add_topo_counts(
                    topo_counts,
                    t6c.topographic_violation_counts(
                        logits=logits.detach(), target=target,
                        topography=topo.squeeze(1),
                        ignore_index=int(loss_cfg.get("ignore_index", -1)),
                        water_class=int(loss_cfg.get("water_class", 1)),
                        elevation_margin=float(loss_cfg.get("topo", {}).get("elevation_margin", 0.0)),
                        neighborhood=str(loss_cfg.get("topo", {}).get("neighborhood", "4")),
                    ),
                )
            batches += 1
    metrics = t6c.metrics_from_conf(matrix)
    for k, v in totals.items():
        metrics[k] = v / batches if batches else math.nan
    metrics.update(topo_counts)
    metrics["water_pred_pixels"] = int(metrics["tp"]) + int(metrics["fp"])
    metrics["batches"] = batches
    return metrics


def save_pred_samples(model, assembler, loader, out_dir: Path, device, max_samples: int = 6) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved = 0
    with torch.no_grad():
        for raw_batch in loader:
            batch = t6c.move_batch(raw_batch, device)
            logits = model(assembler(batch["image"]))
            pred = torch.argmax(logits, dim=1).cpu().numpy().astype("uint8")
            target = get_target(batch).cpu().numpy().astype("int16")
            for i in range(pred.shape[0]):
                np.savez_compressed(out_dir / f"pred_{saved:03d}.npz", pred=pred[i], target=target[i])
                saved += 1
                if saved >= max_samples:
                    return


# --------------------------------------------------------------------------- #
# main training                                                                #
# --------------------------------------------------------------------------- #
def run(config_path: Path, *, max_epochs_override: int | None = None,
        max_train_batches: int | None = None,
        run_dir_override: Path | None = None,
        run_tag_override: str | None = None) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8-sig") as handle:
        config = yaml.safe_load(handle)

    # CLI overrides (used to run alternate variants without editing the config file).
    if run_dir_override is not None:
        config["run_dir"] = str(run_dir_override)
    if run_tag_override is not None:
        config["run_tag"] = str(run_tag_override)

    # Inject DEM-shuffle map if requested (reproducible derangement file).
    dem_map_file = config.get("dem", {}).get("dem_tile_id_map_file")
    if dem_map_file:
        with open(dem_map_file, encoding="utf-8") as f:
            config.setdefault("dem", {})["dem_tile_id_map"] = json.load(f).get("mapping", {})

    run_dir = Path(config["run_dir"])
    tag = config.get("run_tag", run_dir.name)
    paths = {
        "metrics": run_dir / "metrics",
        "logs": run_dir / "logs",
        "checkpoints": run_dir / "checkpoints",
        "configs": run_dir / "configs",
        "predictions": run_dir / "predictions",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    epoch_csv = paths["metrics"] / "training_epoch_metrics.csv"
    summary_json = paths["metrics"] / f"{tag}_summary.json"
    best_ckpt = paths["checkpoints"] / "best_checkpoint.pt"
    last_ckpt = paths["checkpoints"] / "last_checkpoint.pt"
    pid_file = run_dir / "run.pid"
    configure_logging(paths["logs"] / f"{tag}_training.log")

    if any(f.exists() for f in (best_ckpt, last_ckpt, epoch_csv)) and not max_train_batches:
        raise RuntimeError(f"Refusing to overwrite existing run artifacts in {run_dir}")

    # Write PID file so the chain launcher can detect orphaned training processes.
    # Left on disk intentionally after completion so the chain can distinguish
    # "ran and finished" (summary status=done) from "ran and was killed" (no done status).
    pid_file.write_text(str(os.getpid()), encoding="utf-8")

    seed = int(config.get("seed", config.get("seed_everything", 42)))
    seed_everything(seed)
    shutil.copy2(config_path, paths["configs"] / config_path.name)

    if config.get("dem", {}).get("use_as_model_input", False):
        raise RuntimeError("DEM as model input is forbidden; DEM is loss-only.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = config["trainer"]
    batch_size = int(trainer["batch_size"])
    grad_accum = int(trainer.get("gradient_accumulation_steps", 1))
    max_epochs = max_epochs_override or int(trainer["max_epochs"])
    es_patience = int(trainer.get("early_stopping_patience", max_epochs))
    es_min_epochs = int(trainer.get("early_stopping_min_epochs", 0))

    logging.info("SegMAN run start: tag=%s dir=%s loss_mode=%s seed=%d", tag, run_dir,
                 config["loss"]["mode"], seed)
    logging.info("device=%s batch_size=%d grad_accum=%d max_epochs=%d", device, batch_size,
                 grad_accum, max_epochs)

    dm = TopographyDataModule(config, batch_size=batch_size)
    dm.setup("fit")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    assembler = InputAssembler(config, device)
    model_cfg = dict(config["model"])
    model_cfg.setdefault("in_chans", assembler.in_chans)
    model = build_segman(model_cfg).to(device)
    criterion = build_loss(config).to(device)
    logging.info("model=SegMAN-%s in_chans=%d params=%.2fM",
                 model_cfg.get("variant", "s"), assembler.in_chans,
                 sum(p.numel() for p in model.parameters()) / 1e6)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=float(config["optimizer"]["init_args"]["lr"]),
        weight_decay=float(config["optimizer"]["init_args"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    summary: dict[str, Any] = {
        "model": "SegMAN", "variant": model_cfg.get("variant", "s"), "run_tag": tag,
        "loss_mode": config["loss"]["mode"], "seed": seed, "status": "running",
        "run_dir": str(run_dir).replace("\\", "/"), "in_chans": assembler.in_chans,
        "dem_as_model_input": False, "dem_in_loss_only": config["loss"]["mode"] in
        ("dice_ce_topo", "dice_ce_topo_dem_shuffled"), "started_at": now_utc(),
    }
    write_json(summary_json, summary)

    best_miou = -math.inf
    best_epoch = 0
    no_improve = 0
    train_start = time.time()

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        epoch_lambda = lambda_for_epoch(config, epoch)
        criterion.set_lambda_topo(epoch_lambda)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_totals = {"loss_total": 0.0, "loss_ce": 0.0, "loss_dice": 0.0, "loss_topo": 0.0}
        train_matrix = [[0, 0], [0, 0]]
        max_grad_norm = 0.0
        batches = 0
        n_train = len(train_loader) if max_train_batches is None else min(max_train_batches, len(train_loader))

        for batch_idx, raw_batch in enumerate(train_loader, start=1):
            if max_train_batches is not None and batch_idx > max_train_batches:
                break
            batch = t6c.move_batch(raw_batch, device)
            x = assembler(batch["image"])
            target = get_target(batch)
            topo = batch.get("topography")
            logits = model(x)
            losses = criterion(logits, target, topo)
            loss_total = losses["loss_total"]
            if not torch.isfinite(loss_total).item():
                raise RuntimeError(f"Non-finite loss at epoch={epoch} batch={batch_idx}")
            (loss_total / grad_accum).backward()
            if batch_idx % grad_accum == 0 or batch_idx == n_train:
                grad_sq = sum(float(p.grad.detach().pow(2).sum()) for p in params if p.grad is not None)
                max_grad_norm = max(max_grad_norm, math.sqrt(grad_sq))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            pred = torch.argmax(logits.detach(), dim=1)
            train_matrix = t6c.add_conf(train_matrix, t6c.confusion(target, pred))
            for k in train_totals:
                train_totals[k] += float(losses[k].detach().cpu())
            batches += 1

        train_metrics = t6c.metrics_from_conf(train_matrix)
        for k, v in train_totals.items():
            train_metrics[k] = v / batches if batches else math.nan

        val_metrics = evaluate_split(model, criterion, assembler, val_loader, config, device)
        cur_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_metrics["mean_iou"])

        improved = val_metrics["mean_iou"] > best_miou
        ckpt_payload = {
            "epoch": epoch, "best_validation_miou": max(best_miou, val_metrics["mean_iou"]),
            "best_epoch": best_epoch if not improved else epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
            "config": config, "saved_at": now_utc(),
        }
        if improved:
            best_miou = float(val_metrics["mean_iou"])
            best_epoch = epoch
            no_improve = 0
            torch.save(ckpt_payload, best_ckpt)
        else:
            no_improve += 1
        torch.save(ckpt_payload, last_ckpt)

        row = {
            "epoch": epoch, "lambda_topo_epoch": epoch_lambda,
            "train_loss_total": train_metrics["loss_total"], "train_loss_ce": train_metrics["loss_ce"],
            "train_loss_dice": train_metrics["loss_dice"], "train_loss_topo": train_metrics["loss_topo"],
            "train_miou": train_metrics["mean_iou"],
            "val_loss_total": val_metrics["loss_total"], "val_loss_ce": val_metrics["loss_ce"],
            "val_loss_dice": val_metrics["loss_dice"], "val_loss_topo": val_metrics["loss_topo"],
            "val_miou": val_metrics["mean_iou"], "val_iou_water": val_metrics["iou_water"],
            "val_f1_water": val_metrics["f1_water"], "val_precision_water": val_metrics["precision_water"],
            "val_recall_water": val_metrics["recall_water"], "val_water_pred_pixels": val_metrics["water_pred_pixels"],
            "val_topo_violation_fraction": val_metrics["topo_violation_fraction"],
            "learning_rate": cur_lr, "best_epoch": best_epoch, "best_validation_miou": best_miou,
            "no_improve": no_improve, "epoch_max_grad_norm": max_grad_norm,
            "elapsed_seconds": round(time.time() - epoch_start, 3), "improved": improved,
        }
        append_csv(epoch_csv, row)
        logging.info(
            "epoch=%d lambda=%.4f train_total=%.5f val_miou=%.5f val_iou_water=%.5f "
            "val_water_px=%d loss_ce=%.4f loss_dice=%.4f loss_topo=%.6f grad=%.3e best=%d",
            epoch, epoch_lambda, row["train_loss_total"], row["val_miou"], row["val_iou_water"],
            row["val_water_pred_pixels"], row["val_loss_ce"], row["val_loss_dice"],
            row["val_loss_topo"], max_grad_norm, best_epoch,
        )
        summary.update({"last_epoch": epoch, "best_epoch": best_epoch,
                        "best_validation_miou": best_miou,
                        "training_elapsed_seconds": round(time.time() - train_start, 3)})
        write_json(summary_json, summary)

        if no_improve >= es_patience and epoch >= es_min_epochs:
            logging.info("early stop epoch=%d no_improve=%d best_epoch=%d", epoch, no_improve, best_epoch)
            break

    # Final eval on best checkpoint over valid/test/bolivia + save predictions.
    evals: dict[str, Any] = {}
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device)["model_state_dict"])
        for split in ("valid", "test", "bolivia"):
            try:
                dm.setup("test") if split == "test" else None
                dm.split = split
                dm.setup("test")
                loader = dm.test_dataloader()
                evals[split] = evaluate_split(model, criterion, assembler, loader, config, device)
                logging.info("final_%s mIoU=%.5f iou_water=%.5f f1=%.5f", split,
                             evals[split]["mean_iou"], evals[split]["iou_water"], evals[split]["f1_water"])
            except Exception as exc:  # noqa: BLE001
                logging.warning("final eval for %s failed: %s", split, exc)
        save_pred_samples(model, assembler, dm.test_dataloader(), paths["predictions"] / "test", device)

    summary.update({"status": "done", "completed_at": now_utc(), "evaluations": evals,
                    "best_epoch": best_epoch, "best_validation_miou": best_miou})
    write_json(summary_json, summary)
    logging.info("SegMAN run complete: tag=%s best_epoch=%d best_val_miou=%.5f", tag, best_epoch, best_miou)
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--max-epochs", type=int, default=None, help="override for smoke tests")
    ap.add_argument("--max-train-batches", type=int, default=None, help="limit batches/epoch (smoke)")
    ap.add_argument("--run-dir", type=Path, default=None, dest="run_dir",
                    help="override run_dir from config (safe re-run without editing yaml)")
    ap.add_argument("--run-tag", type=str, default=None, dest="run_tag",
                    help="override run_tag from config")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    summary = run(
        args.config,
        max_epochs_override=args.max_epochs,
        max_train_batches=args.max_train_batches,
        run_dir_override=args.run_dir,
        run_tag_override=args.run_tag,
    )
    return 0 if summary.get("status") == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
