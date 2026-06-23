"""
STEP 5S-A bs2/accum4 — TerraMind-L + UPerNet corrected indices, Dice, FP32.

batch_size=2, gradient_accumulation_steps=4, effective_batch=8.
Corrected large-backbone UPerNet feature indices: [5, 11, 17, 23].
No physics loss. No topographic loss. No DEM input. No DARN. No STURM-Flood.

Resume support:
  python step5s_a_bs2_accum4_train.py --resume <checkpoint_path> [--log-file <path>]

Checkpoint format (v2, written by this script from resumed run onward):
  epoch, best_validation_miou, best_epoch, no_improve,
  model_state_dict, optimizer_state_dict, scheduler_state_dict, config

Checkpoint format (v1, written by original run, epoch 1-14):
  epoch, best_validation_miou, model_state_dict, optimizer_state_dict, config
  (scheduler_state_dict, best_epoch, no_improve absent — safely defaulted)
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import logging
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

RUN_DIR = Path("E:/flood_research/experiments/terramind_baseline/runs/step5s_a_terramind_l_upernet_corrected_indices_dice_bs2_accum4")
CONFIG_PATH = REPO_ROOT / "configs" / "step5s_a_terramind_l_upernet_corrected_indices_dice_bs2_accum4.yaml"
SPLIT_FILES = {
    "train":   Path("E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/manifests/flood_train_step5e_filtered.txt"),
    "valid":   Path("E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/manifests/flood_valid_step5e_filtered.txt"),
    "test":    Path("E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/manifests/flood_test_step5e_filtered.txt"),
    "bolivia": Path("E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/manifests/flood_bolivia_step5e_filtered.txt"),
}
EPOCH_CSV         = RUN_DIR / "metrics" / "training_epoch_metrics.csv"
SUMMARY_JSON      = RUN_DIR / "metrics" / "step5s_a_bs2_accum4_summary.json"
TRAINING_STATE    = RUN_DIR / "metrics" / "training_state.json"
BEST_CKPT         = RUN_DIR / "checkpoints" / "best_checkpoint.pt"
LAST_CKPT         = RUN_DIR / "checkpoints" / "last_checkpoint.pt"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def ensure_dirs() -> None:
    for sub in ["logs", "checkpoints", "metrics", "configs", "scripts",
                "predictions/valid", "predictions/test", "predictions/bolivia"]:
        (RUN_DIR / sub).mkdir(parents=True, exist_ok=True)


def json_safe(v: Any) -> Any:
    try:
        import numpy as np
    except Exception:
        np = None
    if isinstance(v, dict):
        return {str(k): json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [json_safe(x) for x in v]
    if np is not None and isinstance(v, np.integer):
        return int(v)
    if np is not None and isinstance(v, np.floating):
        v = float(v)
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, Path):
        return str(v).replace("\\", "/")
    return v


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a" if exists else "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def config_hash(config: dict) -> str:
    return hashlib.md5(json.dumps(json_safe(config), sort_keys=True).encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Checkpointing (v2 — full state)
# ---------------------------------------------------------------------------

def save_ckpt(
    path: Path,
    task: Any,
    optimizer: Any,
    scheduler: Any,
    epoch: int,
    best_miou: float,
    best_epoch: int,
    no_improve: int,
    config: dict,
) -> None:
    """Atomic checkpoint save: write to .tmp then os.replace() to avoid corruption on TDR crash."""
    import os
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({
        "step": "5S-A-bs2-accum4",
        "ckpt_version": 2,
        "epoch": epoch,
        "best_validation_miou": best_miou,
        "best_epoch": best_epoch,
        "no_improve": no_improve,
        "model_state_dict": task.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": config,
        "saved_at": now_utc(),
    }, tmp)
    os.replace(tmp, path)  # atomic rename on NTFS — preserves previous file if write crashes


def write_training_state(
    epoch: int,
    best_miou: float,
    best_epoch: int,
    no_improve: int,
    optimizer: Any,
    config: dict,
) -> None:
    write_json(TRAINING_STATE, {
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_validation_miou": best_miou,
        "no_improve": no_improve,
        "current_lr": optimizer.param_groups[0]["lr"],
        "config_hash": config_hash(config),
        "updated_at": now_utc(),
    })


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

def load_checkpoint(
    path: Path,
    task: Any,
    optimizer: Any,
    scheduler: Any,
    device: Any,
) -> tuple[int, float, int, int, bool]:
    """
    Returns (start_epoch, best_miou, best_epoch, no_improve, scheduler_restored).

    v1 checkpoints (original run, epochs 1-14) lack scheduler_state_dict,
    best_epoch, and no_improve. These are safely defaulted:
      - best_epoch = ckpt['epoch'] (the saved epoch was the best by definition,
        since best_checkpoint.pt is only written on improvement)
      - no_improve = 0 (same reason: checkpoint was written on improvement)
      - scheduler: reinitialized (at epoch 14 the scheduler had never fired,
        LR remained 2e-5, so reinit is equivalent to exact restore)
    """
    import torch

    logging.info("Loading checkpoint: %s", path)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    version = ckpt.get("ckpt_version", 1)
    logging.info("Checkpoint version: v%d  keys: %s", version, list(ckpt.keys()))

    task.load_state_dict(ckpt["model_state_dict"])
    logging.info("Model state dict restored (%d tensors).", len(ckpt["model_state_dict"]))

    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    logging.info("Optimizer state dict restored.")

    epoch       = int(ckpt["epoch"])
    best_miou   = float(ckpt["best_validation_miou"])
    best_epoch  = int(ckpt.get("best_epoch", epoch))
    no_improve  = int(ckpt.get("no_improve", 0))

    scheduler_restored = False
    if "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scheduler_restored = True
        logging.info("Scheduler state dict restored.")
    else:
        logging.warning(
            "scheduler_state_dict absent (v1 checkpoint). Scheduler reinitialized. "
            "At epoch %d the LR was 2e-5 and the scheduler had not fired, "
            "so reinit is equivalent to exact restore. "
            "This is a warm-start for the scheduler only — model and optimizer are fully restored.",
            epoch,
        )

    start_epoch = epoch + 1
    logging.info(
        "Resume: start_epoch=%d  best_miou=%.6f  best_epoch=%d  no_improve=%d  scheduler_restored=%s",
        start_epoch, best_miou, best_epoch, no_improve, scheduler_restored,
    )
    return start_epoch, best_miou, best_epoch, no_improve, scheduler_restored


# ---------------------------------------------------------------------------
# Model / data helpers
# ---------------------------------------------------------------------------

def set_bn_eval(module: Any) -> int:
    import torch.nn as nn
    count = 0
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            m.eval()
            for p in m.parameters(recurse=False):
                p.requires_grad = False
            count += 1
    return count


def move_batch(batch: dict, device: Any) -> dict:
    import torch
    out: dict = {}
    for k, v in batch.items():
        if isinstance(v, dict):
            out[k] = {ik: iv.to(device) if torch.is_tensor(iv) else iv for ik, iv in v.items()}
        elif torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def get_logits(model_output: Any) -> Any:
    if hasattr(model_output, "output"):
        return model_output.output
    if isinstance(model_output, dict):
        return model_output.get("output", model_output.get("out", next(iter(model_output.values()))))
    if isinstance(model_output, (list, tuple)):
        return model_output[0]
    return model_output


def compute_loss(task: Any, batch: dict) -> tuple[Any, Any]:
    model_out = task(batch["image"])
    logits    = get_logits(model_out)
    target    = task.squeeze_ground_truth(batch["mask"])
    loss_dict = task.train_loss_handler.compute_loss(model_out, target, task.criterion, task.aux_loss)
    return loss_dict["loss"], logits


def confusion(y_true: Any, y_pred: Any) -> list[list[int]]:
    import torch
    valid = y_true != -1
    yt = y_true[valid].long()
    yp = y_pred[valid].long()
    mat = torch.zeros((2, 2), dtype=torch.int64, device=yt.device)
    if yt.numel() > 0:
        mat = torch.bincount(yt * 2 + yp, minlength=4).reshape(2, 2)
    return [[int(x) for x in row] for row in mat.cpu().tolist()]


def add_conf(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
    return [[a[0][0]+b[0][0], a[0][1]+b[0][1]], [a[1][0]+b[1][0], a[1][1]+b[1][1]]]


def metrics_from_conf(mat: list[list[int]]) -> dict[str, Any]:
    tn, fp = mat[0]
    fn, tp = mat[1]
    total    = tn + fp + fn + tp
    prec     = tp / (tp+fp)   if (tp+fp)   else math.nan
    rec      = tp / (tp+fn)   if (tp+fn)   else math.nan
    f1       = 2*prec*rec/(prec+rec) if math.isfinite(prec+rec) and (prec+rec) else math.nan
    iou_bg   = tn / (tn+fp+fn) if (tn+fp+fn) else math.nan
    iou_w    = tp / (tp+fp+fn) if (tp+fp+fn) else math.nan
    miou     = (iou_bg+iou_w)/2 if math.isfinite(iou_bg) and math.isfinite(iou_w) else math.nan
    return {
        "accuracy": (tn+tp)/total if total else math.nan,
        "precision_water": prec, "recall_water": rec, "f1_water": f1,
        "iou_background": iou_bg, "iou_water": iou_w, "mean_iou": miou,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "support_background": tn+fp, "support_water": fn+tp, "valid_pixel_count": total,
    }


def evaluate_split(task: Any, loader: Any, device: Any) -> dict[str, Any]:
    import torch
    task.eval()
    set_bn_eval(task)
    total_loss = 0.0
    batches    = 0
    mat        = [[0, 0], [0, 0]]
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            loss, logits = compute_loss(task, batch)
            pred = torch.argmax(logits.detach(), dim=1)
            mat  = add_conf(mat, confusion(batch["mask"], pred))
            total_loss += float(loss.detach().cpu())
            batches    += 1
    m = metrics_from_conf(mat)
    m["loss"]    = total_loss / batches if batches else math.nan
    m["batches"] = batches
    return m


def build_datamodule(config: dict, *, split: str | None = None, batch_size: int = 2):
    import albumentations as A
    from albumentations.pytorch.transforms import ToTensorV2
    from terratorch.datamodules import GenericMultiModalDataModule

    args = dict(config["data"]["init_args"])
    args["batch_size"]     = batch_size
    args["num_workers"]    = 0
    args["pin_memory"]     = True
    args["train_transform"] = [A.D4(), ToTensorV2()]
    args["val_transform"]  = None
    args["test_transform"] = None
    if split in {"valid", "test", "bolivia"}:
        args["test_split"] = str(SPLIT_FILES[split])
    return GenericMultiModalDataModule(**args)


def build_task(config: dict):
    from terratorch.tasks import SemanticSegmentationTask
    mi = config["model"]["init_args"]
    return SemanticSegmentationTask(
        model_factory=mi["model_factory"],
        model_args=mi["model_args"],
        loss=mi["loss"],
        ignore_index=mi["ignore_index"],
        freeze_backbone=mi.get("freeze_backbone", False),
        freeze_decoder=mi.get("freeze_decoder", False),
        class_names=mi.get("class_names", ["Others", "Flood"]),
    )


def configure_logging(log_file: Path) -> None:
    ensure_dirs()
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(sh)
    root.addHandler(fh)


# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------

def run_final_evals(task: Any, config: dict, device: Any) -> dict[str, Any]:
    import torch
    logging.info("Loading best checkpoint for final evaluation: %s", BEST_CKPT)
    ckpt = torch.load(BEST_CKPT, map_location=device, weights_only=False)
    task.load_state_dict(ckpt["model_state_dict"])
    evals: dict[str, Any] = {}
    for split in ["valid", "test", "bolivia"]:
        dm = build_datamodule(config, split=split, batch_size=2)
        dm.setup("test")
        loader = dm.test_dataloader()
        m = evaluate_split(task, loader, device)
        evals[split] = m
        logging.info(
            "%s: mIoU=%.6f  iou_water=%.6f  f1_water=%.6f  loss=%.6f",
            split, m["mean_iou"], m["iou_water"], m["f1_water"], m["loss"],
        )
    return evals


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resume",   type=Path, default=None,
                   help="Path to checkpoint to resume from.")
    p.add_argument("--log-file", type=Path, default=None,
                   help="Override log file path (default: step5s_a_bs2_accum4_training.log).")
    p.add_argument("--config",   type=Path, default=CONFIG_PATH,
                   help="Config YAML path.")
    return p.parse_args()


def main() -> int:
    args   = parse_args()
    is_resume = args.resume is not None

    default_log = RUN_DIR / "logs" / "step5s_a_bs2_accum4_training.log"
    log_file = args.log_file if args.log_file else default_log

    ensure_dirs()
    configure_logging(log_file)

    logging.info("=" * 70)
    logging.info("STEP 5S-A bs2/accum4: TerraMind-L + UPerNet corrected indices Dice FP32")
    logging.info("Guardrails: no physics loss, no DEM input, no DARN, no STURM")
    logging.info("batch_size=2  grad_accum=4  effective_batch=8  max_epochs=80")
    if is_resume:
        logging.info("MODE: RESUME from checkpoint: %s", args.resume)
    else:
        logging.info("MODE: fresh start")
    logging.info("=" * 70)

    import torch
    torch.manual_seed(42)

    config_path = args.config if args.config.exists() else CONFIG_PATH
    with config_path.open("r", encoding="utf-8-sig") as f:
        config = yaml.safe_load(f)

    shutil.copy2(config_path, RUN_DIR / "configs" / config_path.name)
    shutil.copy2(Path(__file__), RUN_DIR / "scripts" / Path(__file__).name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Device: %s", device)
    if device.type == "cuda":
        logging.info("GPU: %s  VRAM: %.0f MB",
                     torch.cuda.get_device_name(0),
                     torch.cuda.get_device_properties(0).total_memory / 1024**2)

    batch_size   = int(config["trainer"]["batch_size"])
    grad_accum   = int(config["trainer"]["gradient_accumulation_steps"])
    max_epochs   = int(config["trainer"]["max_epochs"])
    es_patience  = int(config["trainer"]["early_stopping_patience"])
    es_min_epochs = int(config["trainer"]["early_stopping_min_epochs"])

    dm = build_datamodule(config, batch_size=batch_size)
    dm.setup("fit")
    train_loader = dm.train_dataloader()
    val_loader   = dm.val_dataloader()

    task = build_task(config).to(device)
    params = [p for p in task.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=float(config["optimizer"]["init_args"]["lr"]),
        weight_decay=float(config["optimizer"]["init_args"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3,
    )

    # --- Resume or fresh start ---
    start_epoch     = 1
    best_miou       = -math.inf
    best_epoch      = 0
    no_improve      = 0
    scheduler_restored = False

    if is_resume:
        start_epoch, best_miou, best_epoch, no_improve, scheduler_restored = load_checkpoint(
            args.resume, task, optimizer, scheduler, device,
        )
        resume_note = (
            "Full training-state resume: model + optimizer restored from checkpoint. "
            "Scheduler reinitialized (equivalent: LR=2e-5, patience count=0 at epoch 14)."
            if not scheduler_restored else
            "Full training-state resume: model + optimizer + scheduler all restored."
        )
        logging.info(resume_note)

    # Load pre-existing epoch CSV rows for appending
    epoch_rows: list[dict[str, Any]] = []
    if EPOCH_CSV.exists() and is_resume:
        import csv as _csv
        with EPOCH_CSV.open("r", encoding="utf-8") as f:
            epoch_rows = list(_csv.DictReader(f))
        logging.info("Loaded %d existing epoch rows from CSV for continuity.", len(epoch_rows))

    summary: dict[str, Any] = {
        "step": "5S-A-bs2-accum4",
        "status": "running",
        "run_dir": str(RUN_DIR),
        "config_path": str(config_path),
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "effective_batch_size": batch_size * grad_accum,
        "corrected_indices": [5, 11, 17, 23],
        "physics_loss_training_started": False,
        "topographic_loss": False,
        "dem_input": False,
        "raw_data_modified": False,
        "training_started": True,
        "training_completed": False,
        "is_resume": is_resume,
        "resume_checkpoint": str(args.resume) if is_resume else None,
        "resume_start_epoch": start_epoch if is_resume else None,
        "scheduler_restored": scheduler_restored,
        "started_at": now_utc(),
    }
    write_json(SUMMARY_JSON, summary)
    write_training_state(start_epoch - 1, best_miou, best_epoch, no_improve, optimizer, config)

    t_train_start = time.time()

    try:
        for epoch in range(start_epoch, max_epochs + 1):
            t_epoch = time.time()
            task.train()
            bn_count = set_bn_eval(task)
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            batches      = 0
            train_mat    = [[0, 0], [0, 0]]

            for batch_idx, batch in enumerate(train_loader, start=1):
                batch  = move_batch(batch, device)
                loss, logits = compute_loss(task, batch)
                scaled = loss / grad_accum

                if not torch.isfinite(loss).item():
                    raise RuntimeError(
                        f"Non-finite loss at epoch={epoch} batch={batch_idx}: {float(loss.detach().cpu())}"
                    )
                if not torch.isfinite(logits).all().item():
                    raise RuntimeError(f"Non-finite logits at epoch={epoch} batch={batch_idx}")

                scaled.backward()

                if batch_idx % grad_accum == 0 or batch_idx == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    set_bn_eval(task)

                pred = torch.argmax(logits.detach(), dim=1)
                train_mat    = add_conf(train_mat, confusion(batch["mask"], pred))
                running_loss += float(loss.detach().cpu())
                batches      += 1

            train_m = metrics_from_conf(train_mat)
            val_m   = evaluate_split(task, val_loader, device)
            cur_lr  = optimizer.param_groups[0]["lr"]
            scheduler.step(val_m["mean_iou"])
            next_lr = optimizer.param_groups[0]["lr"]

            improved = val_m["mean_iou"] > best_miou
            if improved:
                best_miou  = float(val_m["mean_iou"])
                best_epoch = epoch
                no_improve = 0
                save_ckpt(BEST_CKPT, task, optimizer, scheduler,
                          epoch, best_miou, best_epoch, no_improve, config)
            else:
                no_improve += 1

            # Always save last checkpoint every epoch
            save_ckpt(LAST_CKPT, task, optimizer, scheduler,
                      epoch, best_miou, best_epoch, no_improve, config)

            write_training_state(epoch, best_miou, best_epoch, no_improve, optimizer, config)

            row = {
                "epoch": epoch,
                "train_loss": running_loss / batches if batches else math.nan,
                "validation_loss": val_m["loss"],
                "train_miou": train_m["mean_iou"],
                "validation_miou": val_m["mean_iou"],
                "validation_iou_water": val_m["iou_water"],
                "validation_f1_water": val_m["f1_water"],
                "learning_rate": cur_lr,
                "learning_rate_after_scheduler": next_lr,
                "best_epoch": best_epoch,
                "epochs_without_improvement": no_improve,
                "batchnorm_eval_modules": bn_count,
                "precision": config["trainer"]["precision"],
                "elapsed_seconds": round(time.time() - t_epoch, 3),
                "improved": improved,
            }
            epoch_rows.append(row)
            # Append single row to CSV rather than rewriting entire file
            write_csv(EPOCH_CSV, [row], list(row.keys()))

            logging.info(
                "epoch=%d  train_loss=%.6f  val_loss=%.6f  val_miou=%.6f"
                "  val_iou_water=%.6f  val_f1_water=%.6f  lr=%.2e  next_lr=%.2e"
                "  no_improve=%d  best_epoch=%d  bn_eval=%d",
                epoch, row["train_loss"], row["validation_loss"], row["validation_miou"],
                row["validation_iou_water"], row["validation_f1_water"],
                cur_lr, next_lr, no_improve, best_epoch, bn_count,
            )

            summary.update({
                "last_epoch": epoch,
                "best_epoch": best_epoch,
                "best_validation_miou": best_miou,
                "training_elapsed_seconds": round(time.time() - t_train_start, 3),
            })
            write_json(SUMMARY_JSON, summary)

            if epoch >= es_min_epochs and no_improve >= es_patience:
                logging.info(
                    "Early stopping at epoch=%d: no improvement for %d epochs after epoch %d",
                    epoch, es_patience, best_epoch,
                )
                break

        evals = run_final_evals(task, config, device)
        summary.update({
            "status": "done",
            "training_completed": True,
            "completed_at": now_utc(),
            "evaluations": evals,
        })
        write_json(SUMMARY_JSON, summary)
        write_training_state(
            summary.get("last_epoch", start_epoch),
            best_miou, best_epoch, no_improve, optimizer, config,
        )
        logging.info(
            "STEP 5S-A bs2/accum4 complete. "
            "Human validation required before STEP 5S-B or STEP 6C."
        )

    except torch.cuda.OutOfMemoryError:
        logging.exception("CUDA OOM during training.")
        summary.update({"status": "oom", "training_completed": False, "blocked_at": now_utc()})
        write_json(SUMMARY_JSON, summary)
        logging.error(
            "OOM with batch_size=2. Fallback: relaunch with batch_size=1 grad_accum=8 "
            "in run directory: step5s_a_terramind_l_upernet_corrected_indices_dice_bs1_fallback"
        )
        return 3

    except RuntimeError as exc:
        err_str = str(exc)
        if "CUDA error" in err_str or "device-side assert" in err_str:
            # Windows WDDM TDR driver reset — CUDA is dead but Python process may still run.
            # Checkpoints were saved atomically before this batch started, so they are intact.
            logging.error(
                "CUDA driver error (likely Windows TDR reset): %s\n"
                "Checkpoints are atomically saved — last good state is on disk.\n"
                "Resume with: python %s --resume %s --log-file <new_log>",
                err_str, __file__, LAST_CKPT,
            )
            summary.update({"status": "tdr_crash", "training_completed": False, "blocked_at": now_utc(),
                            "tdr_error": err_str, "resume_from": str(LAST_CKPT)})
            write_json(SUMMARY_JSON, summary)
            return 2
        logging.exception("STEP 5S-A bs2/accum4 training failed.")
        summary.update({"status": "blocked", "training_completed": False, "blocked_at": now_utc()})
        write_json(SUMMARY_JSON, summary)
        return 1

    except Exception:
        logging.exception("STEP 5S-A bs2/accum4 training failed.")
        summary.update({"status": "blocked", "training_completed": False, "blocked_at": now_utc()})
        write_json(SUMMARY_JSON, summary)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
