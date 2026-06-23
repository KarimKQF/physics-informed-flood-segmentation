"""
STEP 6C lambda=0.5 training.

TerraMind-L + UPerNet corrected indices, Dice + 0.5 * TopographicInconsistencyLoss.
DEM is used only inside the loss and is never passed to the model input.

Resume support:
  python step6c_lambda05_train.py --resume <checkpoint_path> [--log-file <path>]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
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
import rasterio
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from losses.combined_loss import CombinedDicePhysicsLoss  # noqa: E402


RUN_DIR = Path("E:/flood_research/experiments/terramind_baseline/runs/step6c_terramind_l_upernet_dice_topographic_lambda05")
CONFIG_PATH = REPO_ROOT / "configs" / "step6c_terramind_l_upernet_dice_topographic_lambda05.yaml"
SPLIT_FILES = {
    "train": Path("E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/manifests/flood_train_step5e_filtered.txt"),
    "valid": Path("E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/manifests/flood_valid_step5e_filtered.txt"),
    "test": Path("E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/manifests/flood_test_step5e_filtered.txt"),
    "bolivia": Path("E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/manifests/flood_bolivia_step5e_filtered.txt"),
}

EPOCH_CSV = RUN_DIR / "metrics" / "training_epoch_metrics.csv"
SUMMARY_JSON = RUN_DIR / "metrics" / "step6c_lambda05_summary.json"
TRAINING_STATE = RUN_DIR / "metrics" / "training_state.json"
FINAL_JSON = RUN_DIR / "metrics" / "step6c_lambda05_final_metrics.json"
FINAL_CSV = RUN_DIR / "metrics" / "step6c_lambda05_final_metrics.csv"
BEST_CKPT = RUN_DIR / "checkpoints" / "best_checkpoint.pt"
LAST_CKPT = RUN_DIR / "checkpoints" / "last_checkpoint.pt"


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def ensure_dirs() -> None:
    for sub in ["logs", "checkpoints", "metrics", "configs", "scripts", "predictions/valid", "predictions/test", "predictions/bolivia"]:
        (RUN_DIR / sub).mkdir(parents=True, exist_ok=True)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value).replace("\\", "/")
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv_append(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a" if exists else "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def write_csv_overwrite(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def config_hash(config: dict[str, Any]) -> str:
    return hashlib.md5(json.dumps(json_safe(config), sort_keys=True).encode()).hexdigest()[:8]


def configure_logging(log_file: Path) -> None:
    ensure_dirs()
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(stream)
    root.addHandler(file_handler)


def set_bn_eval(module: Any) -> int:
    import torch.nn as nn

    count = 0
    for child in module.modules():
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            child.eval()
            for param in child.parameters(recurse=False):
                param.requires_grad = False
            count += 1
    return count


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            moved[key] = {
                inner_key: inner_value.to(device, non_blocking=True) if torch.is_tensor(inner_value) else inner_value
                for inner_key, inner_value in value.items()
            }
        elif torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def get_logits(model_output: Any) -> torch.Tensor:
    if hasattr(model_output, "output"):
        return model_output.output
    if isinstance(model_output, dict):
        return model_output.get("output", model_output.get("out", next(iter(model_output.values()))))
    if isinstance(model_output, (list, tuple)):
        return model_output[0]
    return model_output


def tile_id_from_batch(batch: dict[str, Any], index: int) -> str:
    mask_name = batch["filename"]["mask"][index]
    stem = Path(mask_name).stem
    return stem.replace("_LabelHand", "")


def dem_path_for_sample(config: dict[str, Any], *, split: str, tile_id: str) -> Path:
    dem_root = Path(config["dem"]["aligned_dem_root"])
    pattern = config["dem"]["dem_filename_pattern"]
    return dem_root / pattern.format(split=split, tile_id=tile_id)


def load_dem(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Aligned DEM missing: {path}")
    with rasterio.open(path) as dataset:
        array = dataset.read(1).astype("float32")
    return torch.from_numpy(array)


def load_dem_batch(config: dict[str, Any], batch: dict[str, Any], *, split: str) -> torch.Tensor:
    tile_ids = [tile_id_from_batch(batch, index) for index in range(len(batch["mask"]))]
    return torch.stack([load_dem(dem_path_for_sample(config, split=split, tile_id=tile_id)) for tile_id in tile_ids], dim=0)


def apply_d4(tensor: torch.Tensor, op: int) -> torch.Tensor:
    if op == 0:
        out = tensor
    elif op == 1:
        out = torch.rot90(tensor, 1, dims=(-2, -1))
    elif op == 2:
        out = torch.rot90(tensor, 2, dims=(-2, -1))
    elif op == 3:
        out = torch.rot90(tensor, 3, dims=(-2, -1))
    elif op == 4:
        out = torch.flip(tensor, dims=(-1,))
    elif op == 5:
        out = torch.flip(tensor, dims=(-2,))
    elif op == 6:
        out = tensor.transpose(-2, -1)
    elif op == 7:
        out = torch.flip(tensor.transpose(-2, -1), dims=(-2, -1))
    else:
        raise ValueError(f"Unsupported D4 op: {op}")
    return out.contiguous()


def apply_d4_to_batch(batch: dict[str, Any], dem: torch.Tensor, rng: random.Random) -> None:
    for index in range(len(batch["mask"])):
        op = rng.randrange(8)
        for modality in batch["image"]:
            batch["image"][modality][index] = apply_d4(batch["image"][modality][index], op)
        batch["mask"][index] = apply_d4(batch["mask"][index], op)
        dem[index] = apply_d4(dem[index], op)


def crop_pair(tensor: torch.Tensor, dy: int, dx: int) -> tuple[torch.Tensor, torch.Tensor]:
    height = tensor.shape[-2]
    width = tensor.shape[-1]
    y_a_start = max(0, -dy)
    y_a_end = height - max(0, dy)
    x_a_start = max(0, -dx)
    x_a_end = width - max(0, dx)
    y_b_start = max(0, dy)
    y_b_end = height - max(0, -dy)
    x_b_start = max(0, dx)
    x_b_end = width - max(0, -dx)
    return (
        tensor[:, y_a_start:y_a_end, x_a_start:x_a_end],
        tensor[:, y_b_start:y_b_end, x_b_start:x_b_end],
    )


def neighbor_offsets(neighborhood: str) -> tuple[tuple[int, int], ...]:
    if neighborhood == "4":
        return ((0, 1), (1, 0))
    if neighborhood == "8":
        return ((0, 1), (1, 0), (1, 1), (1, -1))
    raise ValueError(f"Unsupported neighborhood: {neighborhood}")


@torch.no_grad()
def topographic_violation_counts(
    *,
    logits: torch.Tensor,
    target: torch.Tensor,
    topography: torch.Tensor,
    ignore_index: int,
    water_class: int,
    elevation_margin: float,
    neighborhood: str,
) -> dict[str, float | int]:
    pred = torch.argmax(logits, dim=1)
    valid_pixel = (target != ignore_index) & torch.isfinite(topography)
    descending_count = 0.0
    violation_count = 0.0

    def add_direction(pred_high: torch.Tensor, pred_low: torch.Tensor, h_high: torch.Tensor, h_low: torch.Tensor, valid_pair: torch.Tensor) -> None:
        nonlocal descending_count, violation_count
        descending = valid_pair & ((h_high - h_low - elevation_margin) > 0)
        descending_count += float(descending.sum().detach().cpu())
        violation = descending & (pred_high == water_class) & (pred_low != water_class)
        violation_count += float(violation.sum().detach().cpu())

    for dy, dx in neighbor_offsets(neighborhood):
        pred_a, pred_b = crop_pair(pred, dy, dx)
        h_a, h_b = crop_pair(topography, dy, dx)
        valid_a, valid_b = crop_pair(valid_pixel, dy, dx)
        valid_pair = valid_a & valid_b
        add_direction(pred_a, pred_b, h_a, h_b, valid_pair)
        add_direction(pred_b, pred_a, h_b, h_a, valid_pair)

    return {
        "topo_descending_pair_count": int(descending_count),
        "topo_violation_pair_count": int(violation_count),
        "topo_violation_fraction": violation_count / descending_count if descending_count > 0 else math.nan,
    }


def add_topo_counts(a: dict[str, float | int], b: dict[str, float | int]) -> dict[str, float | int]:
    descending = int(a.get("topo_descending_pair_count", 0)) + int(b.get("topo_descending_pair_count", 0))
    violations = int(a.get("topo_violation_pair_count", 0)) + int(b.get("topo_violation_pair_count", 0))
    return {
        "topo_descending_pair_count": descending,
        "topo_violation_pair_count": violations,
        "topo_violation_fraction": violations / descending if descending > 0 else math.nan,
    }


def confusion(y_true: torch.Tensor, y_pred: torch.Tensor) -> list[list[int]]:
    valid = y_true != -1
    yt = y_true[valid].long()
    yp = y_pred[valid].long()
    mat = torch.zeros((2, 2), dtype=torch.int64, device=yt.device)
    if yt.numel() > 0:
        mat = torch.bincount(yt * 2 + yp, minlength=4).reshape(2, 2)
    return [[int(x) for x in row] for row in mat.cpu().tolist()]


def add_conf(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
    return [[a[0][0] + b[0][0], a[0][1] + b[0][1]], [a[1][0] + b[1][0], a[1][1] + b[1][1]]]


def metrics_from_conf(mat: list[list[int]]) -> dict[str, Any]:
    tn, fp = mat[0]
    fn, tp = mat[1]
    total = tn + fp + fn + tp
    precision = tp / (tp + fp) if (tp + fp) else math.nan
    recall = tp / (tp + fn) if (tp + fn) else math.nan
    f1 = 2 * precision * recall / (precision + recall) if math.isfinite(precision + recall) and (precision + recall) else math.nan
    iou_bg = tn / (tn + fp + fn) if (tn + fp + fn) else math.nan
    iou_water = tp / (tp + fp + fn) if (tp + fp + fn) else math.nan
    miou = (iou_bg + iou_water) / 2 if math.isfinite(iou_bg) and math.isfinite(iou_water) else math.nan
    return {
        "accuracy": (tn + tp) / total if total else math.nan,
        "precision_water": precision,
        "recall_water": recall,
        "f1_water": f1,
        "iou_background": iou_bg,
        "iou_water": iou_water,
        "mean_iou": miou,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "support_background": tn + fp,
        "support_water": fn + tp,
        "valid_pixel_count": total,
    }


def build_datamodule(config: dict[str, Any], *, split: str | None = None, batch_size: int = 2, train_aug: bool = True):
    import albumentations as A
    from albumentations.pytorch.transforms import ToTensorV2
    from terratorch.datamodules import GenericMultiModalDataModule

    args = dict(config["data"]["init_args"])
    args["batch_size"] = batch_size
    args["num_workers"] = 0
    args["pin_memory"] = True
    if train_aug:
        # D4 is applied manually to image/mask/DEM together inside the train loop.
        args["train_transform"] = [ToTensorV2()]
    else:
        args["train_transform"] = [A.D4(), ToTensorV2()]
    args["val_transform"] = None
    args["test_transform"] = None
    if split in {"valid", "test", "bolivia"}:
        args["test_split"] = str(SPLIT_FILES[split])
    return GenericMultiModalDataModule(**args)


def build_task(config: dict[str, Any]):
    from terratorch.tasks import SemanticSegmentationTask

    model_init = config["model"]["init_args"]
    return SemanticSegmentationTask(
        model_factory=model_init["model_factory"],
        model_args=model_init["model_args"],
        loss=model_init["loss"],
        ignore_index=model_init["ignore_index"],
        freeze_backbone=model_init.get("freeze_backbone", False),
        freeze_decoder=model_init.get("freeze_decoder", False),
        class_names=model_init.get("class_names", ["Others", "Flood"]),
    )


def build_loss(config: dict[str, Any]) -> CombinedDicePhysicsLoss:
    physics = config["physics_loss"]
    return CombinedDicePhysicsLoss(
        lambda_topo=float(physics["lambda_topo"]),
        ignore_index=int(physics["ignore_index"]),
        water_class=int(physics["water_class"]),
        elevation_margin=float(physics["elevation_margin"]),
        elevation_scale=float(physics["elevation_scale"]),
        use_elevation_weight=bool(physics["use_elevation_weight"]),
        neighborhood=str(physics["neighborhood"]),
    )


def compute_loss(task: Any, criterion: CombinedDicePhysicsLoss, batch: dict[str, Any], dem: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    model_out = task(batch["image"])
    logits = get_logits(model_out)
    target = task.squeeze_ground_truth(batch["mask"]).long()
    losses = criterion(logits=logits, target=target, topography=dem)
    return losses, logits, target


def evaluate_split(
    *,
    task: Any,
    criterion: CombinedDicePhysicsLoss,
    loader: Any,
    config: dict[str, Any],
    split: str,
    device: torch.device,
) -> dict[str, Any]:
    task.eval()
    bn_count = set_bn_eval(task)
    totals = {"loss_total": 0.0, "loss_dice": 0.0, "loss_topo": 0.0}
    batches = 0
    matrix = [[0, 0], [0, 0]]
    topo_counts = {"topo_descending_pair_count": 0, "topo_violation_pair_count": 0, "topo_violation_fraction": math.nan}
    physics = config["physics_loss"]

    with torch.no_grad():
        for raw_batch in loader:
            dem_cpu = load_dem_batch(config, raw_batch, split=split)
            batch = move_batch(raw_batch, device)
            dem = dem_cpu.to(device, non_blocking=True)
            losses, logits, target = compute_loss(task, criterion, batch, dem)
            for key in totals:
                totals[key] += float(losses[key].detach().cpu())
            pred = torch.argmax(logits.detach(), dim=1)
            matrix = add_conf(matrix, confusion(target, pred))
            topo_counts = add_topo_counts(
                topo_counts,
                topographic_violation_counts(
                    logits=logits.detach(),
                    target=target,
                    topography=dem,
                    ignore_index=int(physics["ignore_index"]),
                    water_class=int(physics["water_class"]),
                    elevation_margin=float(physics["elevation_margin"]),
                    neighborhood=str(physics["neighborhood"]),
                ),
            )
            batches += 1

    metrics = metrics_from_conf(matrix)
    for key, value in totals.items():
        metrics[key] = value / batches if batches else math.nan
    metrics["lambda_topo_loss"] = float(criterion.lambda_topo) * metrics["loss_topo"]
    metrics.update(topo_counts)
    metrics["topographic_inconsistency_score"] = metrics["loss_topo"]
    metrics["batches"] = batches
    metrics["batchnorm_eval_modules"] = bn_count
    return metrics


def save_ckpt(
    path: Path,
    task: Any,
    optimizer: Any,
    scheduler: Any,
    epoch: int,
    best_miou: float,
    best_epoch: int,
    no_improve: int,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "step": "6C-lambda05",
            "ckpt_version": 1,
            "epoch": epoch,
            "best_validation_miou": best_miou,
            "best_epoch": best_epoch,
            "no_improve": no_improve,
            "model_state_dict": task.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config,
            "saved_at": now_utc(),
        },
        tmp,
    )
    os.replace(tmp, path)


def write_training_state(epoch: int, best_miou: float, best_epoch: int, no_improve: int, optimizer: Any, config: dict[str, Any]) -> None:
    write_json(
        TRAINING_STATE,
        {
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_validation_miou": best_miou,
            "no_improve": no_improve,
            "current_lr": optimizer.param_groups[0]["lr"],
            "config_hash": config_hash(config),
            "updated_at": now_utc(),
        },
    )


def load_checkpoint(path: Path, task: Any, optimizer: Any, scheduler: Any, device: torch.device) -> tuple[int, float, int, int]:
    logging.info("Loading checkpoint: %s", path)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    task.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    epoch = int(ckpt["epoch"])
    return epoch + 1, float(ckpt["best_validation_miou"]), int(ckpt.get("best_epoch", epoch)), int(ckpt.get("no_improve", 0))


def run_final_evals(task: Any, criterion: CombinedDicePhysicsLoss, config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    logging.info("Loading best checkpoint for final evaluation: %s", BEST_CKPT)
    ckpt = torch.load(BEST_CKPT, map_location=device, weights_only=False)
    task.load_state_dict(ckpt["model_state_dict"])
    evals: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    for split in ["valid", "test", "bolivia"]:
        dm = build_datamodule(config, split=split, batch_size=int(config["trainer"]["batch_size"]))
        dm.setup("test")
        metrics = evaluate_split(task=task, criterion=criterion, loader=dm.test_dataloader(), config=config, split=split, device=device)
        evals[split] = metrics
        csv_rows.append({"split": split, **metrics})
        logging.info(
            "final_%s mIoU=%.6f iou_water=%.6f f1_water=%.6f loss_total=%.6f loss_dice=%.6f loss_topo=%.8f topo_violation_fraction=%.8f",
            split,
            metrics["mean_iou"],
            metrics["iou_water"],
            metrics["f1_water"],
            metrics["loss_total"],
            metrics["loss_dice"],
            metrics["loss_topo"],
            metrics["topo_violation_fraction"],
        )
    write_json(FINAL_JSON, {"step": "6C-lambda05", "generated_at": now_utc(), "evaluations": evals})
    write_csv_overwrite(FINAL_CSV, csv_rows, list(csv_rows[0].keys()))
    return evals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    is_resume = args.resume is not None
    log_file = args.log_file if args.log_file else RUN_DIR / "logs" / "step6c_lambda05_training.log"

    ensure_dirs()
    configure_logging(log_file)
    logging.info("=" * 72)
    logging.info("STEP 6C lambda=0.5: TerraMind-L + UPerNet Dice + TopographicInconsistencyLoss")
    logging.info("Guardrails: DEM loss-only, no DEM model input, no DARN, no STURM, raw data unchanged")
    logging.info("=" * 72)

    if not is_resume and any(path.exists() for path in [BEST_CKPT, LAST_CKPT, EPOCH_CSV, TRAINING_STATE, FINAL_JSON]):
        raise RuntimeError(f"Refusing to overwrite existing run artifacts in {RUN_DIR}")

    with args.config.open("r", encoding="utf-8-sig") as handle:
        config = yaml.safe_load(handle)

    torch.manual_seed(int(config.get("seed_everything", 42)))
    random.seed(int(config.get("seed_everything", 42)))
    np.random.seed(int(config.get("seed_everything", 42)))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config.get("seed_everything", 42)))

    shutil.copy2(args.config, RUN_DIR / "configs" / args.config.name)
    shutil.copy2(Path(__file__), RUN_DIR / "scripts" / Path(__file__).name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Device: %s", device)
    if device.type == "cuda":
        logging.info("GPU: %s VRAM=%.0f MB", torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 1024**2)

    batch_size = int(config["trainer"]["batch_size"])
    grad_accum = int(config["trainer"]["gradient_accumulation_steps"])
    max_epochs = int(config["trainer"]["max_epochs"])
    es_patience = int(config["trainer"]["early_stopping_patience"])
    es_min_epochs = int(config["trainer"]["early_stopping_min_epochs"])
    lambda_topo = float(config["physics_loss"]["lambda_topo"])

    if lambda_topo != 0.5:
        raise ValueError(f"STEP 6C lambda05 runner requires lambda_topo=0.5, found {lambda_topo}")
    if config["dem"].get("use_as_model_input", False):
        raise ValueError("DEM as model input is forbidden for STEP 6C.")

    dm = build_datamodule(config, batch_size=batch_size, train_aug=True)
    dm.setup("fit")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    task = build_task(config).to(device)
    initial_bn_count = set_bn_eval(task)
    criterion = build_loss(config).to(device)
    params = [param for param in task.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=float(config["optimizer"]["init_args"]["lr"]),
        weight_decay=float(config["optimizer"]["init_args"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    start_epoch = 1
    best_miou = -math.inf
    best_epoch = 0
    no_improve = 0
    if is_resume:
        start_epoch, best_miou, best_epoch, no_improve = load_checkpoint(args.resume, task, optimizer, scheduler, device)

    summary: dict[str, Any] = {
        "step": "6C-lambda05",
        "status": "running",
        "run_dir": str(RUN_DIR),
        "config_path": str(args.config),
        "training_started": True,
        "training_completed": False,
        "lambda_topo": lambda_topo,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "effective_batch_size": batch_size * grad_accum,
        "precision": config["trainer"]["precision"],
        "dem_as_model_input": False,
        "dem_in_loss_only": True,
        "darn_started": False,
        "sturm_training_started": False,
        "raw_data_modified": False,
        "initialization": "original TerraMind pretrained checkpoint",
        "batchnorm_eval_modules_initial": initial_bn_count,
        "started_at": now_utc(),
        "is_resume": is_resume,
        "resume_checkpoint": str(args.resume) if is_resume else None,
    }
    write_json(SUMMARY_JSON, summary)
    write_training_state(start_epoch - 1, best_miou, best_epoch, no_improve, optimizer, config)

    d4_rng = random.Random(int(config.get("seed_everything", 42)))
    train_start = time.time()

    try:
        for epoch in range(start_epoch, max_epochs + 1):
            epoch_start = time.time()
            task.train()
            bn_count = set_bn_eval(task)
            optimizer.zero_grad(set_to_none=True)
            matrix = [[0, 0], [0, 0]]
            train_totals = {"loss_total": 0.0, "loss_dice": 0.0, "loss_topo": 0.0}
            batches = 0

            for batch_idx, raw_batch in enumerate(train_loader, start=1):
                dem_cpu = load_dem_batch(config, raw_batch, split="train")
                apply_d4_to_batch(raw_batch, dem_cpu, d4_rng)
                batch = move_batch(raw_batch, device)
                dem = dem_cpu.to(device, non_blocking=True)

                losses, logits, target = compute_loss(task, criterion, batch, dem)
                loss_total = losses["loss_total"]
                if not torch.isfinite(loss_total).item():
                    raise RuntimeError(f"Non-finite total loss at epoch={epoch} batch={batch_idx}: {float(loss_total.detach().cpu())}")
                if not torch.isfinite(logits).all().item():
                    raise RuntimeError(f"Non-finite logits at epoch={epoch} batch={batch_idx}")

                (loss_total / grad_accum).backward()
                if batch_idx % grad_accum == 0 or batch_idx == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    set_bn_eval(task)

                pred = torch.argmax(logits.detach(), dim=1)
                matrix = add_conf(matrix, confusion(target, pred))
                for key in train_totals:
                    train_totals[key] += float(losses[key].detach().cpu())
                batches += 1

            train_m = metrics_from_conf(matrix)
            for key, value in train_totals.items():
                train_m[key] = value / batches if batches else math.nan
            train_m["lambda_topo_loss"] = lambda_topo * train_m["loss_topo"]
            train_m["batches"] = batches

            val_m = evaluate_split(task=task, criterion=criterion, loader=val_loader, config=config, split="valid", device=device)
            cur_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_m["mean_iou"])
            next_lr = optimizer.param_groups[0]["lr"]

            improved = val_m["mean_iou"] > best_miou
            if improved:
                best_miou = float(val_m["mean_iou"])
                best_epoch = epoch
                no_improve = 0
                save_ckpt(BEST_CKPT, task, optimizer, scheduler, epoch, best_miou, best_epoch, no_improve, config)
            else:
                no_improve += 1
            save_ckpt(LAST_CKPT, task, optimizer, scheduler, epoch, best_miou, best_epoch, no_improve, config)
            write_training_state(epoch, best_miou, best_epoch, no_improve, optimizer, config)

            row = {
                "epoch": epoch,
                "train_loss_total": train_m["loss_total"],
                "train_loss_dice": train_m["loss_dice"],
                "train_loss_topo": train_m["loss_topo"],
                "train_lambda_topo_loss": train_m["lambda_topo_loss"],
                "train_miou": train_m["mean_iou"],
                "val_loss_total": val_m["loss_total"],
                "val_loss_dice": val_m["loss_dice"],
                "val_loss_topo": val_m["loss_topo"],
                "val_lambda_topo_loss": val_m["lambda_topo_loss"],
                "val_miou": val_m["mean_iou"],
                "val_iou_water": val_m["iou_water"],
                "val_f1_water": val_m["f1_water"],
                "val_topographic_inconsistency_score": val_m["topographic_inconsistency_score"],
                "val_topo_descending_pair_count": val_m["topo_descending_pair_count"],
                "val_topo_violation_pair_count": val_m["topo_violation_pair_count"],
                "val_topo_violation_fraction": val_m["topo_violation_fraction"],
                "learning_rate": cur_lr,
                "learning_rate_after_scheduler": next_lr,
                "best_epoch": best_epoch,
                "best_validation_miou": best_miou,
                "no_improve": no_improve,
                "batchnorm_eval_modules": bn_count,
                "precision": config["trainer"]["precision"],
                "elapsed_seconds": round(time.time() - epoch_start, 3),
                "improved": improved,
            }
            write_csv_append(EPOCH_CSV, [row], list(row.keys()))
            logging.info(
                "epoch=%d train_loss_total=%.6f train_loss_dice=%.6f train_loss_topo=%.8f train_lambda_topo_loss=%.8f "
                "val_loss_total=%.6f val_loss_dice=%.6f val_loss_topo=%.8f val_miou=%.6f val_iou_water=%.6f "
                "val_f1_water=%.6f val_topo_score=%.8f val_topo_violation_fraction=%.8f lr=%.2e best_epoch=%d "
                "no_improve=%d bn_eval=%d",
                epoch,
                row["train_loss_total"],
                row["train_loss_dice"],
                row["train_loss_topo"],
                row["train_lambda_topo_loss"],
                row["val_loss_total"],
                row["val_loss_dice"],
                row["val_loss_topo"],
                row["val_miou"],
                row["val_iou_water"],
                row["val_f1_water"],
                row["val_topographic_inconsistency_score"],
                row["val_topo_violation_fraction"],
                cur_lr,
                best_epoch,
                no_improve,
                bn_count,
            )

            summary.update(
                {
                    "last_epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_validation_miou": best_miou,
                    "training_elapsed_seconds": round(time.time() - train_start, 3),
                }
            )
            write_json(SUMMARY_JSON, summary)

            if epoch >= es_min_epochs and no_improve >= es_patience:
                logging.info("Early stopping at epoch=%d: no improvement for %d epochs after epoch %d", epoch, es_patience, best_epoch)
                break

        evals = run_final_evals(task, criterion, config, device)
        summary.update({"status": "done", "training_completed": True, "completed_at": now_utc(), "evaluations": evals})
        write_json(SUMMARY_JSON, summary)
        write_training_state(summary.get("last_epoch", start_epoch), best_miou, best_epoch, no_improve, optimizer, config)
        logging.info("STEP 6C lambda=0.5 complete. Human validation required before STEP 6D.")

    except torch.cuda.OutOfMemoryError:
        logging.exception("CUDA OOM during STEP 6C lambda=0.5 training.")
        summary.update({"status": "oom", "training_completed": False, "blocked_at": now_utc()})
        write_json(SUMMARY_JSON, summary)
        return 3
    except RuntimeError as exc:
        err_str = str(exc)
        if "CUDA error" in err_str or "device-side assert" in err_str:
            logging.error("CUDA driver error: %s. Resume from %s if checkpoint exists.", err_str, LAST_CKPT)
            summary.update({"status": "cuda_error", "training_completed": False, "blocked_at": now_utc(), "error": err_str, "resume_from": str(LAST_CKPT)})
            write_json(SUMMARY_JSON, summary)
            return 2
        logging.exception("STEP 6C lambda=0.5 training failed.")
        summary.update({"status": "blocked", "training_completed": False, "blocked_at": now_utc(), "error": err_str})
        write_json(SUMMARY_JSON, summary)
        return 1
    except Exception as exc:
        logging.exception("STEP 6C lambda=0.5 training failed.")
        summary.update({"status": "blocked", "training_completed": False, "blocked_at": now_utc(), "error": repr(exc)})
        write_json(SUMMARY_JSON, summary)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
