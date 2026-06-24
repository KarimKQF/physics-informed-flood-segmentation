"""
STEP 6C v3 runner.

Goal: preserve the successful STEP 5S-A dataloader-side Albumentations D4 path
while adding aligned DEM to the batch for loss computation only.

DEM is returned as batch["topography"] and is never included in batch["image"].
Do not launch full training with this runner unless explicitly instructed.
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

import albumentations as A
import numpy as np
import rasterio
import torch
import yaml
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for path in (str(SRC_ROOT), str(SCRIPTS_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from losses.combined_loss import CombinedDicePhysicsLoss  # noqa: E402
from terratorch.datasets.generic_multimodal_dataset import GenericMultimodalSegmentationDataset  # noqa: E402

import step6c_lambda05_train as t6c  # noqa: E402


SPLIT_FILES = {
    "train": Path("E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/manifests/flood_train_step5e_filtered.txt"),
    "valid": Path("E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/manifests/flood_valid_step5e_filtered.txt"),
    "test": Path("E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/manifests/flood_test_step5e_filtered.txt"),
    "bolivia": Path("E:/flood_research/experiments/terramind_baseline/runs/step5e_tiny_unetdecoder_baseline/manifests/flood_bolivia_step5e_filtered.txt"),
}


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def path_text(path: Path) -> str:
    return str(path).replace("\\", "/")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return path_text(value)
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


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
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


def configure_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
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


def build_transform(transform_cfg: list[dict[str, Any]] | None, image_modalities: list[str], *, include_topography: bool) -> A.Compose | None:
    if not transform_cfg:
        return None
    transforms = []
    for item in transform_cfg:
        class_path = item.get("class_path") if isinstance(item, dict) else str(item)
        if class_path == "albumentations.D4":
            transforms.append(A.D4())
        elif class_path == "albumentations.pytorch.transforms.ToTensorV2":
            transforms.append(ToTensorV2())
        else:
            raise ValueError(f"Unsupported v3 transform: {class_path}")
    additional_targets = {modality: "image" for modality in image_modalities}
    if include_topography:
        additional_targets["topography"] = "image"
    return A.Compose(transforms, is_check_shapes=False, additional_targets=additional_targets)


def tile_id_from_mask_path(mask_path: str | Path) -> str:
    return Path(mask_path).stem.replace("_LabelHand", "")


def dem_path_for_sample(config: dict[str, Any], *, split: str, tile_id: str) -> Path:
    return Path(config["dem"]["aligned_dem_root"]) / str(config["dem"]["dem_filename_pattern"]).format(split=split, tile_id=tile_id)


def load_dem_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Aligned DEM missing: {path}")
    with rasterio.open(path) as dataset:
        return dataset.read(1).astype("float32")


def ensure_topography_chw(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        tensor = value
    else:
        array = np.asarray(value)
        if array.ndim == 2:
            array = array[None, :, :]
        elif array.ndim == 3 and array.shape[-1] == 1:
            array = np.moveaxis(array, -1, 0)
        tensor = torch.from_numpy(np.ascontiguousarray(array))
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3 and tensor.shape[-1] == 1:
        tensor = tensor.permute(2, 0, 1).contiguous()
    if tensor.ndim != 3 or tensor.shape[0] != 1:
        raise ValueError(f"topography must be [1,H,W], got {tuple(tensor.shape)}")
    return tensor.float().contiguous()


class TopographySegmentationDataset(GenericMultimodalSegmentationDataset):
    """Generic TerraTorch segmentation dataset with aligned DEM as a loss-only tensor."""

    def __init__(
        self,
        *args: Any,
        config: dict[str, Any],
        split_name: str,
        model_modalities: list[str],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        self.split_name = split_name
        self.model_modalities = list(model_modalities)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if isinstance(index, tuple):
            raise ValueError("STEP 6C v3 does not support sampled modality tuples.")

        sample = self.samples[index]
        output: dict[str, Any] = {}
        for modality, file in sample.items():
            data = self._load_file(
                file,
                nan_replace=self.no_label_replace if modality == "mask" else self.no_data_replace,
                modality=modality,
            )
            if modality == "mask" and not self.scalar_label:
                data = data[0]
            if modality in self.image_modalities and len(data.shape) >= 3 and self.channel_position:
                data = np.moveaxis(data, self.channel_position, -1)
            if modality in self.filter_indices:
                data = data[..., self.filter_indices[modality]]
            if modality in self.constant_scale:
                data = data.astype(np.float32) * self.constant_scale[modality]
            if data.dtype == np.float64:
                data = data.astype(np.float32)
            output[modality] = data

        if "mask" in output and self.reduce_zero_label:
            output["mask"] -= 1

        tile_id = tile_id_from_mask_path(sample["mask"])
        dem_map = self.config.get("dem", {}).get("dem_tile_id_map") or {}
        dem_tile_id = dem_map.get(tile_id, tile_id)  # shuffled control: map real tile to different DEM
        topo_path = dem_path_for_sample(self.config, split=self.split_name, tile_id=dem_tile_id)
        output["topography"] = load_dem_array(topo_path)[:, :, None]

        if self.transform:
            output = self.transform(output)

        topography = ensure_topography_chw(output.pop("topography"))
        image = {modality: output.pop(modality) for modality in self.model_modalities if modality in output}
        if "topography" in image:
            raise RuntimeError("DEM leaked into model input image dictionary.")

        output["image"] = image
        output["topography"] = topography
        output["mask"] = output["mask"].long()
        output["filename"] = {**sample, "topography": path_text(topo_path)}
        return output


class TopographyDataModule:
    """Small non-Lightning datamodule matching GenericMultiModalDataModule sampling."""

    def __init__(self, config: dict[str, Any], *, batch_size: int, split: str | None = None) -> None:
        self.config = config
        self.batch_size = batch_size
        self.split = split
        self.num_workers = 0
        self.pin_memory = True

    def _dataset(self, split_name: str, *, train: bool) -> TopographySegmentationDataset:
        args = dict(self.config["data"]["init_args"])
        modalities = list(args["modalities"])
        transform_cfg = args["train_transform"] if train else None
        transform = build_transform(transform_cfg, modalities, include_topography=True)
        data_root_key = "train_data_root" if split_name == "train" else ("val_data_root" if split_name == "valid" else "test_data_root")
        label_root_key = "train_label_data_root" if split_name == "train" else ("val_label_data_root" if split_name == "valid" else "test_label_data_root")
        split_file = args["train_split"] if split_name == "train" else (args["val_split"] if split_name == "valid" else args["test_split"])
        if split_name == "bolivia":
            split_file = str(SPLIT_FILES["bolivia"])
        return TopographySegmentationDataset(
            data_root=args[data_root_key],
            label_data_root=args[label_root_key],
            num_classes=args.get("num_classes"),
            image_grep=args["image_grep"],
            label_grep=args["label_grep"],
            split=Path(split_file),
            image_modalities=modalities,
            rgb_indices={args.get("rgb_modality", modalities[0]): args.get("rgb_indices", [0, 1, 2])},
            transform=transform,
            no_data_replace=args.get("no_data_replace"),
            no_label_replace=args.get("no_label_replace", -1),
            config=self.config,
            split_name=split_name,
            model_modalities=modalities,
        )

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = self._dataset("train", train=True)
            self.val_dataset = self._dataset("valid", train=False)
        elif stage == "test":
            split_name = self.split or "test"
            self.test_dataset = self._dataset(split_name, train=False)
        else:
            raise ValueError(f"Unsupported stage: {stage}")

    def _loader(self, dataset: Any, *, train: bool) -> DataLoader:
        sampler = RandomSampler(dataset) if train else SequentialSampler(dataset)
        batch_sampler = BatchSampler(sampler, batch_size=self.batch_size, drop_last=train)
        from terratorch.datamodules.generic_multimodal_data_module import collate_samples

        return DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            collate_fn=collate_samples,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, train=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, train=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, train=False)


class RunPaths:
    def __init__(self, config: dict[str, Any]) -> None:
        self.run_dir = Path(config["run_dir"])
        self.tag = config.get("run_tag") or self.run_dir.name
        self.metrics = self.run_dir / "metrics"
        self.logs = self.run_dir / "logs"
        self.checkpoints = self.run_dir / "checkpoints"
        self.configs = self.run_dir / "configs"
        self.scripts = self.run_dir / "scripts"
        self.epoch_csv = self.metrics / "training_epoch_metrics.csv"
        self.summary_json = self.metrics / f"{self.tag}_summary.json"
        self.metrics_json = self.metrics / "pure_dice_parity_metrics.json"
        self.final_csv = self.metrics / f"{self.tag}_final_metrics.csv"
        self.training_state = self.metrics / "training_state.json"
        self.best_ckpt = self.checkpoints / "best_checkpoint.pt"
        self.last_ckpt = self.checkpoints / "last_checkpoint.pt"
        self.log = self.logs / f"{self.tag}_training.log"

    def ensure_dirs(self) -> None:
        for path in [self.metrics, self.logs, self.checkpoints, self.configs, self.scripts]:
            path.mkdir(parents=True, exist_ok=True)


def build_loss(config: dict[str, Any]) -> CombinedDicePhysicsLoss:
    physics = config["physics_loss"]
    return CombinedDicePhysicsLoss(
        lambda_topo=float(physics["lambda_topo"]),
        ignore_index=int(physics["ignore_index"]),
        water_class=int(physics["water_class"]),
        dice_smooth=float(physics.get("dice_smooth", 0.0)),
        elevation_margin=float(physics["elevation_margin"]),
        elevation_scale=float(physics["elevation_scale"]),
        use_elevation_weight=bool(physics["use_elevation_weight"]),
        neighborhood=str(physics["neighborhood"]),
    )


def lambda_for_epoch(config: dict[str, Any], epoch: int) -> float:
    physics = config["physics_loss"]
    base = float(physics["lambda_topo"])
    schedule = physics.get("lambda_schedule") or {"type": "constant"}
    schedule_type = str(schedule.get("type", "constant")).lower()
    if schedule_type == "constant":
        return base
    if schedule_type == "warmup_linear":
        warmup_epochs = int(schedule.get("warmup_epochs", 0))
        ramp_epochs = int(schedule.get("ramp_epochs", 0))
        if epoch <= warmup_epochs:
            return 0.0
        if ramp_epochs <= 0 or epoch >= warmup_epochs + ramp_epochs:
            return base
        return base * ((epoch - warmup_epochs) / ramp_epochs)
    raise ValueError(f"Unsupported lambda schedule: {schedule_type}")


def compute_loss(task: Any, criterion: CombinedDicePhysicsLoss, batch: dict[str, Any]) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    if "topography" in batch["image"]:
        raise RuntimeError("DEM leaked into model input.")
    topography = batch["topography"]
    model_output = task(batch["image"])
    logits = t6c.get_logits(model_output)
    target = task.squeeze_ground_truth(batch["mask"]).long()
    return criterion(logits=logits, target=target, topography=topography), logits, target


def evaluate_split(task: Any, criterion: CombinedDicePhysicsLoss, loader: DataLoader, config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    task.eval()
    bn_count = t6c.set_bn_eval(task)
    totals = {"loss_total": 0.0, "loss_dice": 0.0, "loss_topo": 0.0}
    matrix = [[0, 0], [0, 0]]
    topo_counts = {"topo_descending_pair_count": 0, "topo_violation_pair_count": 0, "topo_violation_fraction": math.nan}
    batches = 0
    physics = config["physics_loss"]
    with torch.no_grad():
        for raw_batch in loader:
            batch = t6c.move_batch(raw_batch, device)
            losses, logits, target = compute_loss(task, criterion, batch)
            for key in totals:
                totals[key] += float(losses[key].detach().cpu())
            pred = torch.argmax(logits.detach(), dim=1)
            matrix = t6c.add_conf(matrix, t6c.confusion(target, pred))
            topo_counts = t6c.add_topo_counts(
                topo_counts,
                t6c.topographic_violation_counts(
                    logits=logits.detach(),
                    target=target,
                    topography=batch["topography"].squeeze(1),
                    ignore_index=int(physics["ignore_index"]),
                    water_class=int(physics["water_class"]),
                    elevation_margin=float(physics["elevation_margin"]),
                    neighborhood=str(physics["neighborhood"]),
                ),
            )
            batches += 1
    metrics = t6c.metrics_from_conf(matrix)
    for key, value in totals.items():
        metrics[key] = value / batches if batches else math.nan
    metrics["lambda_topo_loss"] = float(criterion.lambda_topo) * metrics["loss_topo"]
    metrics["water_pred_pixels"] = int(metrics["tp"]) + int(metrics["fp"])
    metrics["pred_water_fraction"] = metrics["water_pred_pixels"] / metrics["valid_pixel_count"] if metrics["valid_pixel_count"] else math.nan
    metrics["topographic_inconsistency_score"] = metrics["loss_topo"]
    metrics.update(topo_counts)
    metrics["batches"] = batches
    metrics["batchnorm_eval_modules"] = bn_count
    return metrics


def save_ckpt(path: Path, task: Any, optimizer: Any, scheduler: Any, epoch: int, best_miou: float, best_epoch: int, no_improve: int, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "step": "6C-v3",
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


def run(config_path: Path) -> tuple[dict[str, Any], RunPaths]:
    with config_path.open("r", encoding="utf-8-sig") as handle:
        config = yaml.safe_load(handle)
    # If a shuffled-DEM map file is specified, load and inject the mapping.
    dem_map_file = config.get("dem", {}).get("dem_tile_id_map_file")
    if dem_map_file:
        with open(dem_map_file, encoding="utf-8") as _f:
            _map_data = json.load(_f)
        config.setdefault("dem", {})["dem_tile_id_map"] = _map_data.get("mapping", {})
    paths = RunPaths(config)
    paths.ensure_dirs()
    configure_logging(paths.log)

    if any(path.exists() for path in [paths.best_ckpt, paths.last_ckpt, paths.epoch_csv, paths.metrics_json, paths.training_state]):
        raise RuntimeError(f"Refusing to overwrite existing run artifacts in {paths.run_dir}")

    seed = int(config.get("seed_everything", 42))
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    shutil.copy2(config_path, paths.configs / config_path.name)
    shutil.copy2(Path(__file__), paths.scripts / Path(__file__).name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(config["trainer"]["batch_size"])
    grad_accum = int(config["trainer"]["gradient_accumulation_steps"])
    max_epochs = int(config["trainer"]["max_epochs"])

    if max_epochs > 3 and "pure_dice_parity" in paths.tag:
        raise RuntimeError("Pure-Dice parity run is limited to 2 or 3 epochs.")
    if config["dem"].get("use_as_model_input", False):
        raise RuntimeError("DEM as model input is forbidden.")

    logging.info("STEP 6C v3 start: run_tag=%s run_dir=%s", paths.tag, paths.run_dir)
    logging.info("Dataloader-side D4 is active; DEM is batch['topography'], not model input.")
    logging.info("Device=%s batch_size=%d grad_accum=%d max_epochs=%d", device, batch_size, grad_accum, max_epochs)

    dm = TopographyDataModule(config, batch_size=batch_size)
    dm.setup("fit")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    task = t6c.build_task(config).to(device)
    criterion = build_loss(config).to(device)
    params = [param for param in task.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=float(config["optimizer"]["init_args"]["lr"]),
        weight_decay=float(config["optimizer"]["init_args"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_miou = -math.inf
    best_epoch = 0
    no_improve = 0
    epoch_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "step": "6C-v3",
        "run_tag": paths.tag,
        "status": "running",
        "run_dir": path_text(paths.run_dir),
        "config_path": path_text(config_path),
        "lambda_topo_base": float(config["physics_loss"]["lambda_topo"]),
        "dem_as_model_input": False,
        "dem_returned_in_batch": True,
        "dataloader_side_d4": True,
        "full_training": False,
        "started_at": now_utc(),
    }
    write_json(paths.metrics_json, summary)

    early_stopping_patience = int(config["trainer"].get("early_stopping_patience", max_epochs))
    early_stopping_min_epochs = int(config["trainer"].get("early_stopping_min_epochs", 0))

    train_start = time.time()
    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        epoch_lambda = lambda_for_epoch(config, epoch)
        criterion.set_lambda_topo(epoch_lambda)
        task.train()
        bn_count = t6c.set_bn_eval(task)
        optimizer.zero_grad(set_to_none=True)
        train_totals = {"loss_total": 0.0, "loss_dice": 0.0, "loss_topo": 0.0}
        train_matrix = [[0, 0], [0, 0]]
        max_grad_norm = 0.0
        batches = 0

        for batch_idx, raw_batch in enumerate(train_loader, start=1):
            batch = t6c.move_batch(raw_batch, device)
            losses, logits, target = compute_loss(task, criterion, batch)
            loss_total = losses["loss_total"]
            if not torch.isfinite(loss_total).item():
                raise RuntimeError(f"Non-finite loss at epoch={epoch} batch={batch_idx}")
            (loss_total / grad_accum).backward()
            if batch_idx % grad_accum == 0 or batch_idx == len(train_loader):
                grad_sq = 0.0
                for param in params:
                    if param.grad is not None:
                        grad_sq += float(param.grad.detach().pow(2).sum().item())
                max_grad_norm = max(max_grad_norm, math.sqrt(grad_sq))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                t6c.set_bn_eval(task)
            pred = torch.argmax(logits.detach(), dim=1)
            train_matrix = t6c.add_conf(train_matrix, t6c.confusion(target, pred))
            for key in train_totals:
                train_totals[key] += float(losses[key].detach().cpu())
            batches += 1

        train_metrics = t6c.metrics_from_conf(train_matrix)
        for key, value in train_totals.items():
            train_metrics[key] = value / batches if batches else math.nan
        train_metrics["lambda_topo_loss"] = epoch_lambda * train_metrics["loss_topo"]

        val_metrics = evaluate_split(task, criterion, val_loader, config, device)
        cur_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_metrics["mean_iou"])

        improved = val_metrics["mean_iou"] > best_miou
        if improved:
            best_miou = float(val_metrics["mean_iou"])
            best_epoch = epoch
            no_improve = 0
            save_ckpt(paths.best_ckpt, task, optimizer, scheduler, epoch, best_miou, best_epoch, no_improve, config)
        else:
            no_improve += 1
        save_ckpt(paths.last_ckpt, task, optimizer, scheduler, epoch, best_miou, best_epoch, no_improve, config)

        row = {
            "epoch": epoch,
            "lambda_topo_epoch": epoch_lambda,
            "train_loss_total": train_metrics["loss_total"],
            "train_loss_dice": train_metrics["loss_dice"],
            "train_loss_topo": train_metrics["loss_topo"],
            "train_lambda_topo_loss": train_metrics["lambda_topo_loss"],
            "train_miou": train_metrics["mean_iou"],
            "val_loss_total": val_metrics["loss_total"],
            "val_loss_dice": val_metrics["loss_dice"],
            "val_loss_topo": val_metrics["loss_topo"],
            "val_miou": val_metrics["mean_iou"],
            "val_iou_water": val_metrics["iou_water"],
            "val_f1_water": val_metrics["f1_water"],
            "val_water_pred_pixels": val_metrics["water_pred_pixels"],
            "val_pred_water_fraction": val_metrics["pred_water_fraction"],
            "val_topographic_inconsistency_score": val_metrics["topographic_inconsistency_score"],
            "val_topo_violation_fraction": val_metrics["topo_violation_fraction"],
            "learning_rate": cur_lr,
            "best_epoch": best_epoch,
            "best_validation_miou": best_miou,
            "no_improve": no_improve,
            "batchnorm_eval_modules": bn_count,
            "epoch_max_grad_norm": max_grad_norm,
            "elapsed_seconds": round(time.time() - epoch_start, 3),
            "improved": improved,
        }
        epoch_rows.append(row)
        write_csv(paths.epoch_csv, [row], list(row.keys()))
        write_json(
            paths.training_state,
            {
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_validation_miou": best_miou,
                "no_improve": no_improve,
                "current_lr": optimizer.param_groups[0]["lr"],
                "updated_at": now_utc(),
            },
        )
        logging.info(
            "epoch=%d lambda=%.4f train_dice=%.6f val_miou=%.6f val_iou_water=%.6f "
            "val_water_pred_pixels=%d grad_norm=%.6e bn_eval=%d",
            epoch,
            epoch_lambda,
            row["train_loss_dice"],
            row["val_miou"],
            row["val_iou_water"],
            row["val_water_pred_pixels"],
            max_grad_norm,
            bn_count,
        )

        if no_improve >= early_stopping_patience and epoch >= early_stopping_min_epochs:
            logging.info(
                "early_stop epoch=%d no_improve=%d patience=%d best_epoch=%d best_miou=%.6f",
                epoch, no_improve, early_stopping_patience, best_epoch, best_miou,
            )
            break

    last_two = epoch_rows[: min(2, len(epoch_rows))]
    parity_passed = (
        any(row["val_water_pred_pixels"] > 0 for row in last_two)
        and any(row["val_iou_water"] > 0.0 for row in epoch_rows)
        and all(row["epoch_max_grad_norm"] > 1e-6 for row in epoch_rows)
    )
    summary.update(
        {
            "status": "passed" if parity_passed else "failed",
            "completed_at": now_utc(),
            "training_elapsed_seconds": round(time.time() - train_start, 3),
            "best_epoch": best_epoch,
            "best_validation_miou": best_miou,
            "epoch_metrics": epoch_rows,
            "acceptance": {
                "no_all_background_collapse": any(row["val_water_pred_pixels"] > 0 for row in epoch_rows),
                "water_pixels_positive_by_epoch_1_or_2": any(row["val_water_pred_pixels"] > 0 for row in last_two),
                "val_iou_water_not_zero": any(row["val_iou_water"] > 0.0 for row in epoch_rows),
                "gradients_nonzero": all(row["epoch_max_grad_norm"] > 1e-6 for row in epoch_rows),
                "pure_dice_parity_passed": parity_passed,
            },
        }
    )
    write_json(paths.metrics_json, summary)
    return summary, paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary, _ = run(args.config)
    return 0 if summary["acceptance"]["pure_dice_parity_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
