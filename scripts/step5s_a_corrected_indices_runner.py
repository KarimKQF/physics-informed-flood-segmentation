"""STEP 5S-A TerraMind-L UPerNet corrected-indices runner.

This runner performs classical segmentation only. It never uses DEM inputs,
topographic loss, physics loss, DARN, or STURM-Flood.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

BASELINE_ROOT = Path("E:/flood_research/experiments/terramind_baseline")
RUNS_ROOT = BASELINE_ROOT / "runs"
RUN_DIR = RUNS_ROOT / "step5s_a_terramind_l_upernet_corrected_indices_dice"
BASE_CONFIG = RUNS_ROOT / "step5p_terramind_l_upernet_big_classical_training" / "configs" / "terramind_l_upernet_big_classical_train.yaml"
REPO_CONFIG = REPO_ROOT / "configs" / "step5s_a_terramind_l_upernet_corrected_indices_dice.yaml"
RUN_CONFIG = RUN_DIR / "configs" / "step5s_a_terramind_l_upernet_corrected_indices_dice.yaml"
REPORT_PATH = REPO_ROOT / "reports" / "STEP_5S_A_terramind_l_upernet_corrected_indices_dice_report.md"
RUN_REPORT_PATH = RUN_DIR / "reports" / REPORT_PATH.name
SMOKE_JSON = RUN_DIR / "metrics" / "step5s_a_corrected_indices_smoke_summary.json"
SUMMARY_JSON = RUN_DIR / "metrics" / "step5s_a_summary.json"
TRAINING_LOG = RUN_DIR / "logs" / "step5s_a_training.log"
LAUNCH_INFO_JSON = RUN_DIR / "metadata" / "step5s_a_launch_info.json"
PIPELINE_STATUS = REPO_ROOT / "pipeline_status.json"
EVAL_SCRIPT = REPO_ROOT / "scripts" / "05_evaluate_predictions.py"
MANIFEST_CSV = RUNS_ROOT / "step5e_tiny_unetdecoder_baseline" / "manifests" / "sen1floods11_handlabeled_index_e_paths.csv"

SPLIT_FILES = {
    "train": RUNS_ROOT / "step5e_tiny_unetdecoder_baseline" / "manifests" / "flood_train_step5e_filtered.txt",
    "valid": RUNS_ROOT / "step5e_tiny_unetdecoder_baseline" / "manifests" / "flood_valid_step5e_filtered.txt",
    "test": RUNS_ROOT / "step5e_tiny_unetdecoder_baseline" / "manifests" / "flood_test_step5e_filtered.txt",
    "bolivia": RUNS_ROOT / "step5e_tiny_unetdecoder_baseline" / "manifests" / "flood_bolivia_step5e_filtered.txt",
}

EXCLUDED_TILE_IDS = ["Ghana_234935", "Ghana_26376", "Ghana_277", "Ghana_5079", "Ghana_83483"]
CORRECTED_INDICES = [5, 11, 17, 23]
OLD_INDICES = [2, 5, 8, 11]


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def path_text(path: Path) -> str:
    return str(path).replace("\\", "/")


def ensure_dirs() -> None:
    for subdir in [
        "configs",
        "logs",
        "checkpoints",
        "predictions",
        "predictions/valid",
        "predictions/test",
        "predictions/bolivia",
        "metrics",
        "reports",
        "scripts",
        "figures",
        "metadata",
    ]:
        (RUN_DIR / subdir).mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "configs").mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "reports").mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def json_safe(value: Any) -> Any:
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        np = None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if np is not None and isinstance(value, np.integer):
        return int(value)
    if np is not None and isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return path_text(value)
    return value


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return yaml.safe_load(handle) or {}


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return len([line for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()])


def update_pipeline_status(**updates: Any) -> None:
    status = read_json(PIPELINE_STATUS)
    status.update(
        {
            "current_step": "5S-A",
            "physics_loss_training_started": False,
            "physics_loss_started": False,
            "topographic_alignment_validated": True,
            "full_topographic_alignment_completed": True,
            "darn_started": False,
            "sturm_started": False,
            "sturm_training_started": False,
            "raw_data_modified": False,
            "official_split_files_modified": False,
            "next_step_allowed": False,
            "human_validation_required": True,
            "step5s_a_run_dir": path_text(RUN_DIR),
            "step5s_a_config": path_text(REPO_CONFIG),
            "step5s_a_run_config": path_text(RUN_CONFIG),
            "step5s_a_report": path_text(REPORT_PATH),
            "step5s_a_run_report": path_text(RUN_REPORT_PATH),
            "step5s_a_log": path_text(TRAINING_LOG),
            "generated_at": now_utc(),
        }
    )
    status.update(updates)
    write_json(PIPELINE_STATUS, status)


def build_config() -> dict[str, Any]:
    base = read_yaml(BASE_CONFIG)
    config = {
        "step": "5S-A",
        "status": "prepared_not_launched",
        "seed_everything": 42,
        "run_dir": path_text(RUN_DIR),
        "guardrails": {
            "classical_segmentation_only": True,
            "physics_loss_training_started": False,
            "topographic_loss": False,
            "dem_input": False,
            "darn_started": False,
            "sturm_training_started": False,
            "raw_data_modified": False,
            "official_split_files_modified": False,
        },
        "trainer": {
            "accelerator": "gpu",
            "devices": 1,
            "precision": "32",
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "effective_batch_size": 8,
            "max_epochs": 80,
            "early_stopping_monitor": "validation_miou",
            "early_stopping_mode": "max",
            "early_stopping_patience": 15,
            "early_stopping_min_epochs": 30,
            "log_every_n_steps": 10,
            "batchnorm_eval_policy": "BatchNorm modules kept in eval mode and affine parameters frozen because UPerNet PSP pool scale 1 produces 1x1 features at batch size 1.",
        },
        "data": base["data"],
        "model": base["model"],
        "optimizer": {
            "class_path": "torch.optim.AdamW",
            "init_args": {
                "lr": 2e-5,
                "weight_decay": 1e-4,
            },
        },
        "lr_scheduler": {
            "class_path": "torch.optim.lr_scheduler.ReduceLROnPlateau",
            "init_args": {
                "mode": "max",
                "factor": 0.5,
                "patience": 3,
            },
            "monitor": "validation_miou",
        },
        "dataset_policy": {
            "dataset": "Sen1Floods11",
            "train": count_lines(SPLIT_FILES["train"]),
            "valid": count_lines(SPLIT_FILES["valid"]),
            "test": count_lines(SPLIT_FILES["test"]),
            "bolivia": count_lines(SPLIT_FILES["bolivia"]),
            "excluded_fully_invalid_tile_ids": EXCLUDED_TILE_IDS,
            "keep_no_water": True,
            "keep_warning_review": True,
            "ignore_index": -1,
            "water_class_index": 1,
        },
        "comparison": {
            "old_indices": OLD_INDICES,
            "corrected_large_backbone_indices": CORRECTED_INDICES,
            "baseline_targets": ["STEP_5O", "STEP_5I"],
            "secondary_reference": "STEP_5P",
        },
        "stability_notes": {
            "mixed_precision_preference": "16-mixed was preferred by the brief if stable.",
            "mixed_precision_observation": "Initial 16-mixed background attempt produced non-finite epoch losses, so STEP 5S-A uses fp32 like STEP 5O/5P.",
        },
    }

    data_args = config["data"]["init_args"]
    data_args["batch_size"] = 1
    data_args["num_workers"] = 0
    data_args["pin_memory"] = True
    data_args["train_split"] = path_text(SPLIT_FILES["train"])
    data_args["val_split"] = path_text(SPLIT_FILES["valid"])
    data_args["test_split"] = path_text(SPLIT_FILES["test"])
    data_args["train_transform"] = [
        {"class_path": "albumentations.D4"},
        {"class_path": "albumentations.pytorch.transforms.ToTensorV2"},
    ]
    data_args["val_transform"] = None
    data_args["test_transform"] = None

    model_init = config["model"]["init_args"]
    model_init["loss"] = "dice"
    model_init["ignore_index"] = -1
    model_init["freeze_backbone"] = False
    model_init["freeze_decoder"] = False
    model_args = model_init["model_args"]
    model_args["backbone"] = "terramind_v1_large"
    model_args["decoder"] = "UperNetDecoder"
    model_args["num_classes"] = 2
    model_args["backbone_modalities"] = ["S2L1C", "S1GRD"]
    model_args["backbone_merge_method"] = "mean"
    model_args["backbone_ckpt_path"] = "E:/flood_research/experiments/terramind_baseline/checkpoints/pretrained/TerraMind_v1_large.pt"
    model_args["necks"][0]["indices"] = CORRECTED_INDICES
    return config


def prepare() -> dict[str, Any]:
    ensure_dirs()
    config = build_config()
    write_yaml(REPO_CONFIG, config)
    write_yaml(RUN_CONFIG, config)
    shutil.copy2(Path(__file__), RUN_DIR / "scripts" / Path(__file__).name)
    update_pipeline_status(
        status="running",
        corrected_large_upernet_indices_tested=False,
        corrected_large_upernet_smoke_passed=False,
        corrected_large_upernet_training_started=False,
        corrected_large_upernet_training_completed=False,
    )
    write_report(training_status="prepared", smoke=read_json(SMOKE_JSON), summary=read_json(SUMMARY_JSON))
    return config


def instantiate_transform(config_items: Any) -> Any:
    if not config_items:
        return None
    import albumentations as A
    from albumentations.pytorch.transforms import ToTensorV2

    transforms = []
    for item in config_items:
        class_path = item.get("class_path") if isinstance(item, dict) else str(item)
        if class_path == "albumentations.D4":
            transforms.append(A.D4())
        elif class_path == "albumentations.pytorch.transforms.ToTensorV2":
            transforms.append(ToTensorV2())
        else:
            raise ValueError(f"Unsupported transform in STEP 5S-A config: {class_path}")
    return transforms


def datamodule_args(config: dict[str, Any], *, split: str | None = None, train_aug: bool = False) -> dict[str, Any]:
    args = dict(config["data"]["init_args"])
    if train_aug:
        args["train_transform"] = instantiate_transform(args.get("train_transform"))
    else:
        args["train_transform"] = None
    args["val_transform"] = None
    args["test_transform"] = None
    args["batch_size"] = int(config["trainer"]["batch_size"])
    args["num_workers"] = 0
    args["pin_memory"] = True
    if split in {"valid", "test", "bolivia"}:
        args["test_split"] = path_text(SPLIT_FILES[split])
    return args


def build_datamodule(config: dict[str, Any], *, split: str | None = None, train_aug: bool = False):
    from terratorch.datamodules import GenericMultiModalDataModule

    return GenericMultiModalDataModule(**datamodule_args(config, split=split, train_aug=train_aug))


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


def set_batchnorm_eval(module: Any) -> int:
    import torch.nn as nn

    count = 0
    for child in module.modules():
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            child.eval()
            count += 1
            for parameter in child.parameters(recurse=False):
                parameter.requires_grad = False
    return count


def move_batch(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    import torch

    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            moved[key] = {
                inner_key: inner_value.to(device) if torch.is_tensor(inner_value) else inner_value
                for inner_key, inner_value in value.items()
            }
        elif torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def output_tensor(model_output: Any):
    if hasattr(model_output, "output"):
        return model_output.output
    if isinstance(model_output, dict):
        if "output" in model_output:
            return model_output["output"]
        if "out" in model_output:
            return model_output["out"]
        return next(iter(model_output.values()))
    if isinstance(model_output, (list, tuple)):
        return model_output[0]
    return model_output


def compute_task_loss(task: Any, batch: dict[str, Any]) -> tuple[Any, Any]:
    model_output = task(batch["image"])
    target = task.squeeze_ground_truth(batch["mask"])
    loss_dict = task.train_loss_handler.compute_loss(model_output, target, task.criterion, task.aux_loss)
    return loss_dict["loss"], output_tensor(model_output)


def confusion_from_tensors(y_true: Any, y_pred: Any) -> list[list[int]]:
    import torch

    valid = y_true != -1
    y_true = y_true[valid].long()
    y_pred = y_pred[valid].long()
    matrix = torch.zeros((2, 2), dtype=torch.int64, device=y_true.device)
    if y_true.numel() > 0:
        indices = y_true * 2 + y_pred
        counts = torch.bincount(indices, minlength=4)
        matrix = counts.reshape(2, 2)
    return [[int(v) for v in row] for row in matrix.detach().cpu().tolist()]


def add_matrices(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
    return [[a[0][0] + b[0][0], a[0][1] + b[0][1]], [a[1][0] + b[1][0], a[1][1] + b[1][1]]]


def metrics_from_matrix(matrix: list[list[int]]) -> dict[str, Any]:
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    total = tn + fp + fn + tp
    precision = tp / (tp + fp) if (tp + fp) else math.nan
    recall = tp / (tp + fn) if (tp + fn) else math.nan
    f1 = 2 * precision * recall / (precision + recall) if math.isfinite(precision + recall) and (precision + recall) else math.nan
    iou_background = tn / (tn + fp + fn) if (tn + fp + fn) else math.nan
    iou_water = tp / (tp + fp + fn) if (tp + fp + fn) else math.nan
    mean_iou = (iou_background + iou_water) / 2 if math.isfinite(iou_background) and math.isfinite(iou_water) else math.nan
    return {
        "accuracy": (tn + tp) / total if total else math.nan,
        "precision_water": precision,
        "recall_water": recall,
        "f1_water": f1,
        "iou_background": iou_background,
        "iou_water": iou_water,
        "mean_iou": mean_iou,
        "support_background": tn + fp,
        "support_water": fn + tp,
        "valid_pixel_count": total,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def evaluate_loader(task: Any, loader: Any, device: Any, *, mixed_precision: bool) -> dict[str, Any]:
    import torch

    task.eval()
    set_batchnorm_eval(task)
    total_loss = 0.0
    batches = 0
    matrix = [[0, 0], [0, 0]]
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            with torch.amp.autocast("cuda", enabled=mixed_precision):
                loss, logits = compute_task_loss(task, batch)
            pred = torch.argmax(logits.detach(), dim=1)
            matrix = add_matrices(matrix, confusion_from_tensors(batch["mask"], pred))
            total_loss += float(loss.detach().cpu())
            batches += 1
    metrics = metrics_from_matrix(matrix)
    metrics["loss"] = total_loss / batches if batches else math.nan
    metrics["batches"] = batches
    return metrics


def save_checkpoint(path: Path, task: Any, optimizer: Any, scaler: Any, epoch: int, best_metric: float, config: dict[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": "5S-A",
            "epoch": epoch,
            "best_validation_miou": best_metric,
            "model_state_dict": task.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "config": config,
        },
        path,
    )


def run_smoke(config: dict[str, Any]) -> dict[str, Any]:
    import torch

    ensure_dirs()
    started = time.time()
    summary: dict[str, Any] = {
        "step": "5S-A",
        "status": "running",
        "corrected_indices": CORRECTED_INDICES,
        "old_indices": OLD_INDICES,
        "mixed_precision": config["trainer"]["precision"],
        "batch_size": config["trainer"]["batch_size"],
        "gradient_accumulation_steps": config["trainer"]["gradient_accumulation_steps"],
        "output_shape_expected": [1, 2, 512, 512],
        "physics_loss_training_started": False,
        "topographic_loss": False,
        "dem_input": False,
        "raw_data_modified": False,
        "generated_at": now_utc(),
    }
    try:
        torch.manual_seed(int(config.get("seed_everything", 42)))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dm = build_datamodule(config, train_aug=True)
        dm.setup("fit")
        batch = next(iter(dm.train_dataloader()))
        input_shapes = {key: list(value.shape) for key, value in batch["image"].items()}
        task = build_task(config).to(device)
        task.train()
        bn_eval_modules = set_batchnorm_eval(task)
        batch = move_batch(batch, device)
        params = [param for param in task.parameters() if param.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
            lr=float(config["optimizer"]["init_args"]["lr"]),
            weight_decay=float(config["optimizer"]["init_args"]["weight_decay"]),
        )
        mixed = config["trainer"]["precision"] == "16-mixed" and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=mixed)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
        with torch.amp.autocast("cuda", enabled=mixed):
            loss, logits = compute_task_loss(task, batch)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        output_shape = list(logits.shape)
        summary.update(
            {
                "status": "passed",
                "passed": True,
                "input_shapes": input_shapes,
                "mask_shape": list(batch["mask"].shape),
                "output_shape": output_shape,
                "output_shape_ok": output_shape == [1, 2, 512, 512],
                "loss": float(loss.detach().cpu()),
                "loss_finite": bool(torch.isfinite(loss).item()),
                "output_finite": bool(torch.isfinite(logits).all().item()),
                "backward_ok": True,
                "optimizer_step_ok": True,
                "batchnorm_eval_modules": bn_eval_modules,
                "gpu_peak_memory_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1024 / 1024, 3)
                if device.type == "cuda"
                else None,
                "elapsed_seconds": round(time.time() - started, 3),
            }
        )
        if not summary["output_shape_ok"] or not summary["loss_finite"] or not summary["output_finite"]:
            raise RuntimeError(f"Smoke produced invalid summary: {summary}")
        update_pipeline_status(
            status="running",
            corrected_large_upernet_indices_tested=True,
            corrected_large_upernet_smoke_passed=True,
            corrected_large_upernet_training_started=False,
            corrected_large_upernet_training_completed=False,
        )
    except Exception as exc:  # noqa: BLE001
        summary.update(
            {
                "status": "blocked",
                "passed": False,
                "error": repr(exc),
                "elapsed_seconds": round(time.time() - started, 3),
            }
        )
        update_pipeline_status(
            status="blocked",
            corrected_large_upernet_indices_tested=True,
            corrected_large_upernet_smoke_passed=False,
            corrected_large_upernet_training_started=False,
            corrected_large_upernet_training_completed=False,
        )
    write_json(SMOKE_JSON, summary)
    write_report(training_status="blocked" if not summary.get("passed") else "smoke_passed", smoke=summary, summary=read_json(SUMMARY_JSON))
    return summary


def configure_logging() -> None:
    ensure_dirs()
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(TRAINING_LOG, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(stream)
    root.addHandler(file_handler)


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    import torch

    configure_logging()
    logging.info("Starting STEP 5S-A corrected TerraMind-L UPerNet Dice training.")
    logging.info("Guardrails: no physics loss, no topographic loss, no DEM input, no DARN, no STURM.")
    update_pipeline_status(
        status="running",
        corrected_large_upernet_indices_tested=True,
        corrected_large_upernet_smoke_passed=bool(read_json(SMOKE_JSON).get("passed")),
        corrected_large_upernet_training_started=True,
        corrected_large_upernet_training_completed=False,
    )
    summary: dict[str, Any] = {
        "step": "5S-A",
        "status": "running",
        "run_dir": path_text(RUN_DIR),
        "config_path": path_text(REPO_CONFIG),
        "run_config_path": path_text(RUN_CONFIG),
        "corrected_indices": CORRECTED_INDICES,
        "old_indices": OLD_INDICES,
        "training_started": True,
        "training_completed": False,
        "physics_loss_training_started": False,
        "topographic_loss": False,
        "dem_input": False,
        "raw_data_modified": False,
        "started_at": now_utc(),
    }
    write_json(SUMMARY_JSON, summary)
    try:
        torch.manual_seed(int(config.get("seed_everything", 42)))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mixed = config["trainer"]["precision"] == "16-mixed" and device.type == "cuda"
        dm = build_datamodule(config, train_aug=True)
        dm.setup("fit")
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        task = build_task(config).to(device)
        params = [param for param in task.parameters() if param.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
            lr=float(config["optimizer"]["init_args"]["lr"]),
            weight_decay=float(config["optimizer"]["init_args"]["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(config["lr_scheduler"]["init_args"]["factor"]),
            patience=int(config["lr_scheduler"]["init_args"]["patience"]),
        )
        scaler = torch.amp.GradScaler("cuda", enabled=mixed)
        max_epochs = int(config["trainer"]["max_epochs"])
        grad_accum = int(config["trainer"]["gradient_accumulation_steps"])
        patience = int(config["trainer"]["early_stopping_patience"])
        min_epochs = int(config["trainer"]["early_stopping_min_epochs"])
        best_metric = -math.inf
        best_epoch = 0
        no_improve = 0
        epoch_rows: list[dict[str, Any]] = []
        training_started = time.time()

        for epoch in range(1, max_epochs + 1):
            epoch_started = time.time()
            task.train()
            bn_eval_modules = set_batchnorm_eval(task)
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            batches = 0
            train_matrix = [[0, 0], [0, 0]]
            for batch_index, batch in enumerate(train_loader, start=1):
                batch = move_batch(batch, device)
                with torch.amp.autocast("cuda", enabled=mixed):
                    loss, logits = compute_task_loss(task, batch)
                    scaled_loss = loss / grad_accum
                if not torch.isfinite(loss).item():
                    raise RuntimeError(f"Non-finite training loss at epoch={epoch}, batch={batch_index}: {float(loss.detach().cpu())}")
                if not torch.isfinite(logits).all().item():
                    raise RuntimeError(f"Non-finite logits at epoch={epoch}, batch={batch_index}")
                scaler.scale(scaled_loss).backward()
                if batch_index % grad_accum == 0 or batch_index == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    set_batchnorm_eval(task)
                pred = torch.argmax(logits.detach(), dim=1)
                train_matrix = add_matrices(train_matrix, confusion_from_tensors(batch["mask"], pred))
                running_loss += float(loss.detach().cpu())
                batches += 1

            train_metrics = metrics_from_matrix(train_matrix)
            val_metrics = evaluate_loader(task, val_loader, device, mixed_precision=mixed)
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_metrics["mean_iou"])
            next_lr = optimizer.param_groups[0]["lr"]
            improved = bool(val_metrics["mean_iou"] > best_metric)
            if improved:
                best_metric = float(val_metrics["mean_iou"])
                best_epoch = epoch
                no_improve = 0
                save_checkpoint(RUN_DIR / "checkpoints" / "best_checkpoint.pt", task, optimizer, scaler, epoch, best_metric, config)
            else:
                no_improve += 1
            save_checkpoint(RUN_DIR / "checkpoints" / "last_checkpoint.pt", task, optimizer, scaler, epoch, best_metric, config)
            row = {
                "epoch": epoch,
                "train_loss": running_loss / batches if batches else math.nan,
                "validation_loss": val_metrics["loss"],
                "train_miou": train_metrics["mean_iou"],
                "validation_miou": val_metrics["mean_iou"],
                "validation_iou_water": val_metrics["iou_water"],
                "validation_f1_water": val_metrics["f1_water"],
                "learning_rate": current_lr,
                "learning_rate_after_scheduler": next_lr,
                "best_epoch": best_epoch,
                "epochs_without_improvement": no_improve,
                "batchnorm_eval_modules": bn_eval_modules,
                "precision": config["trainer"]["precision"],
                "elapsed_seconds": round(time.time() - epoch_started, 3),
                "improved": improved,
            }
            epoch_rows.append(row)
            write_csv(RUN_DIR / "metrics" / "training_epoch_metrics.csv", epoch_rows, list(row.keys()))
            logging.info(
                "epoch=%s train_loss=%.6f val_loss=%.6f val_miou=%.6f val_iou_water=%.6f val_f1_water=%.6f lr=%.8f next_lr=%.8f no_improve=%s precision=%s bn_eval=%s",
                epoch,
                row["train_loss"],
                row["validation_loss"],
                row["validation_miou"],
                row["validation_iou_water"],
                row["validation_f1_water"],
                current_lr,
                next_lr,
                no_improve,
                config["trainer"]["precision"],
                bn_eval_modules,
            )
            summary.update(
                {
                    "epoch_rows": epoch_rows,
                    "best_epoch": best_epoch,
                    "best_validation_miou": best_metric,
                    "last_epoch": epoch,
                    "training_elapsed_seconds": round(time.time() - training_started, 3),
                }
            )
            write_json(SUMMARY_JSON, summary)
            if epoch >= min_epochs and no_improve >= patience:
                logging.info("Early stopping at epoch=%s: validation_miou did not improve for %s epochs after epoch %s", epoch, patience, best_epoch)
                break

        evals = evaluate_predictions(config, task, device, mixed_precision=mixed)
        comparison = write_comparison(evals)
        summary.update(
            {
                "status": "done",
                "training_completed": True,
                "completed_at": now_utc(),
                "evaluations": evals,
                "comparison_json": path_text(RUN_DIR / "metrics" / "step5s_a_vs_step5o_vs_step5i_comparison.json"),
                "comparison_csv": path_text(RUN_DIR / "metrics" / "step5s_a_vs_step5o_vs_step5i_comparison.csv"),
                "baseline_decision": comparison.get("baseline_decision", {}),
            }
        )
        write_json(SUMMARY_JSON, summary)
        update_pipeline_status(
            status="done",
            corrected_large_upernet_indices_tested=True,
            corrected_large_upernet_smoke_passed=True,
            corrected_large_upernet_training_started=True,
            corrected_large_upernet_training_completed=True,
            step5s_a_summary_json=path_text(SUMMARY_JSON),
            step5s_a_best_checkpoint=path_text(RUN_DIR / "checkpoints" / "best_checkpoint.pt"),
            step5s_a_last_checkpoint=path_text(RUN_DIR / "checkpoints" / "last_checkpoint.pt"),
        )
        write_report(training_status="done", smoke=read_json(SMOKE_JSON), summary=summary)
        logging.info("STEP 5S-A complete. Human validation required before deciding STEP 5S-B loss ablations or STEP 6C.")
    except Exception as exc:  # noqa: BLE001
        logging.exception("STEP 5S-A training failed.")
        summary.update({"status": "blocked", "training_completed": False, "error": repr(exc), "blocked_at": now_utc()})
        write_json(SUMMARY_JSON, summary)
        update_pipeline_status(
            status="blocked",
            corrected_large_upernet_indices_tested=True,
            corrected_large_upernet_training_started=True,
            corrected_large_upernet_training_completed=False,
        )
        write_report(training_status="blocked", smoke=read_json(SMOKE_JSON), summary=summary)
    return summary


def load_best_checkpoint(task: Any, device: Any) -> None:
    import torch

    checkpoint = torch.load(RUN_DIR / "checkpoints" / "best_checkpoint.pt", map_location=device, weights_only=False)
    task.load_state_dict(checkpoint["model_state_dict"])


def tile_id_from_batch(batch: dict[str, Any], index: int) -> str:
    mask_name = batch["filename"]["mask"][index]
    stem = Path(mask_name).stem
    return stem.replace("_LabelHand", "")


def evaluate_predictions(config: dict[str, Any], task: Any, device: Any, *, mixed_precision: bool) -> dict[str, Any]:
    import numpy as np
    import torch

    load_best_checkpoint(task, device)
    task.eval()
    set_batchnorm_eval(task)
    evals: dict[str, Any] = {}
    for split in ["valid", "test", "bolivia"]:
        pred_dir = RUN_DIR / "predictions" / split
        pred_dir.mkdir(parents=True, exist_ok=True)
        for old in pred_dir.glob("*"):
            if old.is_file():
                old.unlink()
        dm = build_datamodule(config, split=split, train_aug=False)
        dm.setup("test")
        loader = dm.test_dataloader()
        count = 0
        with torch.no_grad():
            for batch in loader:
                original_batch = batch
                batch = move_batch(batch, device)
                with torch.amp.autocast("cuda", enabled=mixed_precision):
                    model_output = task(batch["image"])
                logits = output_tensor(model_output)
                preds = torch.argmax(logits.detach(), dim=1).cpu().numpy().astype(np.int16)
                for index in range(preds.shape[0]):
                    tile_id = tile_id_from_batch(original_batch, index)
                    np.save(pred_dir / f"{tile_id}_pred.npy", preds[index])
                    count += 1
        logging.info("Saved %s %s predictions to %s", count, split, pred_dir)
        summary_path = RUN_DIR / "metrics" / f"{split}_summary.json"
        subprocess.run(
            [
                sys.executable,
                str(EVAL_SCRIPT),
                "--prediction-dir",
                str(pred_dir),
                "--manifest-csv",
                str(MANIFEST_CSV),
                "--output-csv",
                str(RUN_DIR / "metrics" / f"{split}_per_tile_metrics.csv"),
                "--output-grouped-csv",
                str(RUN_DIR / "metrics" / f"{split}_grouped_metrics.csv"),
                "--output-summary-json",
                str(summary_path),
                "--split",
                split,
            ],
            check=True,
        )
        evals[split] = read_json(summary_path)
    return evals


def split_miou_from_summary(summary: dict[str, Any], split: str) -> float | None:
    for row in summary.get("grouped_metrics", []):
        if row.get("group_type") == "split" and row.get("group_value") == split:
            value = row.get("mean_iou")
            return float(value) if value is not None else None
    return None


def split_metric_from_summary(summary: dict[str, Any], split: str, metric: str) -> float | None:
    for row in summary.get("grouped_metrics", []):
        if row.get("group_type") == "split" and row.get("group_value") == split:
            value = row.get(metric)
            return float(value) if value is not None else None
    return None


def write_comparison(evals: dict[str, Any]) -> dict[str, Any]:
    step5o = read_json(RUNS_ROOT / "step5o_terramind_l_upernet_long_classical_training" / "metrics" / "step5o_summary.json")
    step5i = read_json(RUNS_ROOT / "step5i_base_unetdecoder_pretrained" / "metrics" / "step5i_summary.json")
    rows = []
    for split in ["valid", "test", "bolivia"]:
        current = split_miou_from_summary(evals[split], split)
        old = split_miou_from_summary(step5o.get("evaluations", {}).get(split, {}), split)
        base = split_miou_from_summary(step5i.get("evaluations", {}).get(split, {}), split)
        rows.append(
            {
                "split": split,
                "step5s_a_miou": current,
                "step5o_miou": old,
                "step5i_miou": base,
                "step5s_a_minus_step5o_miou": current - old if current is not None and old is not None else None,
                "step5s_a_minus_step5i_miou": current - base if current is not None and base is not None else None,
                "step5s_a_iou_water": split_metric_from_summary(evals[split], split, "iou_water"),
                "step5s_a_f1_water": split_metric_from_summary(evals[split], split, "f1_score"),
            }
        )
    fieldnames = list(rows[0].keys())
    write_csv(RUN_DIR / "metrics" / "step5s_a_vs_step5o_vs_step5i_comparison.csv", rows, fieldnames)
    payload = {
        "rows": rows,
        "baseline_decision": {
            "candidate": "TerraMind-L pretrained + UPerNet corrected large indices",
            "recommendation": "human_review_required",
            "rule_of_thumb": "Prefer STEP 5S-A over STEP 5O only if validation improves without test/Bolivia degradation.",
        },
    }
    write_json(RUN_DIR / "metrics" / "step5s_a_vs_step5o_vs_step5i_comparison.json", payload)
    return payload


def write_report(training_status: str, smoke: dict[str, Any], summary: dict[str, Any]) -> None:
    ensure_dirs()
    lines = [
        "# STEP 5S-A - TerraMind-L UPerNet Corrected Indices Dice Run",
        "",
        f"Generated at: {now_utc()}",
        "",
        "## Purpose",
        "",
        "STEP 5R found that loss parity is mostly satisfied: the official IBM TerraMind Sen1Floods11 configs and our STEP 5I/5O/5P runs all use Dice. The high-confidence unresolved architecture issue is that STEP 5O/5P used UPerNet feature indices [2, 5, 8, 11], while the official config comment marks [5, 11, 17, 23] for the large backbone.",
        "",
        "## Configuration",
        "",
        f"- Model: TerraMind-L pretrained + UPerNet",
        f"- Old STEP 5O/5P indices: {OLD_INDICES}",
        f"- Corrected large-backbone indices: {CORRECTED_INDICES}",
        "- Loss: Dice",
        "- ignore_index: -1",
        "- Classes: 2",
        "- Water class index: 1",
        "- Inputs: S2L1C + S1GRD only",
        "- No physics loss, no topographic loss, no DEM input, no DARN, no STURM-Flood",
        f"- Config: `{path_text(REPO_CONFIG)}`",
        f"- Run config: `{path_text(RUN_CONFIG)}`",
        "",
        "## Dataset Policy",
        "",
        f"- Train: {count_lines(SPLIT_FILES['train'])}",
        f"- Valid: {count_lines(SPLIT_FILES['valid'])}",
        f"- Test: {count_lines(SPLIT_FILES['test'])}",
        f"- Bolivia: {count_lines(SPLIT_FILES['bolivia'])}",
        f"- Excluded fully invalid tiles: {', '.join(EXCLUDED_TILE_IDS)}",
        "- keep no_water: true",
        "- keep warning_review: true",
        "",
        "## Training Recipe",
        "",
        "- AdamW lr: 2e-5",
        "- Weight decay: 1e-4",
        f"- Precision: {read_yaml(REPO_CONFIG).get('trainer', {}).get('precision', 'unknown')}",
        "- Physical batch size: 1",
        "- Gradient accumulation: 8",
        "- Effective batch size: 8",
        "- D4 augmentation: enabled for training",
        "- BatchNorm policy: eval mode and affine parameters frozen for UPerNet BatchNorm modules",
        "- Mixed precision note: 16-mixed smoke passed, but the first background training attempt produced non-finite epoch losses; fp32 is used for the stable run.",
        "- Scheduler: ReduceLROnPlateau on validation_miou, mode=max, factor=0.5, patience=3",
        "- Early stopping: validation_miou, min_epochs=30, patience=15, max_epochs=80",
        "",
        "## Smoke Result",
        "",
    ]
    if smoke:
        lines.extend(
            [
                f"- Status: {smoke.get('status', 'unknown')}",
                f"- Passed: {smoke.get('passed', False)}",
                f"- Output shape: {smoke.get('output_shape', 'unknown')}",
                f"- Loss finite: {smoke.get('loss_finite', 'unknown')}",
                f"- Backward OK: {smoke.get('backward_ok', 'unknown')}",
                f"- BatchNorm eval modules: {smoke.get('batchnorm_eval_modules', 'unknown')}",
                f"- GPU peak allocated MB: {smoke.get('gpu_peak_memory_allocated_mb', 'unknown')}",
            ]
        )
    else:
        lines.append("- Status: not run yet")
    lines.extend(["", "## Training Status", "", f"- Status: {training_status}", f"- Log: `{path_text(TRAINING_LOG)}`", ""])
    if summary:
        lines.extend(
            [
                f"- Training started: {summary.get('training_started', False)}",
                f"- Training completed: {summary.get('training_completed', False)}",
                f"- Best epoch: {summary.get('best_epoch', 'unknown')}",
                f"- Best validation mIoU: {summary.get('best_validation_miou', 'unknown')}",
            ]
        )
        if summary.get("evaluations"):
            lines.extend(["", "## Metrics", ""])
            for split in ["valid", "test", "bolivia"]:
                split_summary = summary["evaluations"].get(split, {})
                miou = split_miou_from_summary(split_summary, split)
                water = split_metric_from_summary(split_summary, split, "iou_water")
                f1 = split_metric_from_summary(split_summary, split, "f1_score")
                lines.append(f"- {split}: mIoU={miou}, IoU water={water}, F1 water={f1}")
            lines.extend(
                [
                    "",
                    f"- Comparison CSV: `{path_text(RUN_DIR / 'metrics' / 'step5s_a_vs_step5o_vs_step5i_comparison.csv')}`",
                    f"- Comparison JSON: `{path_text(RUN_DIR / 'metrics' / 'step5s_a_vs_step5o_vs_step5i_comparison.json')}`",
                ]
            )
        if summary.get("error"):
            lines.extend(["", "## Error", "", f"`{summary['error']}`"])
    lines.extend(
        [
            "",
            "## Decision Notes",
            "",
            "- This run should become the new TerraMind-L UPerNet classical baseline only if it improves validation mIoU without degrading test/Bolivia behavior.",
            "- CE+Dice / weighted CE+Dice ablations are still deferred until this corrected-architecture run is reviewed.",
            "- Physics-informed STEP 6C remains blocked pending human validation.",
            "",
            "## Guardrails",
            "",
            "- Physics-informed training started: false",
            "- DARN started: false",
            "- STURM-Flood training started: false",
            "- Raw data modified: false",
            "- Official split files modified: false",
            "",
            "## Next Step",
            "",
            "Human validation after completion before deciding STEP 5S-B loss ablations or STEP 6C.",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    shutil.copy2(REPORT_PATH, RUN_REPORT_PATH)


def mark_running(parent_pid: int | None, python_child_pid: int | None) -> None:
    ensure_dirs()
    info = {
        "step": "5S-A",
        "status": "running",
        "parent_pid": parent_pid,
        "python_child_pid": python_child_pid,
        "runner_path": path_text(Path(__file__)),
        "config_path": path_text(REPO_CONFIG),
        "run_config_path": path_text(RUN_CONFIG),
        "run_dir": path_text(RUN_DIR),
        "log_path": path_text(TRAINING_LOG),
        "generated_at": now_utc(),
    }
    write_json(LAUNCH_INFO_JSON, info)
    update_pipeline_status(
        status="running",
        corrected_large_upernet_indices_tested=True,
        corrected_large_upernet_smoke_passed=True,
        corrected_large_upernet_training_started=True,
        corrected_large_upernet_training_completed=False,
        step5s_a_parent_pid=parent_pid,
        step5s_a_python_child_pid=python_child_pid,
        step5s_a_launch_info_json=path_text(LAUNCH_INFO_JSON),
    )
    write_report(training_status="running", smoke=read_json(SMOKE_JSON), summary=read_json(SUMMARY_JSON))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["prepare", "smoke", "train", "mark-running"], required=True)
    parser.add_argument("--config", type=Path, default=REPO_CONFIG)
    parser.add_argument("--parent-pid", type=int, default=None)
    parser.add_argument("--python-child-pid", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "prepare":
        prepare()
        print(path_text(REPO_CONFIG))
        return 0
    config = read_yaml(args.config if args.config.exists() else REPO_CONFIG)
    if args.mode == "smoke":
        summary = run_smoke(config)
        print(json.dumps(json_safe(summary), indent=2))
        return 0 if summary.get("passed") else 2
    if args.mode == "mark-running":
        mark_running(args.parent_pid, args.python_child_pid)
        return 0
    if args.mode == "train":
        run_training(config)
        return 0
    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
