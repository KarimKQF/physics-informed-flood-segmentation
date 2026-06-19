from __future__ import annotations

# ruff: noqa: E402
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from urban_runoff.data import GeoTIFFDataset, geotiff_collate_fn
from urban_runoff.metrics import masked_binary_confusion_counts, violation_rate_topo
from urban_runoff.models import SimpleSegmentationCNN

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved checkpoints over thresholds.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def split_indices(dataset: GeoTIFFDataset, seed: int, config: dict[str, object]) -> list[int]:
    config_val = config.get("val_indices")
    if isinstance(config_val, list) and config_val:
        return [int(index) for index in config_val]
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_count = max(1, min(len(indices) - 1, round(len(indices) * 0.25)))
    return sorted(indices[:val_count])


def metrics_from_counts(counts: dict[str, float], eps: float = 1e-6) -> dict[str, float]:
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    valid = counts["valid_pixel_count"]
    positive = counts["positive_pixel_count"]
    predicted_positive = counts["predicted_positive_pixel_count"]
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    return {
        "iou": tp / (tp + fp + fn + eps),
        "dice": dice,
        "f1": dice,
        "recall": tp / (tp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "target_positive_rate": positive / (valid + eps),
        "predicted_positive_rate": predicted_positive / (valid + eps),
    }


def evaluate_checkpoint(
    *,
    model: SimpleSegmentationCNN,
    loader: DataLoader,
    threshold: float,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    counts = {
        "tp": 0.0,
        "fp": 0.0,
        "fn": 0.0,
        "tn": 0.0,
        "valid_pixel_count": 0.0,
        "positive_pixel_count": 0.0,
        "negative_pixel_count": 0.0,
        "predicted_positive_pixel_count": 0.0,
    }
    violation_rates = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"]
            mask = batch["mask"]
            valid_mask = batch["valid_mask"]
            dem = batch["dem"]
            if (
                not isinstance(image, Tensor)
                or not isinstance(mask, Tensor)
                or not isinstance(valid_mask, Tensor)
                or not isinstance(dem, Tensor)
            ):
                raise TypeError("image, mask, valid_mask and dem must be tensors.")
            image = image.to(device=device, dtype=torch.float32)
            mask = mask.to(device=device, dtype=torch.float32)
            valid_mask = valid_mask.to(device=device, dtype=torch.float32)
            dem = dem.to(device=device, dtype=torch.float32)
            logits = model(image)
            batch_counts = masked_binary_confusion_counts(
                logits,
                mask,
                valid_mask,
                threshold=threshold,
            )
            for key in counts:
                counts[key] += batch_counts[key]
            violation_rates.append(
                violation_rate_topo(
                    logits=logits,
                    dem=dem,
                    valid_mask=valid_mask,
                    threshold=threshold,
                )["violation_rate_topo"]
            )
    metrics = metrics_from_counts(counts)
    metrics["violation_rate_topo"] = float(np.mean(violation_rates)) if violation_rates else 0.0
    return metrics


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "experiment",
        "threshold",
        "iou",
        "dice",
        "f1",
        "recall",
        "precision",
        "violation_rate_topo",
        "target_positive_rate",
        "predicted_positive_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_best_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    by_experiment: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_experiment.setdefault(str(row["experiment"]), []).append(row)
    lines = [
        "| Experiment | Best Dice Threshold | Best Dice | Best IoU Threshold | Best IoU |",
        "|---|---:|---:|---:|---:|",
    ]
    for experiment, experiment_rows in sorted(by_experiment.items()):
        best_dice = max(experiment_rows, key=lambda row: float(row["dice"]))
        best_iou = max(experiment_rows, key=lambda row: float(row["iou"]))
        lines.append(
            "| {experiment} | {dice_threshold:.1f} | {dice:.4f} | {iou_threshold:.1f} | "
            "{iou:.4f} |".format(
                experiment=experiment,
                dice_threshold=float(best_dice["threshold"]),
                dice=float(best_dice["dice"]),
                iou_threshold=float(best_iou["threshold"]),
                iou=float(best_iou["iou"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        device = resolve_device(args.device)
        config_path = args.results_dir / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        max_samples = int(config.get("max_samples", 8))
        seed = int(config.get("seed", 42))
        dataset = GeoTIFFDataset(args.manifest, max_samples=max_samples, require_dem=True)
        val_indices = split_indices(dataset, seed, config)
        loader = DataLoader(
            Subset(dataset, val_indices),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=geotiff_collate_fn,
        )
        checkpoint_dir = args.results_dir / "checkpoints"
        checkpoints = sorted(checkpoint_dir.glob("*_best.pt"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

        rows: list[dict[str, object]] = []
        for checkpoint_path in checkpoints:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            experiment = str(checkpoint["experiment"])
            model = SimpleSegmentationCNN(in_channels=int(checkpoint.get("in_channels", 2))).to(
                device
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            for threshold in THRESHOLDS:
                metrics = evaluate_checkpoint(
                    model=model,
                    loader=loader,
                    threshold=threshold,
                    device=device,
                )
                rows.append({"experiment": experiment, "threshold": threshold, **metrics})

        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(args.output_dir / "threshold_sweep.csv", rows)
        write_best_markdown(args.output_dir / "threshold_sweep_best.md", rows)
        print(f"[OK] threshold_sweep.csv: {args.output_dir / 'threshold_sweep.csv'}")
        print(f"[OK] threshold_sweep_best.md: {args.output_dir / 'threshold_sweep_best.md'}")
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
