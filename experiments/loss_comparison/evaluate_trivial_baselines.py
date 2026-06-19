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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trivial segmentation baselines.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def logits_from_prediction(prediction: Tensor) -> Tensor:
    return torch.where(
        prediction > 0.5,
        torch.full_like(prediction, 10.0),
        torch.full_like(prediction, -10.0),
    )


def metrics_from_counts(counts: dict[str, float], eps: float = 1e-6) -> dict[str, float]:
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    valid = counts["valid_pixel_count"]
    positive = counts["positive_pixel_count"]
    predicted_positive = counts["predicted_positive_pixel_count"]
    iou = tp / (tp + fp + fn + eps)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    return {
        "iou": iou,
        "dice": dice,
        "f1": dice,
        "recall": tp / (tp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "target_positive_rate": positive / (valid + eps),
        "predicted_positive_rate": predicted_positive / (valid + eps),
    }


def aggregate_target_rate(loader: DataLoader) -> float:
    positive = 0.0
    valid = 0.0
    for batch in loader:
        mask = batch["mask"]
        valid_mask = batch["valid_mask"]
        if not isinstance(mask, Tensor) or not isinstance(valid_mask, Tensor):
            raise TypeError("mask and valid_mask must be tensors.")
        positive += float((mask * valid_mask).sum())
        valid += float(valid_mask.sum())
    return positive / valid if valid > 0 else 0.0


def dataset_view(dataset: GeoTIFFDataset, output_dir: Path) -> GeoTIFFDataset | Subset:
    config_path = output_dir / "config.json"
    if not config_path.exists():
        return dataset
    config = json.loads(config_path.read_text(encoding="utf-8"))
    val_indices = config.get("val_indices")
    if isinstance(val_indices, list) and val_indices:
        print(f"[INFO] Evaluating trivial baselines on val_indices from config: {val_indices}")
        return Subset(dataset, [int(index) for index in val_indices])
    return dataset


def evaluate_baseline(
    loader: DataLoader,
    *,
    baseline: str,
    positive_probability: float,
    seed: int,
) -> dict[str, object]:
    rng = torch.Generator()
    rng.manual_seed(seed)
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
    for batch in loader:
        mask = batch["mask"]
        valid_mask = batch["valid_mask"]
        dem = batch["dem"]
        if (
            not isinstance(mask, Tensor)
            or not isinstance(valid_mask, Tensor)
            or not isinstance(dem, Tensor)
        ):
            raise TypeError("mask, valid_mask and dem must be tensors.")
        if baseline == "all_background":
            prediction = torch.zeros_like(mask)
        elif baseline == "all_water":
            prediction = torch.ones_like(mask)
        elif baseline == "random_target_rate":
            prediction = (
                torch.rand(mask.shape, generator=rng, device=mask.device) < positive_probability
            ).float()
        else:
            raise ValueError(f"Unknown baseline: {baseline}")

        logits = logits_from_prediction(prediction)
        batch_counts = masked_binary_confusion_counts(logits, mask, valid_mask)
        for key in counts:
            counts[key] += batch_counts[key]
        violation_rates.append(
            violation_rate_topo(logits=logits, dem=dem, valid_mask=valid_mask)[
                "violation_rate_topo"
            ]
        )

    metrics = metrics_from_counts(counts)
    return {
        "baseline": baseline,
        **metrics,
        "violation_rate_topo": float(np.mean(violation_rates)) if violation_rates else 0.0,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "baseline",
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


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "| Baseline | IoU | Dice | F1 | Recall | Precision | ViolationRateTopo | Target+ | Pred+ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {baseline} | {iou:.4f} | {dice:.4f} | {f1:.4f} | {recall:.4f} | "
            "{precision:.4f} | {violation:.4f} | {target:.4f} | {pred:.4f} |".format(
                baseline=row["baseline"],
                iou=float(row["iou"]),
                dice=float(row["dice"]),
                f1=float(row["f1"]),
                recall=float(row["recall"]),
                precision=float(row["precision"]),
                violation=float(row["violation_rate_topo"]),
                target=float(row["target_positive_rate"]),
                pred=float(row["predicted_positive_rate"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        random.seed(args.seed)
        dataset = GeoTIFFDataset(args.manifest, max_samples=args.max_samples, require_dem=True)
        view = dataset_view(dataset, args.output_dir)
        loader = DataLoader(
            view,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=geotiff_collate_fn,
        )
        positive_probability = aggregate_target_rate(loader)
        rows = [
            evaluate_baseline(
                loader,
                baseline="all_background",
                positive_probability=0.0,
                seed=args.seed,
            ),
            evaluate_baseline(
                loader,
                baseline="all_water",
                positive_probability=1.0,
                seed=args.seed,
            ),
            evaluate_baseline(
                loader,
                baseline="random_target_rate",
                positive_probability=positive_probability,
                seed=args.seed,
            ),
        ]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(args.output_dir / "trivial_baselines.csv", rows)
        write_markdown(args.output_dir / "trivial_baselines.md", rows)
        print(f"[OK] trivial_baselines.csv: {args.output_dir / 'trivial_baselines.csv'}")
        print(f"[OK] trivial_baselines.md: {args.output_dir / 'trivial_baselines.md'}")
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
