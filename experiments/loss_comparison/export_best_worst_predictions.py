from __future__ import annotations

# ruff: noqa: E402
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from torch import Tensor
from torch.utils.data import Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from urban_runoff.data import GeoTIFFDataset
from urban_runoff.metrics import masked_binary_confusion_counts
from urban_runoff.models import SimpleSegmentationCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export best/worst prediction diagnostics.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def normalize_preview(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype="float32")
    low, high = np.percentile(values[finite], [2, 98])
    if high <= low:
        return np.zeros_like(values, dtype="float32")
    return np.clip((values - low) / (high - low), 0, 1).astype("float32")


def split_indices(dataset: GeoTIFFDataset, config: dict[str, object]) -> list[int]:
    config_val = config.get("val_indices")
    if isinstance(config_val, list) and config_val:
        return [int(index) for index in config_val]
    seed = int(config.get("seed", 42))
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_count = max(1, min(len(indices) - 1, round(len(indices) * 0.25)))
    return sorted(indices[:val_count])


def best_experiment_from_summary(summary_path: Path) -> str:
    with summary_path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    best = max(rows, key=lambda row: float(row["best_val_iou"]))
    return str(best["experiment"])


def load_model(results_dir: Path, experiment: str, device: torch.device) -> SimpleSegmentationCNN:
    checkpoint_path = results_dir / "checkpoints" / f"{experiment}_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SimpleSegmentationCNN(in_channels=int(checkpoint.get("in_channels", 2))).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def error_map(prediction: Tensor, mask: Tensor, valid_mask: Tensor) -> np.ndarray:
    pred = prediction.bool()
    target = mask.bool()
    valid = valid_mask.bool()
    error = torch.zeros_like(mask, dtype=torch.int64)
    error[(~pred) & (~target) & valid] = 0
    error[pred & target & valid] = 1
    error[pred & (~target) & valid] = 2
    error[(~pred) & target & valid] = 3
    error[~valid] = 4
    return error.squeeze().detach().cpu().numpy()


def dice_for_sample(prediction: Tensor, mask: Tensor, valid_mask: Tensor) -> float:
    logits = torch.where(
        prediction > 0.5,
        torch.full_like(prediction, 10.0),
        torch.full_like(prediction, -10.0),
    )
    counts = masked_binary_confusion_counts(logits, mask, valid_mask)
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    return (2.0 * tp) / (2.0 * tp + fp + fn + 1e-6)


def save_figure(
    *,
    path: Path,
    title: str,
    image: Tensor,
    mask: Tensor,
    prediction: Tensor,
    valid_mask: Tensor,
    dem: Tensor,
) -> None:
    error_colors = ListedColormap(["#1f2937", "#22c55e", "#ef4444", "#f59e0b", "#8b5cf6"])
    panels = [
        ("Sentinel-1 band 1", normalize_preview(image[0].detach().cpu().numpy())),
        ("Ground truth", mask[0].detach().cpu().numpy()),
        ("Prediction", prediction[0].detach().cpu().numpy()),
        ("Error map", error_map(prediction, mask, valid_mask)),
        ("Valid mask", valid_mask[0].detach().cpu().numpy()),
        ("DEM", normalize_preview(dem[0].detach().cpu().numpy())),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(16, 3))
    fig.suptitle(title)
    for axis, (panel_title, values) in zip(axes, panels, strict=True):
        if panel_title == "Error map":
            axis.imshow(values, cmap=error_colors, vmin=0, vmax=4)
        else:
            axis.imshow(values, cmap="gray")
        axis.set_title(panel_title)
        axis.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def tensor_sample(
    sample: dict[str, Tensor | str | None],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    image = sample["image"]
    mask = sample["mask"]
    valid_mask = sample["valid_mask"]
    dem = sample["dem"]
    if (
        not isinstance(image, Tensor)
        or not isinstance(mask, Tensor)
        or not isinstance(valid_mask, Tensor)
        or not isinstance(dem, Tensor)
    ):
        raise TypeError("image, mask, valid_mask and dem must be tensors.")
    return (
        image.to(device=device, dtype=torch.float32),
        mask.to(device=device, dtype=torch.float32),
        valid_mask.to(device=device, dtype=torch.float32),
        dem.to(device=device, dtype=torch.float32),
    )


def main() -> int:
    args = parse_args()
    try:
        device = resolve_device(args.device)
        config_path = args.results_dir / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        max_samples = int(config.get("max_samples", 8))
        dataset = GeoTIFFDataset(args.manifest, max_samples=max_samples, require_dem=True)
        val_indices = split_indices(dataset, config)
        val_subset = Subset(dataset, val_indices)
        experiment = best_experiment_from_summary(args.results_dir / "summary_results.csv")
        model = load_model(args.results_dir, experiment, device)

        scored = []
        with torch.no_grad():
            for local_index in range(len(val_subset)):
                sample = val_subset[local_index]
                image, mask, valid_mask, dem = tensor_sample(sample, device)
                logits = model(image[None])
                prediction = (torch.sigmoid(logits[0]) >= 0.5).float()
                scored.append(
                    {
                        "score": dice_for_sample(prediction, mask, valid_mask),
                        "sample": sample,
                        "prediction": prediction.cpu(),
                    }
                )
        best = max(scored, key=lambda item: float(item["score"]))
        worst = min(scored, key=lambda item: float(item["score"]))

        for filename, item, title in [
            ("best_prediction_overall.png", best, f"Best prediction overall ({experiment})"),
            ("worst_prediction_overall.png", worst, f"Worst prediction overall ({experiment})"),
        ]:
            image, mask, valid_mask, dem = tensor_sample(item["sample"], torch.device("cpu"))
            save_figure(
                path=args.output_dir / filename,
                title=title,
                image=image,
                mask=mask,
                prediction=item["prediction"],
                valid_mask=valid_mask,
                dem=dem,
            )

        baseline_sample = val_subset[0]
        image, mask, valid_mask, dem = tensor_sample(baseline_sample, torch.device("cpu"))
        save_figure(
            path=args.output_dir / "all_water_baseline_example.png",
            title="All-water baseline",
            image=image,
            mask=mask,
            prediction=torch.ones_like(mask),
            valid_mask=valid_mask,
            dem=dem,
        )
        save_figure(
            path=args.output_dir / "all_background_baseline_example.png",
            title="All-background baseline",
            image=image,
            mask=mask,
            prediction=torch.zeros_like(mask),
            valid_mask=valid_mask,
            dem=dem,
        )
        print(f"[OK] Prediction diagnostics written to: {args.output_dir}")
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
