from __future__ import annotations

# ruff: noqa: E402
import argparse
import csv
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from urban_runoff.data import GeoTIFFDataset, geotiff_collate_fn
from urban_runoff.losses import MaskedBCEDiceLoss
from urban_runoff.metrics import masked_binary_confusion_counts, violation_rate_topo
from urban_runoff.models import SimpleSegmentationCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss", type=str, default="bce_dice")
    parser.add_argument("--use-pos-weight", action="store_true")
    parser.add_argument("--threshold", type=str, default="auto")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_balance(dataset: GeoTIFFDataset, indices: list[int]) -> float:
    positive = 0.0
    valid = 0.0
    for index in indices:
        sample = dataset[index]
        mask = sample["mask"]
        valid_mask = sample["valid_mask"]
        if not isinstance(mask, Tensor) or not isinstance(valid_mask, Tensor):
            raise TypeError("mask and valid_mask must be tensors.")
        positive += float((mask * valid_mask).sum())
        valid += float(valid_mask.sum())
    negative = valid - positive
    return negative / positive if positive > 0 else 1.0


def move_batch(
    batch: dict[str, object], device: torch.device
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    image = batch["image"]
    mask = batch["mask"]
    valid_mask = batch["valid_mask"]
    dem = batch["dem"]
    if not all(isinstance(x, Tensor) for x in (image, mask, valid_mask, dem)):
        raise TypeError("Batch items must be tensors.")
    return (
        image.to(device=device, dtype=torch.float32),
        mask.to(device=device, dtype=torch.float32),
        valid_mask.to(device=device, dtype=torch.float32),
        dem.to(device=device, dtype=torch.float32),
    )


def calculate_metrics(counts: dict[str, float]) -> dict[str, float]:
    eps = 1e-6
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    valid = counts["valid_pixel_count"]
    pos = counts["positive_pixel_count"]
    pred = counts["predicted_positive_pixel_count"]

    iou = tp / (tp + fp + fn + eps)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    recall = tp / (tp + fn + eps)
    precision = tp / (tp + fp + eps)
    target_positive_rate = pos / (valid + eps)
    predicted_positive_rate = pred / (valid + eps)

    return {
        "iou": iou,
        "dice": dice,
        "f1": dice,
        "recall": recall,
        "precision": precision,
        "target_positive_rate": target_positive_rate,
        "predicted_positive_rate": predicted_positive_rate,
    }


def evaluate_threshold(
    model: nn.Module, loader: DataLoader, threshold: float, device: torch.device
) -> float:
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
    with torch.no_grad():
        for batch in loader:
            image, mask, valid_mask, _ = move_batch(batch, device)
            logits = model(image)
            # probs = torch.sigmoid(logits)

            # Use logit threshold trick
            logit_threshold = (
                float(np.log(threshold / (1 - threshold))) if 0 < threshold < 1 else 0.0
            )

            batch_counts = masked_binary_confusion_counts(
                logits - logit_threshold, mask, valid_mask
            )
            for k in counts:
                counts[k] += batch_counts[k]

    return calculate_metrics(counts)["dice"]


def generate_figure(
    sample_id: str,
    image: np.ndarray,
    mask: np.ndarray,
    valid_mask: np.ndarray,
    prediction: np.ndarray,
    dem: np.ndarray,
    output_path: Path,
) -> None:
    # Build error map: 0=TN (black), 1=TP (green), 2=FP (red), 3=FN (blue), 4=Invalid (gray)
    error_map = np.zeros_like(mask, dtype=int)
    error_map[(mask == 1) & (prediction == 1)] = 1  # TP
    error_map[(mask == 0) & (prediction == 1)] = 2  # FP
    error_map[(mask == 1) & (prediction == 0)] = 3  # FN
    error_map[valid_mask == 0] = 4  # Invalid

    # Custom colormap for error map
    cmap = matplotlib.colors.ListedColormap(["black", "green", "red", "blue", "gray"])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    # Normalize S1
    finite = np.isfinite(image)
    s1_vis = np.zeros_like(image, dtype=np.float32)
    if finite.any():
        low, high = np.percentile(image[finite], [2, 98])
        if high > low:
            s1_vis = np.clip((image - low) / (high - low), 0, 1)

    # Normalize DEM
    dem_vis = np.zeros_like(dem, dtype=np.float32)
    dem_min, dem_max = 0.0, 0.0
    finite_dem = np.isfinite(dem)
    if finite_dem.any():
        dem_min, dem_max = dem[finite_dem].min(), dem[finite_dem].max()
        if dem_max > dem_min:
            dem_vis = np.clip((dem - dem_min) / (dem_max - dem_min), 0, 1)

    fig, axes = plt.subplots(1, 6, figsize=(18, 3))

    axes[0].imshow(s1_vis, cmap="gray")
    axes[0].set_title("Sentinel-1 (norm)")

    gt_vis = mask.copy().astype(np.float32)
    gt_vis[valid_mask == 0] = np.nan
    axes[1].imshow(gt_vis, cmap="viridis")
    axes[1].set_title("Ground truth")

    axes[2].imshow(prediction, cmap="gray")
    axes[2].set_title("Prediction")

    axes[3].imshow(error_map, cmap=cmap, norm=norm)
    axes[3].set_title("Error map")

    axes[4].imshow(dem_vis, cmap="terrain")
    axes[4].set_title(f"DEM ({dem_min:.0f}m - {dem_max:.0f}m)")

    axes[5].imshow(valid_mask, cmap="gray")
    axes[5].set_title("Valid mask")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)

    # Limit epochs to a reasonable number for demo
    epochs = min(args.epochs, 20)
    if args.epochs > 20:
        print(f"[INFO] Reduced epochs from {args.epochs} to 20 for quick qualitative demo.")

    dataset = GeoTIFFDataset(args.manifest, require_dem=True)
    train_indices = [i for i, r in enumerate(dataset.rows) if r.get("split") == "train"]
    val_indices = [i for i, r in enumerate(dataset.rows) if r.get("split") == "val"]

    if not train_indices or not val_indices:
        print("[WARNING] 'split' column missing or incomplete, splitting randomly.")
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        val_count = max(2, int(len(indices) * 0.25))
        val_indices = indices[:val_count]
        train_indices = indices[val_count:]

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=geotiff_collate_fn,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices), batch_size=1, shuffle=False, collate_fn=geotiff_collate_fn
    )

    sample = dataset[0]
    in_channels = sample["image"].shape[0]

    model = SimpleSegmentationCNN(in_channels=in_channels, hidden_channels=16).to(device)
    pos_weight = compute_class_balance(dataset, train_indices) if args.use_pos_weight else 1.0

    criterion = MaskedBCEDiceLoss(pos_weight=pos_weight).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"[INFO] Training SimpleCNN for {epochs} epochs on device {device}...")
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            image, mask, valid_mask, _ = move_batch(batch, device)
            optimizer.zero_grad()
            logits = model(image)
            loss = criterion(logits, mask, valid_mask)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}/{epochs} done.")

    # Threshold sweep
    best_threshold = 0.5
    if args.threshold == "auto":
        print("[INFO] Performing threshold sweep...")
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        best_dice = -1.0
        sweep_rows = []
        for th in thresholds:
            dice = evaluate_threshold(model, val_loader, th, device)
            sweep_rows.append({"threshold": th, "dice": dice})
            print(f"Threshold {th}: Dice {dice:.4f}")
            if dice > best_dice:
                best_dice = dice
                best_threshold = th
        print(f"[INFO] Selected best threshold: {best_threshold}")

        sweep_dir = args.output_dir / "threshold_sweep"
        sweep_dir.mkdir(parents=True, exist_ok=True)
        with (sweep_dir / "threshold_sweep.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["threshold", "dice"])
            writer.writeheader()
            writer.writerows(sweep_rows)

        with (sweep_dir / "threshold_sweep_best.md").open("w") as f:
            f.write(f"Best threshold: {best_threshold} (Dice: {best_dice:.4f})\n")
    else:
        best_threshold = float(args.threshold)

    # Evaluate validation samples
    model.eval()
    val_results = []

    logit_threshold = (
        float(np.log(best_threshold / (1 - best_threshold))) if 0 < best_threshold < 1 else 0.0
    )

    for batch in val_loader:
        sample_id = batch["sample_id"][0]
        image, mask, valid_mask, dem = move_batch(batch, device)
        with torch.no_grad():
            logits = model(image)
            prediction = (logits >= logit_threshold).float()

            counts = masked_binary_confusion_counts(logits - logit_threshold, mask, valid_mask)
            metrics = calculate_metrics(counts)
            topo = violation_rate_topo(logits=logits, dem=dem, valid_mask=valid_mask)[
                "violation_rate_topo"
            ]
            metrics["violation_rate_topo"] = float(topo)
            metrics["sample_id"] = sample_id

            val_results.append(
                {
                    "sample_id": sample_id,
                    "metrics": metrics,
                    "image": image[0, 0].cpu().numpy(),
                    "mask": mask[0, 0].cpu().numpy(),
                    "valid_mask": valid_mask[0, 0].cpu().numpy(),
                    "prediction": prediction[0, 0].cpu().numpy(),
                    "dem": dem[0, 0].cpu().numpy(),
                }
            )

    # Sort by Dice score
    val_results.sort(key=lambda x: x["metrics"]["dice"])

    if len(val_results) >= 3:
        best = val_results[-1]
        median = val_results[len(val_results) // 2]
        worst = val_results[0]
        selected = [("best", best), ("median", median), ("worst", worst)]
    elif len(val_results) > 0:
        selected = [("best", val_results[-1])]
    else:
        print("[ERROR] No validation samples found.")
        return 1

    figs_dir = args.output_dir / "figures"
    selected_rows = []

    for label, res in selected:
        fig_path = figs_dir / f"{label}_prediction.png"
        generate_figure(
            res["sample_id"],
            res["image"],
            res["mask"],
            res["valid_mask"],
            res["prediction"],
            res["dem"],
            fig_path,
        )
        m = res["metrics"]
        selected_rows.append(
            {
                "selection_type": label,
                "sample_id": res["sample_id"],
                "threshold": best_threshold,
                "iou": m["iou"],
                "dice": m["dice"],
                "f1": m["f1"],
                "recall": m["recall"],
                "precision": m["precision"],
                "violation_rate_topo": m["violation_rate_topo"],
                "target_positive_rate": m["target_positive_rate"],
                "predicted_positive_rate": m["predicted_positive_rate"],
                "figure_path": str(fig_path.resolve()),
            }
        )

    # Write CSV
    with (args.output_dir / "selected_predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=selected_rows[0].keys())
        writer.writeheader()
        writer.writerows(selected_rows)

    # Write Report
    report_lines = [
        "# Qualitative Demonstration Report",
        "",
        "> Cette démonstration qualitative vise à vérifier que le pipeline complet peut produire "
        "des prédictions visuellement interprétables sur de vraies données Sen1Floods11 "
        "avec DEM aligné. Elle ne constitue pas une évaluation finale de performance. "
        "Les résultats doivent être interprétés comme une validation préliminaire du "
        "fonctionnement du pipeline, avant l’élargissement à un sous-ensemble plus grand "
        "et à plusieurs seeds.",
        "",
        "## Configuration",
        f"- **Manifest**: `{args.manifest.name}`",
        "- **Model**: `SimpleSegmentationCNN`",
        f"- **Loss**: `{args.loss}` (Masked BCE + Dice)",
        f"- **Pos Weight**: `{'True' if args.use_pos_weight else 'False'}`",
        f"- **Epochs**: `{epochs}`",
        f"- **Selected Threshold**: `{best_threshold}`",
        "",
        "## Diagnostic Baselines",
        "Pour information, voici le comportement attendu des baselines triviales :",
        "- **all-background**: Pred+ = 0.0, IoU = 0.0, Dice = 0.0",
        "- **all-water**: Pred+ = 1.0, IoU = ~Target+, Dice = ~2*Target+",
        "",
        "## Selected Predictions",
    ]

    for row in selected_rows:
        label = row["selection_type"].capitalize()
        report_lines.extend(
            [
                f"### {label} Prediction ({row['sample_id']})",
                f"- **IoU**: {row['iou']:.4f}",
                f"- **Dice**: {row['dice']:.4f}",
                f"- **Target+**: {row['target_positive_rate']:.4f}",
                f"- **Pred+**: {row['predicted_positive_rate']:.4f}",
                f"- **Violation Topo**: {row['violation_rate_topo']:.4f}",
                f"![{label} Prediction]({row['figure_path']})",
                "",
            ]
        )

    (args.output_dir / "qualitative_demo_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    print("[OK] Demo successfully finished. Report written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
