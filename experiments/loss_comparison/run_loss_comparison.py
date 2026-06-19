from __future__ import annotations

# ruff: noqa: E402
import argparse
import csv
import json
import random
import sys
from collections.abc import Callable
from dataclasses import dataclass
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
from urban_runoff.losses import (
    BinaryTopographicGradientLoss,
    MaskedBCEDiceLoss,
    MaskedBCEWithLogitsLoss,
    MaskedDiceLoss,
    MaskedFocalLoss,
    MaskedTverskyLoss,
)
from urban_runoff.metrics import (
    masked_binary_confusion_counts,
    violation_rate_topo,
)
from urban_runoff.models import SimpleSegmentationCNN

LossFactory = Callable[[], nn.Module]


@dataclass(frozen=True)
class ExperimentSpec:
    experiment: str
    base_loss_name: str
    display_loss: str
    uses_topographic_loss: bool
    factory: LossFactory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare masked segmentation losses on GeoTIFFs.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha-topo", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use-pos-weight", action="store_true")
    parser.add_argument("--loss-debug", action="store_true")
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


def experiment_specs() -> list[ExperimentSpec]:
    base = [
        ("A1", "masked_bce", "Masked BCE", MaskedBCEWithLogitsLoss),
        ("A2", "masked_dice", "Masked Dice", MaskedDiceLoss),
        ("A3", "masked_bce_dice", "Masked BCE + Dice", MaskedBCEDiceLoss),
        ("A4", "masked_focal", "Masked Focal", MaskedFocalLoss),
        ("A5", "masked_tversky", "Masked Tversky", MaskedTverskyLoss),
    ]
    specs: list[ExperimentSpec] = [
        ExperimentSpec(code, name, display, False, factory) for code, name, display, factory in base
    ]
    specs.extend(
        ExperimentSpec(
            code.replace("A", "B", 1),
            name,
            f"{display} + Topo",
            True,
            factory,
        )
        for code, name, display, factory in base
    )
    return specs


def make_criterion(spec: ExperimentSpec, pos_weight: float | None) -> nn.Module:
    if spec.base_loss_name == "masked_bce":
        return MaskedBCEWithLogitsLoss(pos_weight=pos_weight)
    if spec.base_loss_name == "masked_bce_dice":
        return MaskedBCEDiceLoss(pos_weight=pos_weight)
    if spec.base_loss_name == "masked_focal":
        return MaskedFocalLoss(pos_weight=pos_weight)
    return spec.factory()


def split_indices(dataset: GeoTIFFDataset, seed: int) -> tuple[list[int], list[int], str]:
    train_indices = [index for index, row in enumerate(dataset.rows) if row.get("split") == "train"]
    val_indices = [index for index, row in enumerate(dataset.rows) if row.get("split") == "val"]
    if train_indices and val_indices:
        return train_indices, val_indices, "manifest"

    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_count = max(1, min(len(indices) - 1, round(len(indices) * 0.25)))
    val_indices = sorted(indices[:val_count])
    train_indices = sorted(indices[val_count:])
    return train_indices, val_indices, "controlled_seeded"


def make_loader(
    dataset: GeoTIFFDataset,
    indices: list[int],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=geotiff_collate_fn,
    )


def move_batch(
    batch: dict[str, object],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
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
    return (
        image.to(device=device, dtype=torch.float32),
        mask.to(device=device, dtype=torch.float32),
        valid_mask.to(device=device, dtype=torch.float32),
        dem.to(device=device, dtype=torch.float32),
    )


def compute_class_balance(dataset: GeoTIFFDataset, indices: list[int]) -> dict[str, float]:
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
    pos_weight = negative / positive if positive > 0 else 1.0
    return {
        "target_positive_rate_train": positive / valid if valid > 0 else 0.0,
        "num_positive_pixels": positive,
        "num_negative_pixels": negative,
        "valid_pixel_count_train": valid,
        "pos_weight": pos_weight,
    }


def print_loss_debug(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    experiment: str,
) -> None:
    batch = next(iter(loader))
    image, mask, valid_mask, _ = move_batch(batch, device)
    model.eval()
    with torch.no_grad():
        logits = model(image)
        probabilities = torch.sigmoid(logits)
        counts = masked_binary_confusion_counts(logits, mask, valid_mask)
    print(
        "[DEBUG] {experiment} target_positive_rate={target:.6f} "
        "predicted_positive_rate={pred:.6f} logits_min={logits_min:.6f} "
        "logits_max={logits_max:.6f} prob_min={prob_min:.6f} "
        "prob_max={prob_max:.6f} prob_mean={prob_mean:.6f}".format(
            experiment=experiment,
            target=counts["positive_pixel_count"] / max(counts["valid_pixel_count"], 1.0),
            pred=counts["predicted_positive_pixel_count"] / max(counts["valid_pixel_count"], 1.0),
            logits_min=float(logits.min().detach().cpu()),
            logits_max=float(logits.max().detach().cpu()),
            prob_min=float(probabilities.min().detach().cpu()),
            prob_max=float(probabilities.max().detach().cpu()),
            prob_mean=float(probabilities.mean().detach().cpu()),
        )
    )


def compute_loss(
    *,
    criterion: nn.Module,
    topographic: BinaryTopographicGradientLoss,
    spec: ExperimentSpec,
    logits: Tensor,
    mask: Tensor,
    valid_mask: Tensor,
    dem: Tensor,
    alpha_topo: float,
) -> tuple[Tensor, Tensor, Tensor]:
    base_loss = criterion(logits, mask, valid_mask)
    topo_loss = logits.sum() * 0.0
    if spec.uses_topographic_loss:
        topo_loss = topographic(logits=logits, target=mask, dem=dem, valid_mask=valid_mask)
    total_loss = base_loss + alpha_topo * topo_loss
    return total_loss, base_loss, topo_loss


def train_one_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    topographic: BinaryTopographicGradientLoss,
    spec: ExperimentSpec,
    alpha_topo: float,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    batches = 0
    for batch in loader:
        image, mask, valid_mask, dem = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(image)
        loss, _, _ = compute_loss(
            criterion=criterion,
            topographic=topographic,
            spec=spec,
            logits=logits,
            mask=mask,
            valid_mask=valid_mask,
            dem=dem,
            alpha_topo=alpha_topo,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss in {spec.experiment}")
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        batches += 1
    return total_loss / max(batches, 1)


def evaluate(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    topographic: BinaryTopographicGradientLoss,
    spec: ExperimentSpec,
    alpha_topo: float,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    batches = 0
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
    violation_rates: list[float] = []
    with torch.no_grad():
        for batch in loader:
            image, mask, valid_mask, dem = move_batch(batch, device)
            logits = model(image)
            loss, _, _ = compute_loss(
                criterion=criterion,
                topographic=topographic,
                spec=spec,
                logits=logits,
                mask=mask,
                valid_mask=valid_mask,
                dem=dem,
                alpha_topo=alpha_topo,
            )
            total_loss += float(loss.detach().cpu())
            batches += 1
            batch_counts = masked_binary_confusion_counts(logits, mask, valid_mask)
            for key in counts:
                counts[key] += batch_counts[key]
            violation_rates.append(
                violation_rate_topo(logits=logits, dem=dem, valid_mask=valid_mask)[
                    "violation_rate_topo"
                ]
            )

    metrics = metrics_from_counts(counts)
    metrics["val_loss"] = total_loss / max(batches, 1)
    metrics["val_violation_rate_topo"] = float(np.mean(violation_rates)) if violation_rates else 0.0
    return metrics


def metrics_from_counts(counts: dict[str, float], eps: float = 1e-6) -> dict[str, float]:
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]
    valid_pixel_count = counts["valid_pixel_count"]
    positive_pixel_count = counts["positive_pixel_count"]
    negative_pixel_count = counts["negative_pixel_count"]
    predicted_positive_pixel_count = counts["predicted_positive_pixel_count"]
    iou = tp / (tp + fp + fn + eps)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    recall = tp / (tp + fn + eps)
    precision = tp / (tp + fp + eps)
    target_positive_rate = positive_pixel_count / (valid_pixel_count + eps)
    predicted_positive_rate = predicted_positive_pixel_count / (valid_pixel_count + eps)
    return {
        "val_iou": float(iou),
        "val_dice": float(dice),
        "val_f1": float(dice),
        "val_recall": float(recall),
        "val_precision": float(precision),
        "val_tp": float(tp),
        "val_fp": float(fp),
        "val_fn": float(fn),
        "val_tn": float(tn),
        "val_target_positive_rate": float(target_positive_rate),
        "val_predicted_positive_rate": float(predicted_positive_rate),
        "val_valid_pixel_count": float(valid_pixel_count),
        "val_positive_pixel_count": float(positive_pixel_count),
        "val_negative_pixel_count": float(negative_pixel_count),
    }


def normalize_preview(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype="float32")
    low, high = np.percentile(values[finite], [2, 98])
    if high <= low:
        return np.zeros_like(values, dtype="float32")
    return np.clip((values - low) / (high - low), 0, 1).astype("float32")


def save_prediction_preview(
    *,
    model: nn.Module,
    loader: DataLoader,
    output_dir: Path,
    experiment: str,
    device: torch.device,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        image, mask, valid_mask, dem = move_batch(batch, device)
        logits = model(image)
        prediction = (torch.sigmoid(logits) >= 0.5).float()

    sample_ids = batch.get("sample_id", ["sample"])
    sample_id = str(sample_ids[0]) if isinstance(sample_ids, list) else "sample"
    panels = [
        ("Image band 1", normalize_preview(image[0, 0].detach().cpu().numpy())),
        ("Ground truth", mask[0, 0].detach().cpu().numpy()),
        ("Prediction", prediction[0, 0].detach().cpu().numpy()),
        ("Valid mask", valid_mask[0, 0].detach().cpu().numpy()),
        ("DEM", normalize_preview(dem[0, 0].detach().cpu().numpy())),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(14, 3))
    for axis, (title, values) in zip(axes, panels, strict=True):
        axis.imshow(values, cmap="gray")
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / f"{experiment}_{sample_id}.png", dpi=120)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(
    metrics_rows: list[dict[str, object]],
    selection_metric: str = "val_iou",
) -> list[dict[str, object]]:
    summary = []
    by_experiment: dict[str, list[dict[str, object]]] = {}
    for row in metrics_rows:
        by_experiment.setdefault(str(row["experiment"]), []).append(row)

    for rows in by_experiment.values():
        best = max(rows, key=lambda row: float(row[selection_metric]))
        final = rows[-1]
        summary.append(
            {
                "experiment": final["experiment"],
                "base_loss_name": final["base_loss_name"],
                "uses_topographic_loss": final["uses_topographic_loss"],
                "alpha_topo": final["alpha_topo"],
                "best_epoch": best["epoch"],
                "selection_metric": selection_metric,
                "final_epoch": final["epoch"],
                "final_train_loss": final["train_loss"],
                "final_val_loss": final["val_loss"],
                "final_iou": final["val_iou"],
                "final_dice": final["val_dice"],
                "final_recall": final["val_recall"],
                "final_precision": final["val_precision"],
                "final_predicted_positive_rate": final["val_predicted_positive_rate"],
                "best_val_iou": best["val_iou"],
                "best_val_dice": best["val_dice"],
                "best_val_f1": best["val_f1"],
                "best_val_recall": best["val_recall"],
                "best_val_precision": best["val_precision"],
                "best_predicted_positive_rate": best["val_predicted_positive_rate"],
                "best_val_violation_rate_topo": best["val_violation_rate_topo"],
            }
        )
    return summary


def write_markdown_table(path: Path, summary_rows: list[dict[str, object]]) -> None:
    labels = {
        "A1": "Masked BCE",
        "A2": "Masked Dice",
        "A3": "Masked BCE + Dice",
        "A4": "Masked Focal",
        "A5": "Masked Tversky",
        "B1": "Masked BCE + Topo",
        "B2": "Masked Dice + Topo",
        "B3": "Masked BCE + Dice + Topo",
        "B4": "Masked Focal + Topo",
        "B5": "Masked Tversky + Topo",
    }
    lines = [
        "| Experiment | Loss | Topo | BestEpoch | IoU | Dice | F1 | Recall | Precision | "
        "Pred+ | FinalPred+ | ViolationRateTopo |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        experiment = str(row["experiment"])
        topo = "Yes" if str(row["uses_topographic_loss"]) == "True" else "No"
        lines.append(
            "| {experiment} | {loss} | {topo} | {best_epoch} | {iou:.4f} | {dice:.4f} | "
            "{f1:.4f} | {recall:.4f} | {precision:.4f} | {pred:.4f} | {final_pred:.4f} | "
            "{violation:.4f} |".format(
                experiment=experiment,
                loss=labels.get(experiment, str(row["base_loss_name"])),
                topo=topo,
                best_epoch=row["best_epoch"],
                iou=float(row["best_val_iou"]),
                dice=float(row["best_val_dice"]),
                f1=float(row["best_val_f1"]),
                recall=float(row["best_val_recall"]),
                precision=float(row["best_val_precision"]),
                pred=float(row["best_predicted_positive_rate"]),
                final_pred=float(row["final_predicted_positive_rate"]),
                violation=float(row["best_val_violation_rate_topo"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.epochs <= 0:
        print("[ERROR] --epochs must be positive", file=sys.stderr)
        return 1

    try:
        seed_everything(args.seed)
        device = resolve_device(args.device)
        dataset = GeoTIFFDataset(args.manifest, max_samples=args.max_samples, require_dem=True)
        train_indices, val_indices, split_mode = split_indices(dataset, args.seed)
        sample = dataset[0]
        image = sample["image"]
        if not isinstance(image, Tensor):
            raise TypeError("Dataset image must be a tensor.")

        output_dir = args.output_dir
        predictions_dir = output_dir / "predictions"
        checkpoints_dir = output_dir / "checkpoints"
        output_dir.mkdir(parents=True, exist_ok=True)
        class_balance = compute_class_balance(dataset, train_indices)
        pos_weight = float(class_balance["pos_weight"]) if args.use_pos_weight else None
        config = {
            **vars(args),
            "manifest": args.manifest.as_posix(),
            "output_dir": output_dir.as_posix(),
            "device": str(device),
            "split_mode": split_mode,
            "train_indices": train_indices,
            "val_indices": val_indices,
            **class_balance,
            "effective_pos_weight": pos_weight,
        }
        (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        print(f"[INFO] Device: {device}")
        print(f"[INFO] Split mode: {split_mode}, train={train_indices}, val={val_indices}")
        print(
            "target_positive_rate_train={target_positive_rate_train:.6f} "
            "num_positive_pixels={num_positive_pixels:.0f} "
            "num_negative_pixels={num_negative_pixels:.0f} "
            "pos_weight={pos_weight:.6f}".format(**class_balance)
        )

        metrics_rows: list[dict[str, object]] = []
        specs = experiment_specs()
        for index, spec in enumerate(specs):
            print(f"[INFO] Running experiment: {spec.base_loss_name}")
            seed_everything(args.seed)
            train_loader = make_loader(
                dataset,
                train_indices,
                batch_size=args.batch_size,
                shuffle=True,
                seed=args.seed + index,
            )
            val_loader = make_loader(
                dataset,
                val_indices,
                batch_size=args.batch_size,
                shuffle=False,
                seed=args.seed,
            )
            model = SimpleSegmentationCNN(in_channels=image.shape[0]).to(device)
            criterion = make_criterion(spec, pos_weight).to(device)
            topographic = BinaryTopographicGradientLoss().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            best_metric = float("-inf")
            if args.loss_debug:
                print_loss_debug(
                    model=model,
                    loader=train_loader,
                    device=device,
                    experiment=spec.experiment,
                )

            for epoch in range(1, args.epochs + 1):
                print(f"[INFO] Epoch {epoch}/{args.epochs}")
                train_loss = train_one_epoch(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    topographic=topographic,
                    spec=spec,
                    alpha_topo=args.alpha_topo,
                    device=device,
                )
                val_metrics = evaluate(
                    model=model,
                    loader=val_loader,
                    criterion=criterion,
                    topographic=topographic,
                    spec=spec,
                    alpha_topo=args.alpha_topo,
                    device=device,
                )
                row = {
                    "experiment": spec.experiment,
                    "base_loss_name": spec.base_loss_name,
                    "uses_topographic_loss": spec.uses_topographic_loss,
                    "alpha_topo": args.alpha_topo if spec.uses_topographic_loss else 0.0,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    **val_metrics,
                }
                metrics_rows.append(row)
                if float(val_metrics["val_iou"]) > best_metric:
                    best_metric = float(val_metrics["val_iou"])
                    checkpoints_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "experiment": spec.experiment,
                            "base_loss_name": spec.base_loss_name,
                            "uses_topographic_loss": spec.uses_topographic_loss,
                            "epoch": epoch,
                            "selection_metric": "val_iou",
                            "selection_metric_value": best_metric,
                            "in_channels": int(image.shape[0]),
                            "threshold": 0.5,
                        },
                        checkpoints_dir / f"{spec.experiment}_best.pt",
                    )
                print(
                    "train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                    "val_iou={val_iou:.6f} val_dice={val_dice:.6f} "
                    "val_f1={val_f1:.6f} val_recall={val_recall:.6f} "
                    "val_precision={val_precision:.6f} "
                    "val_predicted_positive_rate={val_predicted_positive_rate:.6f} "
                    "val_violation_rate_topo={val_violation_rate_topo:.6f}".format(
                        train_loss=train_loss,
                        **val_metrics,
                    )
                )

            save_prediction_preview(
                model=model,
                loader=val_loader,
                output_dir=predictions_dir,
                experiment=spec.experiment,
                device=device,
            )

        metrics_fieldnames = [
            "experiment",
            "base_loss_name",
            "uses_topographic_loss",
            "alpha_topo",
            "epoch",
            "train_loss",
            "val_loss",
            "val_iou",
            "val_dice",
            "val_f1",
            "val_recall",
            "val_precision",
            "val_tp",
            "val_fp",
            "val_fn",
            "val_tn",
            "val_target_positive_rate",
            "val_predicted_positive_rate",
            "val_valid_pixel_count",
            "val_positive_pixel_count",
            "val_negative_pixel_count",
            "val_violation_rate_topo",
        ]
        summary_fieldnames = [
            "experiment",
            "base_loss_name",
            "uses_topographic_loss",
            "alpha_topo",
            "best_epoch",
            "selection_metric",
            "final_epoch",
            "final_train_loss",
            "final_val_loss",
            "final_iou",
            "final_dice",
            "final_recall",
            "final_precision",
            "final_predicted_positive_rate",
            "best_val_iou",
            "best_val_dice",
            "best_val_f1",
            "best_val_recall",
            "best_val_precision",
            "best_predicted_positive_rate",
            "best_val_violation_rate_topo",
        ]
        summary_rows = summarize(metrics_rows, selection_metric="val_iou")
        write_csv(output_dir / "metrics_per_epoch.csv", metrics_rows, metrics_fieldnames)
        write_csv(output_dir / "summary_results.csv", summary_rows, summary_fieldnames)
        write_markdown_table(output_dir / "loss_comparison_table.md", summary_rows)
        print(f"[OK] metrics_per_epoch.csv: {output_dir / 'metrics_per_epoch.csv'}")
        print(f"[OK] summary_results.csv: {output_dir / 'summary_results.csv'}")
        print(f"[OK] loss_comparison_table.md: {output_dir / 'loss_comparison_table.md'}")
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
