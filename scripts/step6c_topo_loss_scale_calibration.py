from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import random
import sys
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


CONFIG_PATH = REPO_ROOT / "configs" / "step6c_terramind_l_upernet_dice_topographic_lambda005.yaml"
REPORT_PATH = REPO_ROOT / "reports" / "STEP_6C_TOPO_LOSS_SCALE_CALIBRATION_REPORT.md"
RUN_DIR = Path(
    "E:/flood_research/experiments/terramind_baseline/runs/"
    "step6c_terramind_l_upernet_dice_topographic_lambda005"
)
JSON_PATH = RUN_DIR / "metrics" / "topo_loss_scale_calibration.json"
CSV_PATH = RUN_DIR / "metrics" / "topo_loss_scale_calibration.csv"
LOG_PATH = RUN_DIR / "logs" / "topo_loss_scale_calibration.log"

LAMBDA_VALUES = [0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0]
TARGET_RATIOS = {
    "0.1_percent": 0.001,
    "1_percent": 0.01,
    "5_percent": 0.05,
}
EXTREMELY_SMALL_TOPO_THRESHOLD = 1e-6


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def path_text(path: Path) -> str:
    return path.as_posix()


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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return yaml.safe_load(handle)


def append_log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = f"{dt.datetime.now().isoformat(timespec='seconds')} {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def build_datamodule(config: dict[str, Any], *, batch_size: int):
    from albumentations.pytorch.transforms import ToTensorV2
    from terratorch.datamodules import GenericMultiModalDataModule

    args = dict(config["data"]["init_args"])
    args["batch_size"] = batch_size
    args["num_workers"] = 0
    args["pin_memory"] = True
    # D4 is applied explicitly in this script so the DEM receives the same operation.
    args["train_transform"] = [ToTensorV2()]
    args["val_transform"] = None
    args["test_transform"] = None
    return GenericMultiModalDataModule(**args)


def build_sequential_train_loader(dm: Any, *, batch_size: int):
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset=dm.train_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=dm.collate_fn,
    )


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
            for param in child.parameters(recurse=False):
                param.requires_grad = False
            count += 1
    return count


def freeze_parameters(module: Any) -> int:
    count = 0
    for param in module.parameters():
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


def output_tensor(model_output: Any) -> torch.Tensor:
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


def apply_d4_to_sample(batch: dict[str, Any], dem: torch.Tensor, *, index: int, op: int) -> None:
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
def topographic_pair_stats(
    *,
    target: torch.Tensor,
    topography: torch.Tensor,
    ignore_index: int,
    elevation_margin: float,
    neighborhood: str,
) -> dict[str, float | int]:
    valid_pixel = (target != ignore_index) & torch.isfinite(topography)
    valid_neighbor_pairs = 0.0
    descending_pair_count = 0.0
    delta_sum = 0.0
    delta_sq_sum = 0.0
    delta_max = 0.0

    def add_direction(h_high: torch.Tensor, h_low: torch.Tensor, valid_pair: torch.Tensor) -> None:
        nonlocal descending_pair_count, delta_sum, delta_sq_sum, delta_max
        safe_delta = torch.where(
            valid_pair,
            h_high - h_low - elevation_margin,
            torch.zeros_like(h_high),
        )
        descending = valid_pair & (safe_delta > 0)
        count = float(descending.sum().detach().cpu())
        if count <= 0:
            return
        deltas = safe_delta[descending]
        descending_pair_count += count
        delta_sum += float(deltas.sum().detach().cpu())
        delta_sq_sum += float((deltas * deltas).sum().detach().cpu())
        delta_max = max(delta_max, float(deltas.max().detach().cpu()))

    for dy, dx in neighbor_offsets(neighborhood):
        h_a, h_b = crop_pair(topography, dy, dx)
        valid_a, valid_b = crop_pair(valid_pixel, dy, dx)
        valid_pair = valid_a & valid_b
        valid_neighbor_pairs += float(valid_pair.sum().detach().cpu())
        add_direction(h_a, h_b, valid_pair)
        add_direction(h_b, h_a, valid_pair)

    mean_delta = delta_sum / descending_pair_count if descending_pair_count > 0 else 0.0
    variance = max(delta_sq_sum / descending_pair_count - mean_delta * mean_delta, 0.0) if descending_pair_count > 0 else 0.0
    return {
        "valid_neighbor_pair_count": int(valid_neighbor_pairs),
        "valid_descending_pair_count": int(descending_pair_count),
        "descending_pair_fraction": descending_pair_count / valid_neighbor_pairs if valid_neighbor_pairs > 0 else 0.0,
        "positive_delta_mean": mean_delta,
        "positive_delta_std": math.sqrt(variance),
        "positive_delta_max": delta_max,
    }


def summarize(values: list[float]) -> dict[str, float | int | None]:
    finite = np.asarray([v for v in values if math.isfinite(float(v))], dtype=np.float64)
    if finite.size == 0:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None, "std": None}
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "std": float(finite.std(ddof=0)),
    }


def ratio_assessment(mean_ratio: float) -> str:
    if mean_ratio < 0.001:
        return "too weak"
    if mean_ratio <= 0.05:
        return "reasonable"
    return "too strong"


def recommend_first_run_lambda(lambda_stats: dict[str, Any]) -> float:
    """Pick a conservative lambda from the calibrated sweep.

    Scalar loss ratios are useful for bookkeeping, but optimizer dynamics are
    governed by gradients. The first physics run should be strong enough to be
    visible without letting the topo term dominate at initialization.
    """
    candidates = []
    for lambda_key, stats in lambda_stats.items():
        lambda_value = float(lambda_key)
        scaled_grad_mean = float(stats["scaled_topo_to_dice_grad_ratio"]["mean"])
        candidates.append((lambda_value, scaled_grad_mean))
    in_band = [item for item in candidates if 0.05 <= item[1] <= 0.30]
    if in_band:
        return sorted(in_band, key=lambda item: abs(item[1] - 0.20))[0][0]
    return min(candidates, key=lambda item: abs(item[1] - 0.20))[0]


def make_report(summary: dict[str, Any]) -> str:
    lambda_lines = [
        "| lambda_topo | mean contribution | mean loss ratio | median loss ratio | mean scaled grad ratio |",
        "|---:|---:|---:|---:|---:|",
    ]
    for lambda_key, stats in summary["lambda_stats"].items():
        lambda_lines.append(
            "| {lam:.2f} | {contrib:.8f} | {mean:.6%} | {median:.6%} | {grad:.4f} |".format(
                lam=float(lambda_key),
                contrib=stats["contribution"]["mean"],
                mean=stats["ratio"]["mean"],
                median=stats["ratio"]["median"],
                grad=stats["scaled_topo_to_dice_grad_ratio"]["mean"],
            )
        )

    rec = summary["recommended_lambdas"]
    grad = summary["gradient_norms"]
    pair = summary["topographic_pair_stats"]
    water = summary["water_probability"]
    paths = summary["paths"]
    return f"""# STEP 6C Topographic Loss Scale Calibration

Generated: {summary["generated_at"]}

## Scope

This calibration used the STEP 6C TerraMind-L + UPerNet setup with Dice segmentation loss and `TopographicInconsistencyLoss`. It ran forward passes only, did not create an optimizer, did not call `optimizer.step()`, did not update weights, did not use DEM as model input, and did not modify raw data. DEM was loaded only for the loss.

The train transform family was preserved as D4 + ToTensorV2. For calibration, ToTensorV2 was handled by the dataloader and the D4 operation was applied explicitly to images, masks, and DEM together so the augmented DEM stayed aligned with the augmented labels.

## Configuration

- Config: `{paths["config"]}`
- Script: `{paths["script"]}`
- JSON metrics: `{paths["json"]}`
- CSV metrics: `{paths["csv"]}`
- Log: `{paths["log"]}`
- Batches requested: {summary["requested_batches"]}
- Batches used: {summary["batches_used"]}
- Batch size: {summary["batch_size"]}
- Samples used: {summary["samples_used"]}
- Device: {summary["device"]}
- Backbone initialization: original TerraMind pretrained checkpoint
- Feature indices: `[5, 11, 17, 23]`
- DEM as model input: false

## Loss Scale

| Metric | Mean | Median | Min | Max | Std |
|---|---:|---:|---:|---:|---:|
| loss_dice | {summary["loss_dice"]["mean"]:.8f} | {summary["loss_dice"]["median"]:.8f} | {summary["loss_dice"]["min"]:.8f} | {summary["loss_dice"]["max"]:.8f} | {summary["loss_dice"]["std"]:.8f} |
| loss_topo | {summary["loss_topo"]["mean"]:.8f} | {summary["loss_topo"]["median"]:.8f} | {summary["loss_topo"]["min"]:.8f} | {summary["loss_topo"]["max"]:.8f} | {summary["loss_topo"]["std"]:.8f} |

## Lambda Sweep

{chr(10).join(lambda_lines)}

## Recommendations

Using `lambda_needed = target_ratio * mean(loss_dice) / mean(loss_topo)`:

| Target topo contribution | Recommended lambda |
|---:|---:|
| 0.1% of Dice | {rec["0.1_percent"]:.4f} |
| 1% of Dice | {rec["1_percent"]:.4f} |
| 5% of Dice | {rec["5_percent"]:.4f} |

Original `lambda_topo=0.05` mean scalar-loss ratio: {summary["original_lambda_mean_ratio"]:.8%}. Assessment by scalar loss: **{summary["original_lambda_assessment"]}**. Its mean scaled logit-gradient ratio is {summary["original_lambda_scaled_grad_ratio"]:.4f}, so it is not a zero-gradient term, but it is very weak as a contribution to the reported scalar loss.

The scalar-loss lambdas above are very large because the initial topographic loss value is tiny. They should not be used blindly: at `lambda_topo={rec["1_percent"]:.4f}`, the initial topographic logit-gradient norm would be hundreds of times larger than Dice on average.

Recommended first full STEP 6C run: **lambda_topo={summary["recommended_first_run_lambda"]:.4f}**, chosen from the tested sweep as a gradient-aware first run.

## Topographic Pair Diagnostics

| Metric | Mean | Median | Min | Max | Std |
|---|---:|---:|---:|---:|---:|
| valid descending pairs | {pair["valid_descending_pair_count"]["mean"]:.2f} | {pair["valid_descending_pair_count"]["median"]:.2f} | {pair["valid_descending_pair_count"]["min"]:.2f} | {pair["valid_descending_pair_count"]["max"]:.2f} | {pair["valid_descending_pair_count"]["std"]:.2f} |
| descending pair fraction | {pair["descending_pair_fraction"]["mean"]:.6f} | {pair["descending_pair_fraction"]["median"]:.6f} | {pair["descending_pair_fraction"]["min"]:.6f} | {pair["descending_pair_fraction"]["max"]:.6f} | {pair["descending_pair_fraction"]["std"]:.6f} |
| positive elevation delta mean | {pair["positive_delta_mean"]["mean"]:.8f} | {pair["positive_delta_mean"]["median"]:.8f} | {pair["positive_delta_mean"]["min"]:.8f} | {pair["positive_delta_mean"]["max"]:.8f} | {pair["positive_delta_mean"]["std"]:.8f} |

- Fraction of batches with `loss_topo == 0`: {summary["zero_topo_loss_fraction"]:.2%}
- Fraction of batches with `loss_topo < {EXTREMELY_SMALL_TOPO_THRESHOLD:g}`: {summary["extremely_small_topo_loss_fraction"]:.2%}

## Gradient Diagnostics

Gradients were measured with respect to logits, not model parameters. This checks loss scale without updating weights.

| Metric | Mean | Median | Min | Max | Std |
|---|---:|---:|---:|---:|---:|
| Dice grad L2 | {grad["dice_grad_l2"]["mean"]:.8f} | {grad["dice_grad_l2"]["median"]:.8f} | {grad["dice_grad_l2"]["min"]:.8f} | {grad["dice_grad_l2"]["max"]:.8f} | {grad["dice_grad_l2"]["std"]:.8f} |
| Topo grad L2 | {grad["topo_grad_l2"]["mean"]:.8f} | {grad["topo_grad_l2"]["median"]:.8f} | {grad["topo_grad_l2"]["min"]:.8f} | {grad["topo_grad_l2"]["max"]:.8f} | {grad["topo_grad_l2"]["std"]:.8f} |
| Topo/Dice grad ratio | {grad["topo_to_dice_grad_ratio"]["mean"]:.8f} | {grad["topo_to_dice_grad_ratio"]["median"]:.8f} | {grad["topo_to_dice_grad_ratio"]["min"]:.8f} | {grad["topo_to_dice_grad_ratio"]["max"]:.8f} | {grad["topo_to_dice_grad_ratio"]["std"]:.8f} |

- Topographic-loss gradients finite in all batches: {summary["topo_grad_finite_all_batches"]}
- Topographic-loss gradient norm nonzero in all batches: {summary["topo_grad_nonzero_all_batches"]}

## Probability Diagnostics

| Metric | Mean | Median | Min | Max | Std |
|---|---:|---:|---:|---:|---:|
| p_water mean per batch | {water["p_water_mean"]["mean"]:.8f} | {water["p_water_mean"]["median"]:.8f} | {water["p_water_mean"]["min"]:.8f} | {water["p_water_mean"]["max"]:.8f} | {water["p_water_mean"]["std"]:.8f} |
| p_water std per batch | {water["p_water_std"]["mean"]:.8f} | {water["p_water_std"]["median"]:.8f} | {water["p_water_std"]["min"]:.8f} | {water["p_water_std"]["max"]:.8f} | {water["p_water_std"]["std"]:.8f} |

## Interpretation

`loss_topo` is small mainly because the original TerraMind pretrained initialization is highly saturated toward the water class on these batches. Since the loss term contains `p_high_water * (1 - p_low_water)`, predictions that are nearly water everywhere make the scalar topo loss tiny even when many descending DEM pairs exist. The loss also averages over valid descending neighbor pairs. The valid pair mask is not sparse, positive elevation deltas are nonzero, and the topographic gradient is finite and nonzero, so this looks like a scale-calibration issue rather than an obvious bug.

STEP 6C full training is methodologically ready after choosing a calibrated lambda. Do not use the scalar-loss 1% lambda directly as the first run; use **lambda_topo={summary["recommended_first_run_lambda"]:.4f}** first, then sweep upward/downward after checking stability and physical metrics.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="STEP 6C topographic loss scale calibration.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--max-batches", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "metrics").mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "logs").mkdir(parents=True, exist_ok=True)
    if LOG_PATH.exists():
        LOG_PATH.unlink()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = read_yaml(args.config)
    batch_size = int(config["trainer"]["batch_size"])
    device = torch.device(args.device)
    append_log("STEP 6C topo loss scale calibration started")
    append_log(f"config={args.config}")
    append_log(f"device={device}")
    append_log("guardrails: no optimizer, no optimizer.step, no weight updates, DEM loss-only")

    dm = build_datamodule(config, batch_size=batch_size)
    dm.setup("fit")
    train_loader = build_sequential_train_loader(dm, batch_size=batch_size)
    total_available_batches = len(train_loader)
    batches_to_use = min(args.max_batches, total_available_batches)

    task = build_task(config).to(device)
    frozen_params = freeze_parameters(task)
    task.train()
    bn_eval_count = set_batchnorm_eval(task)
    append_log(f"frozen_parameters={frozen_params}")
    append_log(f"batchnorm_eval_modules={bn_eval_count}")
    append_log(f"train_batches_available={total_available_batches}; batches_to_use={batches_to_use}")

    physics_cfg = config["physics_loss"]
    loss_fn = CombinedDicePhysicsLoss(
        lambda_topo=float(physics_cfg["lambda_topo"]),
        ignore_index=int(physics_cfg["ignore_index"]),
        water_class=int(physics_cfg["water_class"]),
        elevation_margin=float(physics_cfg["elevation_margin"]),
        elevation_scale=float(physics_cfg["elevation_scale"]),
        use_elevation_weight=bool(physics_cfg["use_elevation_weight"]),
        neighborhood=str(physics_cfg["neighborhood"]),
    )

    rows: list[dict[str, Any]] = []
    used_tile_ids: list[str] = []
    d4_rng = random.Random(args.seed)

    for batch_index, batch in enumerate(train_loader, start=1):
        if batch_index > batches_to_use:
            break

        tile_ids = [tile_id_from_batch(batch, i) for i in range(len(batch["mask"]))]
        dem_cpu = torch.stack(
            [load_dem(dem_path_for_sample(config, split="train", tile_id=tile_id)) for tile_id in tile_ids],
            dim=0,
        )
        d4_ops = [d4_rng.randrange(8) for _ in tile_ids]
        for sample_index, op in enumerate(d4_ops):
            apply_d4_to_sample(batch, dem_cpu, index=sample_index, op=op)

        batch = move_batch(batch, device)
        dem = dem_cpu.to(device, non_blocking=True)
        target = task.squeeze_ground_truth(batch["mask"]).long()

        with torch.no_grad():
            model_output = task(batch["image"])
            raw_logits = output_tensor(model_output)
        logits = raw_logits.detach().float().requires_grad_(True)

        loss_parts = loss_fn(logits=logits, target=target, topography=dem)
        loss_dice = loss_parts["loss_dice"]
        loss_topo = loss_parts["loss_topo"]

        if not torch.isfinite(loss_dice).item() or not torch.isfinite(loss_topo).item():
            raise RuntimeError(
                f"Non-finite calibration loss at batch={batch_index}: "
                f"dice={float(loss_dice.detach().cpu())}, topo={float(loss_topo.detach().cpu())}"
            )

        grad_dice = torch.autograd.grad(loss_dice, logits, retain_graph=True)[0]
        grad_topo = torch.autograd.grad(loss_topo, logits, retain_graph=False)[0]
        dice_grad_l2 = float(torch.linalg.vector_norm(grad_dice.detach()).cpu())
        topo_grad_l2 = float(torch.linalg.vector_norm(grad_topo.detach()).cpu())
        topo_grad_finite = bool(torch.isfinite(grad_topo).all().item())
        topo_grad_nonzero = topo_grad_l2 > 0.0

        with torch.no_grad():
            p_water = torch.softmax(logits.detach(), dim=1)[:, int(physics_cfg["water_class"])]
            pair_stats = topographic_pair_stats(
                target=target,
                topography=dem,
                ignore_index=int(physics_cfg["ignore_index"]),
                elevation_margin=float(physics_cfg["elevation_margin"]),
                neighborhood=str(physics_cfg["neighborhood"]),
            )

        loss_dice_value = float(loss_dice.detach().cpu())
        loss_topo_value = float(loss_topo.detach().cpu())
        row: dict[str, Any] = {
            "batch_index": batch_index,
            "tile_ids": ";".join(tile_ids),
            "d4_ops": ";".join(str(op) for op in d4_ops),
            "loss_dice": loss_dice_value,
            "loss_topo": loss_topo_value,
            "dice_grad_l2": dice_grad_l2,
            "topo_grad_l2": topo_grad_l2,
            "topo_to_dice_grad_ratio": topo_grad_l2 / dice_grad_l2 if dice_grad_l2 > 0 else math.nan,
            "topo_grad_finite": topo_grad_finite,
            "topo_grad_nonzero": topo_grad_nonzero,
            "p_water_mean": float(p_water.mean().detach().cpu()),
            "p_water_std": float(p_water.std(unbiased=False).detach().cpu()),
            "p_water_min": float(p_water.min().detach().cpu()),
            "p_water_max": float(p_water.max().detach().cpu()),
            **pair_stats,
        }
        for lambda_value in LAMBDA_VALUES:
            contribution = lambda_value * loss_topo_value
            ratio = contribution / loss_dice_value if loss_dice_value > 0 else math.nan
            scaled_grad_ratio = lambda_value * row["topo_to_dice_grad_ratio"]
            lambda_label = f"lambda_{lambda_value:g}".replace(".", "p")
            row[f"{lambda_label}_contribution"] = contribution
            row[f"{lambda_label}_ratio"] = ratio
            row[f"{lambda_label}_scaled_grad_ratio"] = scaled_grad_ratio
        rows.append(row)
        used_tile_ids.extend(tile_ids)
        append_log(
            f"batch={batch_index}/{batches_to_use} dice={loss_dice_value:.6f} "
            f"topo={loss_topo_value:.8f} lambda0.05_ratio={row['lambda_0p05_ratio']:.8%} "
            f"topo_grad_l2={topo_grad_l2:.6e}"
        )

        del logits, raw_logits, model_output, loss_parts, grad_dice, grad_topo, batch, dem, dem_cpu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(rows) < min(30, total_available_batches):
        raise RuntimeError(f"Calibration produced only {len(rows)} batches; expected at least 30 where available.")

    loss_dice_values = [float(row["loss_dice"]) for row in rows]
    loss_topo_values = [float(row["loss_topo"]) for row in rows]
    mean_dice = float(np.mean(loss_dice_values))
    mean_topo = float(np.mean(loss_topo_values))
    recommended_lambdas = {
        key: (target_ratio * mean_dice / mean_topo if mean_topo > 0 else math.nan)
        for key, target_ratio in TARGET_RATIOS.items()
    }

    lambda_stats: dict[str, Any] = {}
    for lambda_value in LAMBDA_VALUES:
        lambda_label = f"lambda_{lambda_value:g}".replace(".", "p")
        contributions = [float(row[f"{lambda_label}_contribution"]) for row in rows]
        ratios = [float(row[f"{lambda_label}_ratio"]) for row in rows]
        scaled_grad_ratios = [float(row[f"{lambda_label}_scaled_grad_ratio"]) for row in rows]
        lambda_stats[f"{lambda_value:g}"] = {
            "contribution": summarize(contributions),
            "ratio": summarize(ratios),
            "scaled_topo_to_dice_grad_ratio": summarize(scaled_grad_ratios),
        }

    original_mean_ratio = float(lambda_stats["0.05"]["ratio"]["mean"])
    original_scaled_grad_ratio = float(lambda_stats["0.05"]["scaled_topo_to_dice_grad_ratio"]["mean"])
    recommended_first = recommend_first_run_lambda(lambda_stats)

    summary = {
        "step": "6C",
        "status": "completed",
        "generated_at": now_utc(),
        "seed": args.seed,
        "requested_batches": args.max_batches,
        "batches_used": len(rows),
        "samples_used": len(used_tile_ids),
        "unique_samples_used": len(set(used_tile_ids)),
        "batch_size": batch_size,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" and torch.cuda.is_available() else None,
        "peak_vram_mb": (
            float(torch.cuda.max_memory_allocated(0) / 1024**2)
            if device.type == "cuda" and torch.cuda.is_available()
            else None
        ),
        "paths": {
            "script": path_text(Path(__file__)),
            "config": path_text(args.config),
            "report": path_text(REPORT_PATH),
            "json": path_text(JSON_PATH),
            "csv": path_text(CSV_PATH),
            "log": path_text(LOG_PATH),
            "run_dir": path_text(RUN_DIR),
        },
        "guardrails": {
            "optimizer_created": False,
            "optimizer_step_called": False,
            "weights_updated": False,
            "full_training_started": False,
            "dem_as_model_input": False,
            "dem_in_loss_only": True,
            "raw_data_modified": False,
            "darn_started": False,
            "sturm_training_started": False,
        },
        "model_setup": {
            "backbone": config["model"]["init_args"]["model_args"]["backbone"],
            "decoder": config["model"]["init_args"]["model_args"]["decoder"],
            "feature_indices": config["model"]["init_args"]["model_args"]["necks"][0]["indices"],
            "backbone_ckpt_path": config["model"]["init_args"]["model_args"]["backbone_ckpt_path"],
            "initialization_rule": "original_terramind_pretrained_checkpoint",
            "frozen_parameter_tensors_for_calibration": frozen_params,
            "batchnorm_eval_modules": bn_eval_count,
        },
        "loss_config": physics_cfg,
        "loss_dice": summarize(loss_dice_values),
        "loss_topo": summarize(loss_topo_values),
        "lambda_stats": lambda_stats,
        "recommended_lambdas": recommended_lambdas,
        "original_lambda": 0.05,
        "original_lambda_mean_ratio": original_mean_ratio,
        "original_lambda_scaled_grad_ratio": original_scaled_grad_ratio,
        "original_lambda_assessment": ratio_assessment(original_mean_ratio),
        "recommended_first_run_lambda": recommended_first,
        "zero_topo_loss_fraction": float(np.mean([row["loss_topo"] == 0.0 for row in rows])),
        "extremely_small_topo_loss_threshold": EXTREMELY_SMALL_TOPO_THRESHOLD,
        "extremely_small_topo_loss_fraction": float(
            np.mean([row["loss_topo"] < EXTREMELY_SMALL_TOPO_THRESHOLD for row in rows])
        ),
        "topo_grad_finite_all_batches": all(bool(row["topo_grad_finite"]) for row in rows),
        "topo_grad_nonzero_all_batches": all(bool(row["topo_grad_nonzero"]) for row in rows),
        "gradient_norms": {
            "dice_grad_l2": summarize([float(row["dice_grad_l2"]) for row in rows]),
            "topo_grad_l2": summarize([float(row["topo_grad_l2"]) for row in rows]),
            "topo_to_dice_grad_ratio": summarize([float(row["topo_to_dice_grad_ratio"]) for row in rows]),
        },
        "water_probability": {
            "p_water_mean": summarize([float(row["p_water_mean"]) for row in rows]),
            "p_water_std": summarize([float(row["p_water_std"]) for row in rows]),
            "p_water_min": summarize([float(row["p_water_min"]) for row in rows]),
            "p_water_max": summarize([float(row["p_water_max"]) for row in rows]),
        },
        "topographic_pair_stats": {
            "valid_neighbor_pair_count": summarize([float(row["valid_neighbor_pair_count"]) for row in rows]),
            "valid_descending_pair_count": summarize([float(row["valid_descending_pair_count"]) for row in rows]),
            "descending_pair_fraction": summarize([float(row["descending_pair_fraction"]) for row in rows]),
            "positive_delta_mean": summarize([float(row["positive_delta_mean"]) for row in rows]),
            "positive_delta_std": summarize([float(row["positive_delta_std"]) for row in rows]),
            "positive_delta_max": summarize([float(row["positive_delta_max"]) for row in rows]),
        },
        "batch_rows": rows,
    }

    csv_fieldnames = list(rows[0].keys()) if rows else []
    write_csv(CSV_PATH, rows, csv_fieldnames)
    write_json(JSON_PATH, summary)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(make_report(summary), encoding="utf-8")

    append_log(f"status=completed batches={len(rows)}")
    append_log(f"json={JSON_PATH}")
    append_log(f"csv={CSV_PATH}")
    append_log(f"report={REPORT_PATH}")
    append_log(f"recommended_first_run_lambda={recommended_first:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
