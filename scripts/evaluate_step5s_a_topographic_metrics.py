"""
Evaluate STEP 5S-A topographic metrics without training.

This helper reuses the STEP 6C/v3 loss-only DEM dataloader path to compute
topographic metrics for a STEP 5S-A checkpoint. DEM is not passed as model input.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for path in (str(SRC_ROOT), str(SCRIPTS_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import step6c_v3_train as v3  # noqa: E402


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value).replace("\\", "/")
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["valid", "test", "bolivia"], choices=["valid", "test", "bolivia"])
    parser.add_argument("--lambda-topo", type=float, default=0.0, help="Use 0.0 for STEP 5S-A Dice-only loss total; loss_topo is still reported.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.config.open("r", encoding="utf-8-sig") as handle:
        config = yaml.safe_load(handle)

    if config.get("dem", {}).get("use_as_model_input", False):
        raise RuntimeError("DEM as model input is forbidden for this evaluation.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(config["trainer"]["batch_size"])

    task = v3.t6c.build_task(config).to(device)
    criterion = v3.build_loss(config).to(device)
    criterion.set_lambda_topo(float(args.lambda_topo))

    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location=device)
    task.load_state_dict(checkpoint["model_state_dict"])
    task.eval()
    v3.t6c.set_bn_eval(task)

    split_metrics: dict[str, Any] = {}
    for split in args.splits:
        dm = v3.TopographyDataModule(config, batch_size=batch_size, split=split)
        dm.setup("test")
        split_metrics[split] = v3.evaluate_split(task, criterion, dm.test_dataloader(), config, device)

    output_dir = args.output_dir / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "step5s_a_topographic_metrics.json"
    csv_path = output_dir / "step5s_a_topographic_metrics.csv"

    payload = {
        "step": "5S-A-topographic-eval-only",
        "checkpoint": args.checkpoint,
        "config": args.config,
        "dem_as_model_input": False,
        "dem_used_for_metric_only": True,
        "lambda_topo_eval": float(args.lambda_topo),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_epoch": checkpoint.get("best_epoch"),
        "metrics": split_metrics,
    }
    json_path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    rows = []
    for split, metrics in split_metrics.items():
        rows.append(
            {
                "split": split,
                "miou": metrics["mean_iou"],
                "iou_water": metrics["iou_water"],
                "f1_water": metrics["f1_water"],
                "predicted_water_pixels": metrics["water_pred_pixels"],
                "predicted_water_fraction": metrics["pred_water_fraction"],
                "loss_dice": metrics["loss_dice"],
                "loss_topo": metrics["loss_topo"],
                "topo_score": metrics["topographic_inconsistency_score"],
                "topo_violation_fraction": metrics["topo_violation_fraction"],
            }
        )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
