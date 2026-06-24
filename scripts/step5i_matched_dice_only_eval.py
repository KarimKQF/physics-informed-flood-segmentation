"""
5I Dice-only matched baseline — inference-only evaluation on valid / test / Bolivia.

Loads best checkpoint (epoch 77) from the TerraMind Base + UNetDecoder + Dice-only run,
runs inference over valid, test, and Bolivia/OOD splits, writes metrics to JSON.

NO training. DEM used only for topographic metrics, never as model input.
lambda_topo=0.0 throughout (matched baseline for 5I+Physics comparison).
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for _p in (str(SRC_ROOT), str(SCRIPTS_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import step6c_lambda05_train as t6c       # noqa: E402
import step6c_v3_train as v3              # noqa: E402

RUN_DIR = Path("E:/flood_research/experiments/terramind_baseline/runs/step5i_matched_base_unetdecoder_dice_only")
CKPT_PATH = RUN_DIR / "checkpoints" / "best_checkpoint.pt"
CONFIG_PATH = RUN_DIR / "configs" / "step5i_matched_terramind_base_unetdecoder_dice_only.yaml"
METRICS_OUT = RUN_DIR / "metrics" / "step5i_matched_dice_only_test_bolivia_eval.json"
REPORTS_OUT = REPO_ROOT / "reports" / "STEP_5I_MATCHED_DICE_ONLY_TEST_BOLIVIA_EVAL_SUMMARY.json"

SPLITS = ["valid", "test", "bolivia"]


def json_safe(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [json_safe(x) for x in v]
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, int):
        return v
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_task_from_ckpt(config: dict[str, Any], ckpt_path: Path, device: torch.device) -> Any:
    task = t6c.build_task(config).to(device)
    print(f"  Loading checkpoint: {ckpt_path}", flush=True)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = task.load_state_dict(state, strict=True)
    if missing:
        print(f"  WARNING missing keys: {missing}", flush=True)
    if unexpected:
        print(f"  WARNING unexpected keys: {unexpected}", flush=True)
    t6c.set_bn_eval(task)
    task.eval()
    return task


@torch.no_grad()
def evaluate_one_split(task: Any, config: dict[str, Any], split: str, device: torch.device, batch_size: int, criterion: Any) -> dict[str, Any]:
    dm = v3.TopographyDataModule(config, batch_size=batch_size, split=split)
    dm.setup("test")
    loader = dm.test_dataloader()
    return v3.evaluate_split(task, criterion, loader, config, device)


def main() -> int:
    t0 = time.time()
    print("=" * 70, flush=True)
    print("5I Dice-only matched baseline — inference-only eval (valid / test / Bolivia)", flush=True)
    print(f"  checkpoint : {CKPT_PATH}", flush=True)
    print(f"  config     : {CONFIG_PATH}", flush=True)
    print("=" * 70, flush=True)

    if not CKPT_PATH.exists():
        print(f"ERROR: checkpoint not found: {CKPT_PATH}", flush=True)
        return 1

    with CONFIG_PATH.open("r", encoding="utf-8-sig") as fh:
        config = yaml.safe_load(fh)

    if config.get("dem", {}).get("use_as_model_input", False):
        print("ERROR: DEM as model input is forbidden.", flush=True)
        return 1
    print("  DEM as model input: FALSE  (verified from config)", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}", flush=True)

    print("\n[1] Building model and loading checkpoint ...", flush=True)
    task = build_task_from_ckpt(config, CKPT_PATH, device)

    physics = config["physics_loss"]
    criterion = v3.build_loss(config).to(device)
    print(f"  criterion: CombinedDicePhysicsLoss(lambda_topo={float(physics['lambda_topo'])})", flush=True)

    batch_size = int(config["trainer"]["batch_size"])
    print(f"  batch_size: {batch_size}", flush=True)

    results: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        print(f"\n[eval] split={split} ...", flush=True)
        t_split = time.time()
        m = evaluate_one_split(task, config, split, device, batch_size, criterion)
        elapsed = time.time() - t_split
        results[split] = m
        print(
            f"  mIoU={m['mean_iou']:.6f}  iou_water={m['iou_water']:.6f}"
            f"  f1_water={m['f1_water']:.6f}  pred_water={m.get('water_pred_pixels', 'n/a')}"
            f"  topo_viol_frac={m.get('topo_violation_fraction', float('nan')):.6f}"
            f"  elapsed={elapsed:.1f}s",
            flush=True,
        )

    total_elapsed = time.time() - t0
    print(f"\n[done] total elapsed={total_elapsed:.1f}s", flush=True)
    print("\n[guardrails]", flush=True)
    print("  dem_as_model_input       : FALSE", flush=True)
    print("  raw_data_modified        : FALSE", flush=True)
    print("  training_launched        : FALSE", flush=True)
    print("  existing_runs_overwritten: FALSE", flush=True)

    payload = {
        "step": "5I-matched-dice-only-eval",
        "checkpoint": str(CKPT_PATH),
        "config": str(CONFIG_PATH),
        "best_epoch": 77,
        "dem_as_model_input": False,
        "dem_used_for_metric_only_eval": True,
        "lambda_topo_eval": float(physics["lambda_topo"]),
        "device": str(device),
        "batch_size": batch_size,
        "splits_evaluated": SPLITS,
        "elapsed_seconds": round(total_elapsed, 2),
        "metrics": {split: results[split] for split in SPLITS},
        "guardrails": {
            "training_launched": False,
            "dem_as_model_input": False,
            "raw_data_modified": False,
            "existing_runs_overwritten": False,
        },
    }
    write_json(METRICS_OUT, payload)
    print(f"\n[written] {METRICS_OUT}", flush=True)

    summary = {
        "step": "5I-matched-dice-only-eval",
        "generated": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": str(CKPT_PATH),
        "best_epoch": 77,
        "dem_as_model_input": False,
        "lambda_topo_eval": float(physics["lambda_topo"]),
        "metrics": {
            split: {
                "miou": m["mean_iou"],
                "iou_background": m["iou_background"],
                "iou_water": m["iou_water"],
                "f1_water": m["f1_water"],
                "precision_water": m.get("precision_water"),
                "recall_water": m.get("recall_water"),
                "water_pred_pixels": m.get("water_pred_pixels"),
                "pred_water_fraction": m.get("pred_water_fraction"),
                "loss_dice": m.get("loss_dice"),
                "loss_topo": m.get("loss_topo"),
                "topographic_inconsistency_score": m.get("topographic_inconsistency_score"),
                "topo_violation_fraction": m.get("topo_violation_fraction"),
                "topo_descending_pair_count": m.get("topo_descending_pair_count"),
                "topo_violation_pair_count": m.get("topo_violation_pair_count"),
                "batches": m.get("batches"),
                "batchnorm_eval_modules": m.get("batchnorm_eval_modules"),
            }
            for split, m in results.items()
        },
    }
    write_json(REPORTS_OUT, summary)
    print(f"[written] {REPORTS_OUT}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
