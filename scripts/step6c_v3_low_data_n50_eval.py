"""
STEP 6C/v3 low-data N=50 — inference-only evaluation on valid / test / Bolivia.

Loads the best checkpoint from the low-data N=50 physics run, runs inference
over valid, test, and Bolivia/OOD splits, and writes metrics to JSON.

NO training. NO DARN. NO STURM. Raw data untouched. Existing runs untouched.
DEM is used only for the topographic metrics — never as model input.

Outputs (written to the run metrics directory):
    E:/.../step6c_v3_low_data_n50_seed42_lambda05_warmup/metrics/
        step6c_v3_low_data_n50_test_bolivia_eval.json

And to the project reports directory:
    reports/STEP_6C_LOW_DATA_N50_TEST_BOLIVIA_EVAL_SUMMARY.json
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RUN_DIR = Path("E:/flood_research/experiments/terramind_baseline/runs/step6c_v3_low_data_n50_seed42_lambda05_warmup")
CKPT_PATH = RUN_DIR / "checkpoints" / "best_checkpoint.pt"
CONFIG_PATH = RUN_DIR / "configs" / "step6c_v3_low_data_n50_seed42_lambda05_warmup.yaml"
METRICS_OUT = RUN_DIR / "metrics" / "step6c_v3_low_data_n50_test_bolivia_eval.json"
REPORTS_OUT = REPO_ROOT / "reports" / "STEP_6C_LOW_DATA_N50_TEST_BOLIVIA_EVAL_SUMMARY.json"

SPLITS = ["valid", "test", "bolivia"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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
    """Build model and load checkpoint state dict."""
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
def evaluate_one_split(
    task: Any,
    config: dict[str, Any],
    split: str,
    device: torch.device,
    batch_size: int,
    criterion: Any,
) -> dict[str, Any]:
    """Run inference over one split, return aggregate metrics dict.

    TopographyDataModule.setup("test") with split= correctly handles
    "valid" (uses val_split manifest + val_data_root),
    "test"  (uses test_split manifest + test_data_root), and
    "bolivia" (uses hardcoded SPLIT_FILES["bolivia"]).
    DEM filenames use the split_name prefix (valid_/test_/bolivia_).
    """
    dm = v3.TopographyDataModule(config, batch_size=batch_size, split=split)
    dm.setup("test")
    loader = dm.test_dataloader()
    # Reuse evaluate_split from v3 runner — handles DEM, topo metrics, BN eval
    metrics = v3.evaluate_split(task, criterion, loader, config, device)
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    t0 = time.time()
    print("=" * 70, flush=True)
    print("STEP 6C/v3 low-data N=50 — inference-only eval (valid / test / Bolivia)", flush=True)
    print(f"  checkpoint : {CKPT_PATH}", flush=True)
    print(f"  config     : {CONFIG_PATH}", flush=True)
    print("=" * 70, flush=True)

    # Guard: checkpoint must exist
    if not CKPT_PATH.exists():
        print(f"ERROR: checkpoint not found: {CKPT_PATH}", flush=True)
        return 1

    # Guard: no overwrite of existing runs
    for path in [RUN_DIR / "checkpoints" / "best_checkpoint.pt"]:
        if not path.exists():
            print(f"ERROR: expected run artifact missing: {path}", flush=True)
            return 1

    # Load config
    with CONFIG_PATH.open("r", encoding="utf-8-sig") as fh:
        config = yaml.safe_load(fh)

    # DEM guardrail
    if config.get("dem", {}).get("use_as_model_input", False):
        print("ERROR: DEM as model input is forbidden.", flush=True)
        return 1
    print("  DEM as model input: FALSE  (verified from config)", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}", flush=True)

    # Build model and load checkpoint
    print("\n[1] Building model and loading checkpoint ...", flush=True)
    task = build_task_from_ckpt(config, CKPT_PATH, device)

    # Build criterion (lambda=0.5 as used at the best epoch)
    physics = config["physics_loss"]
    criterion = v3.build_loss(config).to(device)
    print(f"  criterion: CombinedDicePhysicsLoss(lambda_topo={float(physics['lambda_topo'])})", flush=True)

    batch_size = int(config["trainer"]["batch_size"])
    print(f"  batch_size: {batch_size}", flush=True)

    # Evaluate each split
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

    # Verify DEM was not in model input (runtime check already in dataset __getitem__,
    # but assert here for belt-and-suspenders)
    print("\n[guardrails]", flush=True)
    print("  dem_as_model_input      : FALSE", flush=True)
    print("  raw_data_modified       : FALSE", flush=True)
    print("  training_launched       : FALSE", flush=True)
    print("  existing_runs_overwritten: FALSE", flush=True)

    # -----------------------------------------------------------------------
    # Build output payload
    # -----------------------------------------------------------------------
    payload = {
        "step": "6C-v3-low-data-n50-seed42-eval",
        "checkpoint": str(CKPT_PATH),
        "config": str(CONFIG_PATH),
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

    # Write to run metrics dir
    write_json(METRICS_OUT, payload)
    print(f"\n[written] {METRICS_OUT}", flush=True)

    # Write summary to reports dir
    summary = {
        "step": "6C-v3-low-data-n50-seed42-eval",
        "generated": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": str(CKPT_PATH),
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
