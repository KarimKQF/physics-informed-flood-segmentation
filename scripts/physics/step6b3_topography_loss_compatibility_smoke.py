from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from losses.combined_loss import CombinedSegmentationPhysicsLoss  # noqa: E402
from losses.physics_topographic_loss import TopographicInconsistencyLoss  # noqa: E402


DEFAULT_RUN_DIR = Path(
    "E:/flood_research/experiments/terramind_baseline/runs/"
    "step6b3_sample_dem_alignment_qc"
)
DEFAULT_SAMPLE_MANIFEST = DEFAULT_RUN_DIR / "manifests" / "topography_sample_manifest.csv"
DEFAULT_SUMMARY = DEFAULT_RUN_DIR / "metrics" / "step6b3_loss_compatibility_smoke_summary.json"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Sample manifest not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_real_sample(row: dict[str, str]) -> tuple[torch.Tensor, torch.Tensor, dict]:
    with rasterio.open(row["label_path"]) as label_ds:
        label = label_ds.read(1).astype("int64")
    with rasterio.open(row["topography_path"]) as topo_ds:
        topography = topo_ds.read(1).astype("float32")

    target_np = np.where((label == 0) | (label == 1), label, -1).astype("int64")
    target = torch.from_numpy(target_np)[None]
    topo = torch.from_numpy(topography)[None]
    finite_ratio = float(np.isfinite(topography).sum() / topography.size)
    valid_ratio = float(((target_np == 0) | (target_np == 1)).sum() / target_np.size)
    return target, topo, {"finite_ratio": finite_ratio, "label_valid_ratio": valid_ratio}


def main() -> int:
    parser = argparse.ArgumentParser(description="STEP 6B3 real aligned topography loss smoke test.")
    parser.add_argument("--sample-manifest", type=Path, default=DEFAULT_SAMPLE_MANIFEST)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--seed", type=int, default=20260621)
    args = parser.parse_args()

    try:
        rows = [row for row in read_manifest(args.sample_manifest) if row.get("status") == "ok"]
        if not rows:
            summary = {
                "step": "6B3",
                "status": "blocked_no_valid_aligned_topography_rows",
                "generated_at": now_utc(),
                "sample_manifest": args.sample_manifest.as_posix(),
                "loss_smoke_passed": False,
                "training_started": False,
            }
            write_summary(args.summary_json, summary)
            print("status=blocked_no_valid_aligned_topography_rows")
            return 0

        row = rows[0]
        target, topography, sample_stats = load_real_sample(row)
        height, width = int(target.shape[-2]), int(target.shape[-1])
        torch.manual_seed(args.seed)

        logits_topo = torch.randn(1, 2, height, width, dtype=torch.float32, requires_grad=True)
        topo_loss_fn = TopographicInconsistencyLoss(neighborhood="4", elevation_scale=1.0)
        loss_topo = topo_loss_fn(logits=logits_topo, target=target, topography=topography)
        loss_topo.backward()
        topo_gradient_l1 = float(logits_topo.grad.detach().abs().sum())

        logits_combined = torch.randn(1, 2, height, width, dtype=torch.float32, requires_grad=True)
        combined_loss_fn = CombinedSegmentationPhysicsLoss(lambda_topo=0.05, neighborhood="4")
        combined = combined_loss_fn(logits=logits_combined, target=target, topography=topography)
        combined["loss_total"].backward()
        combined_gradient_l1 = float(logits_combined.grad.detach().abs().sum())

        loss_topo_value = float(loss_topo.detach())
        loss_total_value = float(combined["loss_total"].detach())
        loss_seg_value = float(combined["loss_seg"].detach())
        combined_topo_value = float(combined["loss_topo"].detach())
        passed = (
            torch.isfinite(loss_topo).item()
            and torch.isfinite(combined["loss_total"]).item()
            and topo_gradient_l1 > 0.0
            and combined_gradient_l1 > 0.0
        )

        summary = {
            "step": "6B3",
            "status": "passed" if passed else "failed",
            "generated_at": now_utc(),
            "sample_manifest": args.sample_manifest.as_posix(),
            "tile_id": row["tile_id"],
            "split": row["split"],
            "label_path": row["label_path"],
            "topography_path": row["topography_path"],
            "logits_shape": [1, 2, height, width],
            "target_shape": list(target.shape),
            "topography_shape": list(topography.shape),
            "loss_topo": loss_topo_value,
            "loss_total": loss_total_value,
            "loss_seg": loss_seg_value,
            "combined_loss_topo": combined_topo_value,
            "lambda_topo": float(combined["lambda_topo"].detach()),
            "topo_gradient_l1": topo_gradient_l1,
            "combined_gradient_l1": combined_gradient_l1,
            "gradient_nonzero": topo_gradient_l1 > 0.0 and combined_gradient_l1 > 0.0,
            "finite_ratio": sample_stats["finite_ratio"],
            "label_valid_ratio": sample_stats["label_valid_ratio"],
            "loss_smoke_passed": passed,
            "uses_real_topography": True,
            "uses_synthetic_logits": True,
            "training_started": False,
        }
        write_summary(args.summary_json, summary)
        print(f"status={summary['status']}")
        print(f"loss_topo={loss_topo_value:.6f}")
        print(f"loss_total={loss_total_value:.6f}")
        print(f"topo_gradient_l1={topo_gradient_l1:.6f}")
        print(f"combined_gradient_l1={combined_gradient_l1:.6f}")
        print(f"summary={args.summary_json}")
        return 0 if passed else 2
    except Exception as exc:  # noqa: BLE001
        summary = {
            "step": "6B3",
            "status": "failed_exception",
            "generated_at": now_utc(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "loss_smoke_passed": False,
            "training_started": False,
        }
        write_summary(args.summary_json, summary)
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
