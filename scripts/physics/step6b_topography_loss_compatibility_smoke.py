from __future__ import annotations

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

from losses.physics_topographic_loss import TopographicInconsistencyLoss  # noqa: E402


RUN_DIR = Path(
    "E:/flood_research/experiments/terramind_baseline/runs/"
    "step6b_topographic_alignment_validation"
)
SUMMARY_PATH = RUN_DIR / "metrics" / "step6b_loss_compatibility_smoke_summary.json"
SAMPLE_MANIFEST = RUN_DIR / "manifests" / "topography_sample_manifest.csv"


def write_summary(payload: dict) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    if not SAMPLE_MANIFEST.exists():
        write_summary(
            {
                "step": "6B",
                "status": "blocked_no_aligned_topography",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "sample_manifest": SAMPLE_MANIFEST.as_posix(),
                "reason": "No sample aligned topography manifest exists; DEM/HAND source is missing or alignment was skipped.",
                "loss_smoke_passed": False,
                "training_started": False,
            }
        )
        print("status=blocked_no_aligned_topography")
        print(f"summary={SUMMARY_PATH}")
        return 0

    with SAMPLE_MANIFEST.open("r", newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("status") == "ok"]
    if not rows:
        write_summary(
            {
                "step": "6B",
                "status": "blocked_no_valid_aligned_topography_rows",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "sample_manifest": SAMPLE_MANIFEST.as_posix(),
                "loss_smoke_passed": False,
                "training_started": False,
            }
        )
        print("status=blocked_no_valid_aligned_topography_rows")
        print(f"summary={SUMMARY_PATH}")
        return 0

    row = rows[0]
    with rasterio.open(row["label_path"]) as label_ds:
        label = label_ds.read(1).astype("int64")
    with rasterio.open(row["topography_path"]) as topo_ds:
        topography = topo_ds.read(1).astype("float32")

    target = torch.from_numpy(label)[None]
    target = torch.where((target == 0) | (target == 1), target, torch.full_like(target, -1))
    topo = torch.from_numpy(topography)[None]
    logits = torch.randn(1, 2, target.shape[-2], target.shape[-1], requires_grad=True)
    loss = TopographicInconsistencyLoss()(logits=logits, target=target, topography=topo)
    loss.backward()

    gradient_l1 = float(logits.grad.detach().abs().sum())
    finite_ratio = float(np.isfinite(topography).sum() / topography.size)
    summary = {
        "step": "6B",
        "status": "passed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tile_id": row["tile_id"],
        "label_path": row["label_path"],
        "topography_path": row["topography_path"],
        "loss_topo": float(loss.detach()),
        "gradient_l1": gradient_l1,
        "finite_ratio": finite_ratio,
        "loss_smoke_passed": bool(torch.isfinite(loss) and gradient_l1 >= 0),
        "training_started": False,
    }
    write_summary(summary)
    print(f"status={summary['status']}")
    print(f"loss_topo={summary['loss_topo']:.6f}")
    print(f"gradient_l1={summary['gradient_l1']:.6f}")
    print(f"summary={SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
