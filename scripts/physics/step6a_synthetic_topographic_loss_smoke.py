from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from losses.combined_loss import CombinedSegmentationPhysicsLoss  # noqa: E402
from losses.physics_topographic_loss import TopographicInconsistencyLoss  # noqa: E402


RUN_DIR = Path(
    "E:/flood_research/experiments/terramind_baseline/runs/"
    "step6a_physics_topographic_loss_implementation"
)
SUMMARY_PATH = RUN_DIR / "smoke" / "step6a_synthetic_loss_smoke_summary.json"


def main() -> int:
    torch.manual_seed(42)
    RUN_DIR.joinpath("smoke").mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    water_scores = torch.tensor(
        [
            [
                [4.0, -4.0, -4.0, -4.0],
                [3.5, -3.0, -3.0, -3.0],
                [-2.0, -2.0, 2.5, -2.0],
                [-2.0, -2.0, -2.0, 2.0],
            ]
        ],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    logits = torch.stack([torch.zeros_like(water_scores), water_scores], dim=1)
    target = torch.tensor(
        [[[1, 0, 0, 0], [1, 0, -1, 0], [0, 0, 1, 0], [0, 0, 0, 1]]],
        dtype=torch.long,
        device=device,
    )
    topography = torch.tensor(
        [[[8.0, 5.0, 4.0, 3.0], [7.0, 4.0, float("nan"), 2.0], [6.0, 3.0, 2.0, 1.0], [5.0, 2.0, 1.0, 0.0]]],
        dtype=torch.float32,
        device=device,
    )

    topo_loss_fn = TopographicInconsistencyLoss(neighborhood="4", elevation_scale=1.0)
    combined_loss_fn = CombinedSegmentationPhysicsLoss(lambda_topo=0.05, neighborhood="4")

    topo_loss = topo_loss_fn(logits=logits, target=target, topography=topography)
    combined = combined_loss_fn(logits=logits, target=target, topography=topography)
    combined["loss_total"].backward()

    gradient_l1 = water_scores.grad.detach().abs().sum().item()
    summary = {
        "step": "6A",
        "status": "passed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "logits_shape": list(logits.shape),
        "target_shape": list(target.shape),
        "topography_shape": list(topography.shape),
        "topographic_loss": float(topo_loss.detach().cpu()),
        "loss_total": float(combined["loss_total"].detach().cpu()),
        "loss_seg": float(combined["loss_seg"].detach().cpu()),
        "loss_topo": float(combined["loss_topo"].detach().cpu()),
        "lambda_topo": float(combined["lambda_topo"].detach().cpu()),
        "backward_ok": water_scores.grad is not None and gradient_l1 > 0.0,
        "gradient_l1": gradient_l1,
        "uses_real_topography": False,
        "training_started": False,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"loss_total={summary['loss_total']:.6f}")
    print(f"loss_seg={summary['loss_seg']:.6f}")
    print(f"loss_topo={summary['loss_topo']:.6f}")
    print(f"lambda_topo={summary['lambda_topo']:.6f}")
    print(f"gradient_l1={summary['gradient_l1']:.6f}")
    print(f"summary={SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
