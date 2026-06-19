from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from urban_runoff.losses import SegmentationWithTopographicLoss  # noqa: E402


def run_on_device(device: torch.device) -> None:
    batch_size, height, width = 2, 16, 16
    logits = torch.randn(batch_size, 2, height, width, device=device, requires_grad=True)
    dem = torch.randn(batch_size, height, width, device=device) * 2.0
    target = torch.randint(0, 2, (batch_size, height, width), device=device)
    target[:, 0, 0] = 255

    criterion = SegmentationWithTopographicLoss(
        class_weights=[1.0, 8.0],
        ignore_index=255,
        lambda_topo=0.1,
        use_eight_neighbors=True,
        height_scale=1.0,
        max_weight=10.0,
    ).to(device)

    out = criterion(logits=logits, target=target, dem=dem)
    out["loss"].backward()
    grad_ok = (
        logits.grad is not None
        and torch.isfinite(logits.grad).all()
        and logits.grad.abs().sum() > 0
    )

    print(f"device used: {device}")
    print(f"loss total: {out['loss'].item():.6f}")
    print(f"loss segmentation: {out['loss_seg'].item():.6f}")
    print(f"loss topographique: {out['loss_topo'].item():.6f}")
    print(f"gradient OK: {bool(grad_ok)}")

    if not grad_ok:
        raise RuntimeError("Gradient check failed.")


def main() -> None:
    run_on_device(torch.device("cpu"))
    if torch.cuda.is_available():
        run_on_device(torch.device("cuda"))


if __name__ == "__main__":
    main()
