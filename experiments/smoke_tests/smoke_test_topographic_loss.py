from __future__ import annotations

# ruff: noqa: E402, I001

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from urban_runoff.data import SmokeRasterDataset, create_synthetic_smoke_dataset  # noqa: E402
from urban_runoff.losses import BinarySegmentationWithTopographicLoss  # noqa: E402
from urban_runoff.models import SimpleSegmentationCNN  # noqa: E402


DATA_ROOT = PROJECT_ROOT / "experiments" / "smoke_tests" / "data_topographic"


def validate_batch_shapes(batch: dict[str, torch.Tensor | list[str]]) -> None:
    image = batch["image"]
    mask = batch["mask"]
    dem = batch["dem"]
    if not all(isinstance(item, torch.Tensor) for item in (image, mask, dem)):
        raise TypeError("Batch image, mask and DEM must be tensors.")
    assert isinstance(image, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert isinstance(dem, torch.Tensor)
    if image.ndim != 4:
        raise ValueError(f"image must have shape [B, C, H, W], got {tuple(image.shape)}.")
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"mask must have shape [B, 1, H, W], got {tuple(mask.shape)}.")
    if dem.ndim != 4 or dem.shape[1] != 1:
        raise ValueError(f"dem must have shape [B, 1, H, W], got {tuple(dem.shape)}.")
    if tuple(mask.shape[-2:]) != tuple(image.shape[-2:]):
        raise ValueError("image and mask spatial dimensions do not match.")
    if tuple(dem.shape[-2:]) != tuple(image.shape[-2:]):
        raise ValueError("image and DEM spatial dimensions do not match.")


def main() -> None:
    torch.manual_seed(123)
    alpha_topo = 0.1
    create_synthetic_smoke_dataset(DATA_ROOT, num_samples=4, overwrite=True)
    dataset = SmokeRasterDataset(DATA_ROOT, include_dem=True)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    print("[OK] Dataset with DEM loaded")

    first_batch = next(iter(dataloader))
    validate_batch_shapes(first_batch)
    print("[OK] Batch shapes valid")
    print("[OK] DEM loaded")

    image = first_batch["image"]
    mask = first_batch["mask"]
    dem = first_batch["dem"]
    if not all(isinstance(item, torch.Tensor) for item in (image, mask, dem)):
        raise TypeError("Batch image, mask and DEM must be tensors.")
    assert isinstance(image, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert isinstance(dem, torch.Tensor)

    model = SimpleSegmentationCNN(in_channels=image.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = BinarySegmentationWithTopographicLoss(alpha_topo=alpha_topo)

    optimizer.zero_grad(set_to_none=True)
    logits = model(image)
    losses = criterion(logits=logits, target=mask, dem=dem)
    classical_loss = losses["loss_classical"]
    topographic_loss = losses["loss_topographic"]
    total_loss = losses["loss"]
    if not torch.isfinite(classical_loss):
        raise RuntimeError("Classical loss is not finite.")
    if not torch.isfinite(topographic_loss):
        raise RuntimeError("Topographic loss is not finite.")
    if not torch.isfinite(total_loss):
        raise RuntimeError("Total loss is not finite.")
    print("[OK] Classical loss computed")
    print("[OK] Topographic loss computed")
    print("[OK] Total loss computed")

    total_loss.backward()
    print("[OK] Backward pass successful")
    optimizer.step()
    print("[OK] Optimizer step successful")

    epoch_losses: list[float] = []
    last_classical = float(classical_loss.detach())
    last_topographic = float(topographic_loss.detach())
    last_total = float(total_loss.detach())

    for epoch in range(1, 3):
        total_epoch_loss = 0.0
        num_batches = 0
        for batch in dataloader:
            image = batch["image"]
            mask = batch["mask"]
            dem = batch["dem"]
            if not all(isinstance(item, torch.Tensor) for item in (image, mask, dem)):
                raise TypeError("Batch image, mask and DEM must be tensors.")
            assert isinstance(image, torch.Tensor)
            assert isinstance(mask, torch.Tensor)
            assert isinstance(dem, torch.Tensor)
            validate_batch_shapes(batch)

            optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            losses = criterion(logits=logits, target=mask, dem=dem)
            total_loss = losses["loss"]
            total_loss.backward()
            optimizer.step()

            last_classical = float(losses["loss_classical"].detach())
            last_topographic = float(losses["loss_topographic"].detach())
            last_total = float(total_loss.detach())
            total_epoch_loss += last_total
            num_batches += 1

        average_loss = total_epoch_loss / max(num_batches, 1)
        epoch_losses.append(average_loss)
        print(
            f"Epoch {epoch}: average total loss = {average_loss:.6f} "
            f"(last classical={last_classical:.6f}, "
            f"last topographic={last_topographic:.6f}, last total={last_total:.6f})"
        )

    print("[OK] Mini topographic training loop completed")
    print(f"Final classical loss: {last_classical:.6f}")
    print(f"Final topographic loss: {last_topographic:.6f}")
    print(f"Final total loss: {last_total:.6f}")
    print("Topographic epoch losses: " + ", ".join(f"{value:.6f}" for value in epoch_losses))


if __name__ == "__main__":
    main()
