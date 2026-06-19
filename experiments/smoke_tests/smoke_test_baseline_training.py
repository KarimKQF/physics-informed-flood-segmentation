from __future__ import annotations

# ruff: noqa: E402, I001

import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from urban_runoff.data import SmokeRasterDataset, create_synthetic_smoke_dataset  # noqa: E402
from urban_runoff.models import SimpleSegmentationCNN  # noqa: E402


DATA_ROOT = PROJECT_ROOT / "experiments" / "smoke_tests" / "data_baseline"


def validate_batch_shapes(batch: dict[str, torch.Tensor | list[str]]) -> None:
    image = batch["image"]
    mask = batch["mask"]
    if not isinstance(image, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise TypeError("Batch image and mask must be tensors.")
    if image.ndim != 4:
        raise ValueError(f"image must have shape [B, C, H, W], got {tuple(image.shape)}.")
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"mask must have shape [B, 1, H, W], got {tuple(mask.shape)}.")
    if tuple(mask.shape[-2:]) != tuple(image.shape[-2:]):
        raise ValueError("image and mask spatial dimensions do not match.")


def main() -> None:
    torch.manual_seed(42)
    create_synthetic_smoke_dataset(DATA_ROOT, num_samples=4, overwrite=True)
    dataset = SmokeRasterDataset(DATA_ROOT, include_dem=False)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    print("[OK] Dataset loaded")

    first_batch = next(iter(dataloader))
    validate_batch_shapes(first_batch)
    print("[OK] Batch shapes valid")

    image = first_batch["image"]
    mask = first_batch["mask"]
    if not isinstance(image, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise TypeError("Batch image and mask must be tensors.")

    model = SimpleSegmentationCNN(in_channels=image.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    print("[OK] Model initialized")

    optimizer.zero_grad(set_to_none=True)
    logits = model(image)
    if tuple(logits.shape) != tuple(mask.shape):
        raise ValueError(
            f"Model output shape {tuple(logits.shape)} does not match {tuple(mask.shape)}."
        )
    print("[OK] Forward pass successful")

    loss = criterion(logits, mask)
    if not torch.isfinite(loss):
        raise RuntimeError("Classical loss is not finite.")
    print("[OK] Classical loss computed")

    loss.backward()
    print("[OK] Backward pass successful")
    optimizer.step()
    print("[OK] Optimizer step successful")

    epoch_losses: list[float] = []
    for epoch in range(1, 3):
        total_loss = 0.0
        num_batches = 0
        for batch in dataloader:
            image = batch["image"]
            mask = batch["mask"]
            if not isinstance(image, torch.Tensor) or not isinstance(mask, torch.Tensor):
                raise TypeError("Batch image and mask must be tensors.")
            validate_batch_shapes(batch)

            optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            loss = criterion(logits, mask)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            num_batches += 1

        average_loss = total_loss / max(num_batches, 1)
        epoch_losses.append(average_loss)
        print(f"Epoch {epoch}: average classical loss = {average_loss:.6f}")

    print("[OK] Mini baseline training loop completed")
    print("Baseline epoch losses: " + ", ".join(f"{value:.6f}" for value in epoch_losses))


if __name__ == "__main__":
    main()
