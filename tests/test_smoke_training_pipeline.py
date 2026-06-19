from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from urban_runoff.data import SmokeRasterDataset, create_synthetic_smoke_dataset
from urban_runoff.losses import BinaryTopographicGradientLoss
from urban_runoff.models import SimpleSegmentationCNN


def test_smoke_dataloader_returns_image_mask_and_dem_shapes(tmp_path: Path) -> None:
    create_synthetic_smoke_dataset(tmp_path, num_samples=2, height=16, width=20, overwrite=True)
    dataset = SmokeRasterDataset(tmp_path, include_dem=True)

    sample = dataset[0]
    image = sample["image"]
    mask = sample["mask"]
    dem = sample["dem"]

    assert isinstance(image, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert isinstance(dem, torch.Tensor)
    assert image.shape == (3, 16, 20)
    assert mask.shape == (1, 16, 20)
    assert dem.shape == (1, 16, 20)
    assert set(torch.unique(mask).tolist()).issubset({0, 1})


def test_smoke_dataloader_works_in_pytorch_dataloader(tmp_path: Path) -> None:
    create_synthetic_smoke_dataset(tmp_path, num_samples=2, height=16, width=20, overwrite=True)
    dataset = SmokeRasterDataset(tmp_path, include_dem=True)
    batch = next(iter(DataLoader(dataset, batch_size=2)))

    assert batch["image"].shape == (2, 3, 16, 20)
    assert batch["mask"].shape == (2, 1, 16, 20)
    assert batch["dem"].shape == (2, 1, 16, 20)


def test_simple_segmentation_model_output_shape() -> None:
    model = SimpleSegmentationCNN(in_channels=3)
    image = torch.randn(2, 3, 16, 20)
    logits = model(image)
    assert logits.shape == (2, 1, 16, 20)


def test_bce_with_logits_loss_computes_on_smoke_batch() -> None:
    logits = torch.randn(2, 1, 16, 20)
    target = torch.randint(0, 2, (2, 1, 16, 20)).float()
    loss = nn.BCEWithLogitsLoss()(logits, target)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_binary_topographic_loss_returns_finite_scalar_and_backward() -> None:
    logits = torch.randn(2, 1, 16, 20, requires_grad=True)
    target = torch.randint(0, 2, (2, 1, 16, 20)).float()
    dem = torch.randn(2, 1, 16, 20)

    loss = BinaryTopographicGradientLoss()(logits=logits, target=target, dem=dem)
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
