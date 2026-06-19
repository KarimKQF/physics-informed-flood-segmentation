from __future__ import annotations

import pytest
import torch
from torch import nn

from urban_runoff.losses import (
    MaskedBCEDiceLoss,
    MaskedBCEWithLogitsLoss,
    MaskedDiceLoss,
    MaskedFocalLoss,
    MaskedTverskyLoss,
)


@pytest.mark.parametrize(
    "criterion",
    [
        MaskedBCEWithLogitsLoss(),
        MaskedDiceLoss(),
        MaskedBCEDiceLoss(),
        MaskedFocalLoss(),
        MaskedTverskyLoss(),
    ],
)
def test_masked_losses_are_finite_scalars_and_backward(criterion: nn.Module) -> None:
    logits = torch.randn(2, 1, 5, 6, requires_grad=True)
    targets = torch.randint(0, 2, (2, 1, 5, 6)).float()
    valid_mask = torch.ones_like(targets)
    valid_mask[..., 0, 0] = 0

    loss = criterion(logits, targets, valid_mask)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize(
    "criterion",
    [
        MaskedBCEWithLogitsLoss(),
        MaskedDiceLoss(),
        MaskedBCEDiceLoss(),
        MaskedFocalLoss(),
        MaskedTverskyLoss(),
    ],
)
def test_masked_losses_ignore_invalid_pixels(criterion: nn.Module) -> None:
    logits_a = torch.zeros(1, 1, 2, 3, requires_grad=True)
    logits_b = logits_a.detach().clone().requires_grad_(True)
    targets_a = torch.zeros(1, 1, 2, 3)
    targets_b = targets_a.clone()
    valid_mask = torch.ones_like(targets_a)
    valid_mask[..., 0, 0] = 0

    with torch.no_grad():
        logits_b[..., 0, 0] = 100.0
    targets_b[..., 0, 0] = 1.0

    loss_a = criterion(logits_a, targets_a, valid_mask)
    loss_b = criterion(logits_b, targets_b, valid_mask)

    assert torch.allclose(loss_a, loss_b, atol=1e-6)


def test_masked_losses_return_zero_when_no_valid_pixels() -> None:
    logits = torch.randn(1, 1, 3, 3, requires_grad=True)
    targets = torch.ones_like(logits)
    valid_mask = torch.zeros_like(logits)

    loss = MaskedDiceLoss()(logits, targets, valid_mask)
    loss.backward()

    assert loss.item() == 0.0
    assert logits.grad is not None
