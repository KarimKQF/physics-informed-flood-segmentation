from __future__ import annotations

import torch
from torch.nn import functional as F

from urban_runoff.losses import MaskedBCEWithLogitsLoss, masked_bce_with_logits_loss


def test_masked_bce_returns_finite_scalar_and_backward() -> None:
    logits = torch.randn(2, 1, 4, 4, requires_grad=True)
    targets = torch.randint(0, 2, (2, 1, 4, 4)).float()
    valid_mask = torch.ones_like(targets)

    loss = MaskedBCEWithLogitsLoss()(logits, targets, valid_mask)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_masked_bce_ignores_invalid_pixel_with_huge_error() -> None:
    logits = torch.tensor([[[[-10.0, 1000.0]]]], requires_grad=True)
    targets = torch.tensor([[[[1.0, 0.0]]]])
    valid_mask = torch.tensor([[[[1.0, 0.0]]]])

    loss = masked_bce_with_logits_loss(logits, targets, valid_mask)
    expected = F.binary_cross_entropy_with_logits(
        logits[..., :1],
        targets[..., :1],
        reduction="mean",
    )

    assert torch.allclose(loss, expected)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[..., 1].abs().item() == 0.0


def test_masked_bce_returns_controlled_zero_when_no_valid_pixels() -> None:
    logits = torch.randn(1, 1, 2, 2, requires_grad=True)
    targets = torch.zeros_like(logits)
    valid_mask = torch.zeros_like(logits)

    loss = masked_bce_with_logits_loss(logits, targets, valid_mask)

    assert loss.ndim == 0
    assert loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum().item() == 0.0
