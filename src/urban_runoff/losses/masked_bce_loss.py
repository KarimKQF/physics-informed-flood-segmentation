from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def masked_bce_with_logits_loss(
    logits: Tensor,
    targets: Tensor,
    valid_mask: Tensor,
    *,
    eps: float = 1e-6,
    pos_weight: float | Tensor | None = None,
) -> Tensor:
    if logits.shape != targets.shape:
        raise ValueError(
            f"logits and targets must have the same shape, got {logits.shape} and {targets.shape}."
        )
    if logits.shape != valid_mask.shape:
        raise ValueError(
            "logits and valid_mask must have the same shape, "
            f"got {logits.shape} and {valid_mask.shape}."
        )
    if eps <= 0:
        raise ValueError("eps must be positive.")

    valid = valid_mask.to(device=logits.device, dtype=logits.dtype)
    targets = targets.to(device=logits.device, dtype=logits.dtype)
    pos_weight_tensor = None
    if pos_weight is not None:
        pos_weight_tensor = torch.as_tensor(pos_weight, device=logits.device, dtype=logits.dtype)
    pixel_loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=pos_weight_tensor,
    )
    denominator = valid.sum()
    if denominator <= 0:
        return logits.sum() * 0.0
    return (pixel_loss * valid).sum() / denominator.clamp_min(eps)


class MaskedBCEWithLogitsLoss(nn.Module):
    """Binary cross-entropy averaged over valid pixels only."""

    def __init__(self, *, eps: float = 1e-6, pos_weight: float | Tensor | None = None) -> None:
        super().__init__()
        self.eps = eps
        self.pos_weight = pos_weight

    def forward(self, logits: Tensor, targets: Tensor, valid_mask: Tensor) -> Tensor:
        return masked_bce_with_logits_loss(
            logits=logits,
            targets=targets,
            valid_mask=valid_mask,
            eps=self.eps,
            pos_weight=self.pos_weight,
        )
