from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from urban_runoff.losses.masked_bce_loss import MaskedBCEWithLogitsLoss


def _validate_inputs(logits: Tensor, targets: Tensor, valid_mask: Tensor) -> tuple[Tensor, Tensor]:
    if logits.shape != targets.shape:
        raise ValueError(
            "logits and targets must have the same shape, "
            f"got {tuple(logits.shape)} and {tuple(targets.shape)}."
        )
    if logits.shape != valid_mask.shape:
        raise ValueError(
            "logits and valid_mask must have the same shape, "
            f"got {tuple(logits.shape)} and {tuple(valid_mask.shape)}."
        )
    targets = targets.to(device=logits.device, dtype=logits.dtype)
    valid = valid_mask.to(device=logits.device, dtype=logits.dtype).clamp(0, 1)
    return targets, valid


def _zero_loss(logits: Tensor) -> Tensor:
    return logits.sum() * 0.0


class MaskedDiceLoss(nn.Module):
    """Soft Dice loss over valid pixels only."""

    def __init__(self, *, eps: float = 1e-6) -> None:
        super().__init__()
        if eps <= 0:
            raise ValueError("eps must be positive.")
        self.eps = eps

    def forward(self, logits: Tensor, targets: Tensor, valid_mask: Tensor) -> Tensor:
        targets, valid = _validate_inputs(logits, targets, valid_mask)
        if valid.sum() <= 0:
            return _zero_loss(logits)

        probabilities = torch.sigmoid(logits)
        probabilities = probabilities * valid
        targets = targets * valid
        intersection = (probabilities * targets).sum()
        denominator = probabilities.sum() + targets.sum()
        dice = (2.0 * intersection + self.eps) / denominator.clamp_min(self.eps)
        loss = 1.0 - dice
        if not torch.isfinite(loss):
            return _zero_loss(logits)
        return loss


class MaskedBCEDiceLoss(nn.Module):
    """Weighted combination of masked BCE and masked Dice losses."""

    def __init__(
        self,
        *,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        eps: float = 1e-6,
        pos_weight: float | Tensor | None = None,
    ) -> None:
        super().__init__()
        if bce_weight < 0 or dice_weight < 0:
            raise ValueError("loss weights must be non-negative.")
        if bce_weight + dice_weight <= 0:
            raise ValueError("at least one loss weight must be positive.")
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = MaskedBCEWithLogitsLoss(eps=eps, pos_weight=pos_weight)
        self.dice = MaskedDiceLoss(eps=eps)

    def forward(self, logits: Tensor, targets: Tensor, valid_mask: Tensor) -> Tensor:
        bce_loss = self.bce(logits, targets, valid_mask)
        dice_loss = self.dice(logits, targets, valid_mask)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class MaskedFocalLoss(nn.Module):
    """Binary focal loss over valid pixels only."""

    def __init__(
        self,
        *,
        alpha: float = 0.25,
        gamma: float = 2.0,
        eps: float = 1e-6,
        pos_weight: float | Tensor | None = None,
    ) -> None:
        super().__init__()
        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be in [0, 1].")
        if gamma < 0:
            raise ValueError("gamma must be non-negative.")
        if eps <= 0:
            raise ValueError("eps must be positive.")
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.pos_weight = pos_weight

    def forward(self, logits: Tensor, targets: Tensor, valid_mask: Tensor) -> Tensor:
        targets, valid = _validate_inputs(logits, targets, valid_mask)
        denominator = valid.sum()
        if denominator <= 0:
            return _zero_loss(logits)

        pos_weight_tensor = None
        if self.pos_weight is not None:
            pos_weight_tensor = torch.as_tensor(
                self.pos_weight,
                device=logits.device,
                dtype=logits.dtype,
            )
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=pos_weight_tensor,
        )
        probabilities = torch.sigmoid(logits)
        p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal = alpha_t * (1.0 - p_t).clamp_min(0.0).pow(self.gamma) * bce
        loss = (focal * valid).sum() / denominator.clamp_min(self.eps)
        if not torch.isfinite(loss):
            return _zero_loss(logits)
        return loss


class MaskedTverskyLoss(nn.Module):
    """Tversky loss over valid pixels only."""

    def __init__(self, *, alpha: float = 0.7, beta: float = 0.3, eps: float = 1e-6) -> None:
        super().__init__()
        if alpha < 0 or beta < 0:
            raise ValueError("alpha and beta must be non-negative.")
        if eps <= 0:
            raise ValueError("eps must be positive.")
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, logits: Tensor, targets: Tensor, valid_mask: Tensor) -> Tensor:
        targets, valid = _validate_inputs(logits, targets, valid_mask)
        if valid.sum() <= 0:
            return _zero_loss(logits)

        probabilities = torch.sigmoid(logits)
        probabilities = probabilities * valid
        targets = targets * valid
        true_positive = (probabilities * targets).sum()
        false_positive = (probabilities * (1.0 - targets) * valid).sum()
        false_negative = ((1.0 - probabilities) * targets).sum()
        denominator = true_positive + self.alpha * false_positive + self.beta * false_negative
        tversky = (true_positive + self.eps) / denominator.clamp_min(self.eps)
        loss = 1.0 - tversky
        if not torch.isfinite(loss):
            return _zero_loss(logits)
        return loss
