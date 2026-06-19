from __future__ import annotations

import torch
from torch import Tensor, nn


def _as_b1hw(tensor: Tensor, name: str) -> Tensor:
    if tensor.ndim == 3:
        return tensor[:, None]
    if tensor.ndim == 4 and tensor.shape[1] == 1:
        return tensor
    raise ValueError(
        f"{name} must have shape [B, H, W] or [B, 1, H, W], got {tuple(tensor.shape)}."
    )


def _validate_binary_shapes(logits: Tensor, target: Tensor, dem: Tensor) -> tuple[Tensor, Tensor]:
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError(f"logits must have shape [B, 1, H, W], got {tuple(logits.shape)}.")

    target_b1hw = _as_b1hw(target, "target").to(device=logits.device, dtype=logits.dtype)
    dem_b1hw = _as_b1hw(dem, "dem").to(device=logits.device, dtype=logits.dtype)
    expected = tuple(logits.shape)
    if tuple(target_b1hw.shape) != expected:
        raise ValueError(f"target must have shape {expected}, got {tuple(target_b1hw.shape)}.")
    if tuple(dem_b1hw.shape) != expected:
        raise ValueError(f"dem must have shape {expected}, got {tuple(dem_b1hw.shape)}.")
    return target_b1hw, dem_b1hw


def _normalized_abs_gradient(values: Tensor, dim: int, eps: float) -> Tensor:
    gradient = values.diff(dim=dim).abs()
    finite = torch.isfinite(gradient)
    safe_gradient = torch.where(finite, gradient, torch.zeros_like(gradient))
    denominator = safe_gradient.mean().detach().clamp_min(eps)
    return safe_gradient / denominator


class BinaryTopographicGradientLoss(nn.Module):
    """DEM-only topographic regularization for binary segmentation logits.

    This simple smoke-test loss compares spatial gradients of sigmoid
    probabilities with target-mask gradients, weighted by local DEM gradients.
    It does not use buildings and does not implement the future q_i mask.
    """

    def __init__(self, *, eps: float = 1e-6) -> None:
        super().__init__()
        if eps <= 0:
            raise ValueError("eps must be positive.")
        self.eps = eps

    def forward(
        self,
        logits: Tensor,
        target: Tensor,
        dem: Tensor,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        target_b1hw, dem_b1hw = _validate_binary_shapes(logits, target, dem)
        if valid_mask is not None:
            valid_b1hw = _as_b1hw(valid_mask, "valid_mask").to(
                device=logits.device,
                dtype=torch.bool,
            )
            if tuple(valid_b1hw.shape) != tuple(logits.shape):
                raise ValueError(
                    f"valid_mask must have shape {tuple(logits.shape)}, "
                    f"got {tuple(valid_b1hw.shape)}."
                )
        else:
            valid_b1hw = torch.ones_like(logits, dtype=torch.bool)
        probabilities = torch.sigmoid(logits)

        total = logits.new_zeros(())
        terms = 0

        for dim in (-1, -2):
            probability_gradient = probabilities.diff(dim=dim)
            target_gradient = target_b1hw.diff(dim=dim)
            dem_weight = _normalized_abs_gradient(dem_b1hw, dim=dim, eps=self.eps)
            dim_index = dim % valid_b1hw.ndim
            valid_a = valid_b1hw.narrow(dim_index, 0, valid_b1hw.shape[dim_index] - 1)
            valid_b = valid_b1hw.narrow(dim_index, 1, valid_b1hw.shape[dim_index] - 1)
            valid = torch.isfinite(probability_gradient) & torch.isfinite(target_gradient)
            valid = valid & torch.isfinite(dem_weight)
            valid = valid & valid_a & valid_b
            if valid.any():
                penalty = (probability_gradient - target_gradient).abs() * dem_weight
                total = total + penalty[valid].mean()
                terms += 1

        if terms == 0:
            return logits.sum() * 0.0

        loss = total / terms
        if not torch.isfinite(loss):
            return logits.sum() * 0.0
        return loss


class BinarySegmentationWithTopographicLoss(nn.Module):
    """BCEWithLogitsLoss plus DEM-only topographic regularization."""

    def __init__(self, *, alpha_topo: float = 0.1, eps: float = 1e-6) -> None:
        super().__init__()
        if alpha_topo < 0:
            raise ValueError("alpha_topo must be non-negative.")
        self.alpha_topo = alpha_topo
        self.classical_loss = nn.BCEWithLogitsLoss()
        self.topographic_loss = BinaryTopographicGradientLoss(eps=eps)

    def forward(self, logits: Tensor, target: Tensor, dem: Tensor) -> dict[str, Tensor]:
        target_b1hw, dem_b1hw = _validate_binary_shapes(logits, target, dem)
        classical = self.classical_loss(logits, target_b1hw)
        topo = self.topographic_loss(logits=logits, target=target_b1hw, dem=dem_b1hw)
        total = classical + self.alpha_topo * topo
        return {
            "loss": total,
            "loss_classical": classical,
            "loss_topographic": topo,
        }
