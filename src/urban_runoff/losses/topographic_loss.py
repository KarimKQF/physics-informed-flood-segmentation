from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _neighbor_offsets(use_eight_neighbors: bool) -> tuple[tuple[int, int], ...]:
    offsets = [(0, 1), (1, 0)]
    if use_eight_neighbors:
        offsets.extend([(1, 1), (1, -1)])
    return tuple(offsets)


def _as_bhw(tensor: Tensor, name: str) -> Tensor:
    if tensor.ndim == 3:
        return tensor
    if tensor.ndim == 4 and tensor.shape[1] == 1:
        return tensor[:, 0]
    raise ValueError(
        f"{name} must have shape [B, H, W] or [B, 1, H, W], got {tuple(tensor.shape)}."
    )


def _crop_pair(tensor: Tensor, dy: int, dx: int) -> tuple[Tensor, Tensor]:
    height = tensor.shape[-2]
    width = tensor.shape[-1]

    y_a_start = max(0, -dy)
    y_a_end = height - max(0, dy)
    x_a_start = max(0, -dx)
    x_a_end = width - max(0, dx)

    y_b_start = max(0, dy)
    y_b_end = height - max(0, -dy)
    x_b_start = max(0, dx)
    x_b_end = width - max(0, -dx)

    first = tensor[:, y_a_start:y_a_end, x_a_start:x_a_end]
    second = tensor[:, y_b_start:y_b_end, x_b_start:x_b_end]
    return first, second


def _validate_logits(logits: Tensor) -> None:
    if logits.ndim != 4 or logits.shape[1] != 2:
        raise ValueError(f"logits must have shape [B, 2, H, W], got {tuple(logits.shape)}.")


def _validate_spatial_match(logits: Tensor, *tensors: tuple[str, Tensor]) -> None:
    expected = (logits.shape[0], logits.shape[-2], logits.shape[-1])
    for name, tensor in tensors:
        if tuple(tensor.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(tensor.shape)}.")


class TopographicLoss(nn.Module):
    """Physics-informed penalty for topographically inconsistent water predictions.

    The penalty is high when a higher pixel is predicted as water while a lower
    neighboring pixel is predicted as dry. The implementation evaluates each
    undirected neighbor edge once and accounts for both possible high-to-low
    directions.
    """

    def __init__(
        self,
        *,
        ignore_index: int = 255,
        use_eight_neighbors: bool = True,
        height_scale: float = 1.0,
        max_weight: float | None = 10.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if height_scale <= 0:
            raise ValueError("height_scale must be strictly positive.")
        if eps <= 0:
            raise ValueError("eps must be strictly positive.")

        self.ignore_index = ignore_index
        self.use_eight_neighbors = use_eight_neighbors
        self.height_scale = height_scale
        self.max_weight = max_weight
        self.eps = eps

    def forward(
        self,
        logits: Tensor,
        dem: Tensor,
        target: Tensor | None = None,
        pixel_reliability: Tensor | None = None,
    ) -> Tensor:
        _validate_logits(logits)

        dem_bhw = _as_bhw(dem, "dem").to(device=logits.device, dtype=logits.dtype)
        target_bhw = target.to(device=logits.device) if target is not None else None
        reliability_bhw = (
            _as_bhw(pixel_reliability, "pixel_reliability").to(
                device=logits.device,
                dtype=logits.dtype,
            )
            if pixel_reliability is not None
            else None
        )

        tensors_to_check: list[tuple[str, Tensor]] = [("dem", dem_bhw)]
        if target_bhw is not None:
            tensors_to_check.append(("target", target_bhw))
        if reliability_bhw is not None:
            tensors_to_check.append(("pixel_reliability", reliability_bhw))
        _validate_spatial_match(logits, *tensors_to_check)

        probs = F.softmax(logits, dim=1)
        p_water = probs[:, 1]

        total = logits.new_zeros(())
        denominator = logits.new_zeros(())

        for dy, dx in _neighbor_offsets(self.use_eight_neighbors):
            p_a, p_b = _crop_pair(p_water, dy, dx)
            z_a, z_b = _crop_pair(dem_bhw, dy, dx)
            valid_pair = torch.isfinite(z_a) & torch.isfinite(z_b)

            if target_bhw is not None:
                t_a, t_b = _crop_pair(target_bhw, dy, dx)
                valid_pair = valid_pair & (t_a != self.ignore_index) & (t_b != self.ignore_index)

            if reliability_bhw is not None:
                q_a, q_b = _crop_pair(reliability_bhw, dy, dx)
                valid_pair = valid_pair & torch.isfinite(q_a) & torch.isfinite(q_b)
                pair_reliability = torch.clamp(q_a, min=0) * torch.clamp(q_b, min=0)
            else:
                pair_reliability = torch.ones_like(z_a)

            forward_loss, forward_denominator = self._directional_penalty(
                z_high=z_a,
                z_low=z_b,
                p_high=p_a,
                p_low=p_b,
                valid_pair=valid_pair,
                pair_reliability=pair_reliability,
            )
            backward_loss, backward_denominator = self._directional_penalty(
                z_high=z_b,
                z_low=z_a,
                p_high=p_b,
                p_low=p_a,
                valid_pair=valid_pair,
                pair_reliability=pair_reliability,
            )
            total = total + forward_loss + backward_loss
            denominator = denominator + forward_denominator + backward_denominator

        return total / denominator.clamp_min(self.eps)

    def _directional_penalty(
        self,
        *,
        z_high: Tensor,
        z_low: Tensor,
        p_high: Tensor,
        p_low: Tensor,
        valid_pair: Tensor,
        pair_reliability: Tensor,
    ) -> tuple[Tensor, Tensor]:
        height_weight = torch.relu((z_high - z_low) / self.height_scale)
        if self.max_weight is not None:
            height_weight = torch.clamp(height_weight, max=self.max_weight)

        high_low = height_weight > 0
        valid_weight = (valid_pair & high_low).to(dtype=p_high.dtype) * pair_reliability
        contribution = height_weight * p_high * (1.0 - p_low) * valid_weight
        return contribution.sum(), valid_weight.sum()


class SegmentationWithTopographicLoss(nn.Module):
    """Cross-entropy segmentation loss combined with a topographic penalty."""

    def __init__(
        self,
        *,
        lambda_topo: float = 0.1,
        class_weights: Iterable[float] | Tensor | None = None,
        ignore_index: int = 255,
        use_eight_neighbors: bool = True,
        height_scale: float = 1.0,
        max_weight: float | None = 10.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if lambda_topo < 0:
            raise ValueError("lambda_topo must be non-negative.")

        weight = None
        if class_weights is not None:
            weight = torch.as_tensor(list(class_weights), dtype=torch.float32)
            if weight.numel() != 2:
                raise ValueError("class_weights must contain exactly two values.")

        self.lambda_topo = lambda_topo
        self.segmentation_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        self.topographic_loss = TopographicLoss(
            ignore_index=ignore_index,
            use_eight_neighbors=use_eight_neighbors,
            height_scale=height_scale,
            max_weight=max_weight,
            eps=eps,
        )

    def forward(
        self,
        *,
        logits: Tensor,
        target: Tensor,
        dem: Tensor,
        pixel_reliability: Tensor | None = None,
    ) -> dict[str, Tensor]:
        _validate_logits(logits)
        target = target.to(device=logits.device, dtype=torch.long)
        loss_seg = self.segmentation_loss(logits, target)
        loss_topo = self.topographic_loss(
            logits=logits,
            dem=dem,
            target=target,
            pixel_reliability=pixel_reliability,
        )
        loss = loss_seg + self.lambda_topo * loss_topo
        return {
            "loss": loss,
            "loss_seg": loss_seg.detach(),
            "loss_topo": loss_topo.detach(),
        }


def build_loss_from_config(config: dict[str, Any]) -> SegmentationWithTopographicLoss:
    loss_config = config.get("loss", config)
    return SegmentationWithTopographicLoss(
        class_weights=loss_config.get("class_weights"),
        ignore_index=int(loss_config.get("ignore_index", 255)),
        lambda_topo=float(loss_config.get("lambda_topo", 0.1)),
        use_eight_neighbors=bool(loss_config.get("use_eight_neighbors", True)),
        height_scale=float(loss_config.get("height_scale", 1.0)),
        max_weight=loss_config.get("max_weight", 10.0),
    )
