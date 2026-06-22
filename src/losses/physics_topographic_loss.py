from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


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


def _neighbor_offsets(neighborhood: str) -> tuple[tuple[int, int], ...]:
    if neighborhood == "4":
        return ((0, 1), (1, 0))
    if neighborhood == "8":
        return ((0, 1), (1, 0), (1, 1), (1, -1))
    raise ValueError('neighborhood must be "4" or "8".')


class TopographicInconsistencyLoss(nn.Module):
    """Differentiable penalty for topographically implausible flood predictions.

    The loss compares local neighboring pixels. If pixel i is higher than a
    neighboring pixel j by more than ``elevation_margin``, then predicting i as
    water while predicting j as dry is penalized:

        w_ij * p_i(water) * (1 - p_j(water)).

    The loss only depends on softmax probabilities and is therefore fully
    differentiable with respect to ``logits``. The topography tensor is treated
    as fixed conditioning data; missing, NaN, and infinite topographic values
    are ignored. Pixels whose label equals ``ignore_index`` are also removed
    from neighbor pairs.

    This is a local monotonicity prior, not a full hydraulic model. It does not
    know flow direction, barriers, drainage connectivity, roughness, rainfall,
    or human infrastructure. STEP 6B must validate that topographic rasters are
    aligned with Sen1Floods11 before this loss is used in real training.
    """

    def __init__(
        self,
        *,
        ignore_index: int = -1,
        water_class: int = 1,
        elevation_margin: float = 0.0,
        elevation_scale: float = 1.0,
        use_elevation_weight: bool = True,
        neighborhood: str = "4",
        eps: float = 1e-6,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if water_class < 0:
            raise ValueError("water_class must be non-negative.")
        if elevation_scale <= 0:
            raise ValueError("elevation_scale must be strictly positive.")
        if eps <= 0:
            raise ValueError("eps must be strictly positive.")
        if reduction not in {"mean", "sum"}:
            raise ValueError('reduction must be "mean" or "sum".')

        self.ignore_index = ignore_index
        self.water_class = water_class
        self.elevation_margin = float(elevation_margin)
        self.elevation_scale = float(elevation_scale)
        self.use_elevation_weight = use_elevation_weight
        self.neighborhood = neighborhood
        self.eps = eps
        self.reduction = reduction
        self._offsets = _neighbor_offsets(neighborhood)

    def forward(self, logits: Tensor, target: Tensor, topography: Tensor) -> Tensor:
        if logits.ndim != 4:
            raise ValueError(f"logits must have shape [B, C, H, W], got {tuple(logits.shape)}.")
        if self.water_class >= logits.shape[1]:
            raise ValueError(
                f"water_class={self.water_class} is outside logits channel dimension "
                f"{logits.shape[1]}."
            )

        target_bhw = _as_bhw(target, "target").to(device=logits.device, dtype=torch.long)
        topo_bhw = _as_bhw(topography, "topography").to(device=logits.device, dtype=logits.dtype)
        expected = (logits.shape[0], logits.shape[-2], logits.shape[-1])
        if tuple(target_bhw.shape) != expected:
            raise ValueError(f"target must have shape {expected}, got {tuple(target_bhw.shape)}.")
        if tuple(topo_bhw.shape) != expected:
            raise ValueError(f"topography must have shape {expected}, got {tuple(topo_bhw.shape)}.")

        p_water = F.softmax(logits, dim=1)[:, self.water_class]
        valid_pixel = (target_bhw != self.ignore_index) & torch.isfinite(topo_bhw)

        total = logits.new_zeros(())
        pair_count = logits.new_zeros(())

        for dy, dx in self._offsets:
            p_a, p_b = _crop_pair(p_water, dy, dx)
            h_a, h_b = _crop_pair(topo_bhw, dy, dx)
            valid_a, valid_b = _crop_pair(valid_pixel, dy, dx)
            valid_pair = valid_a & valid_b

            loss_ab, count_ab = self._directional_penalty(
                p_high=p_a,
                p_low=p_b,
                h_high=h_a,
                h_low=h_b,
                valid_pair=valid_pair,
            )
            loss_ba, count_ba = self._directional_penalty(
                p_high=p_b,
                p_low=p_a,
                h_high=h_b,
                h_low=h_a,
                valid_pair=valid_pair,
            )
            total = total + loss_ab + loss_ba
            pair_count = pair_count + count_ab + count_ba

        if self.reduction == "sum":
            return total
        if pair_count <= 0:
            return logits.sum() * 0.0
        return total / pair_count.clamp_min(self.eps)

    def _directional_penalty(
        self,
        *,
        p_high: Tensor,
        p_low: Tensor,
        h_high: Tensor,
        h_low: Tensor,
        valid_pair: Tensor,
    ) -> tuple[Tensor, Tensor]:
        safe_delta = torch.where(
            valid_pair,
            h_high - h_low - self.elevation_margin,
            torch.zeros_like(h_high),
        )
        high_to_low = valid_pair & (safe_delta > 0)
        if self.use_elevation_weight:
            weights = torch.relu(safe_delta / self.elevation_scale)
        else:
            weights = torch.ones_like(safe_delta)

        valid_weight = high_to_low.to(dtype=p_high.dtype)
        contribution = weights * p_high * (1.0 - p_low) * valid_weight
        return contribution.sum(), valid_weight.sum()
