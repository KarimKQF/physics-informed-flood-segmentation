from __future__ import annotations

import torch
from torch import Tensor

from urban_runoff.losses.topographic_loss import _as_bhw, _crop_pair, _neighbor_offsets


def _as_b1hw(tensor: Tensor, name: str) -> Tensor:
    if tensor.ndim == 3:
        return tensor[:, None]
    if tensor.ndim == 4 and tensor.shape[1] == 1:
        return tensor
    raise ValueError(
        f"{name} must have shape [B, H, W] or [B, 1, H, W], got {tuple(tensor.shape)}."
    )


def violation_rate_topo(
    logits: Tensor,
    dem: Tensor,
    valid_mask: Tensor,
    *,
    threshold: float = 0.5,
    slope_quantile: float = 0.75,
    eps: float = 1e-6,
) -> dict[str, float]:
    """Rate of predicted water pixels located on high-slope valid pixels.

    This is a simple evaluation metric, not a hydrological truth. High-slope
    pixels are defined by a per-batch DEM slope quantile computed on valid
    pixels only.
    """
    if not 0 <= slope_quantile <= 1:
        raise ValueError("slope_quantile must be in [0, 1].")
    if eps <= 0:
        raise ValueError("eps must be positive.")

    logits_b1hw = _as_b1hw(logits, "logits")
    dem_b1hw = _as_b1hw(dem, "dem").to(device=logits.device, dtype=logits.dtype)
    valid_b1hw = _as_b1hw(valid_mask, "valid_mask").to(device=logits.device, dtype=torch.bool)
    if tuple(logits_b1hw.shape) != tuple(dem_b1hw.shape):
        raise ValueError("dem must match logits shape.")
    if tuple(logits_b1hw.shape) != tuple(valid_b1hw.shape):
        raise ValueError("valid_mask must match logits shape.")

    with torch.no_grad():
        safe_dem = torch.nan_to_num(dem_b1hw, nan=0.0, posinf=0.0, neginf=0.0)
        dx = torch.zeros_like(safe_dem)
        dy = torch.zeros_like(safe_dem)
        dx[..., :, 1:] = safe_dem[..., :, 1:] - safe_dem[..., :, :-1]
        dy[..., 1:, :] = safe_dem[..., 1:, :] - safe_dem[..., :-1, :]
        slope = torch.sqrt(dx.square() + dy.square())
        finite_valid = valid_b1hw & torch.isfinite(slope)
        valid_slopes = slope[finite_valid]
        if valid_slopes.numel() == 0:
            return {"violation_rate_topo": 0.0, "slope_threshold": 0.0}

        slope_threshold = torch.quantile(valid_slopes.float(), slope_quantile)
        pred_water = torch.sigmoid(logits_b1hw) >= threshold
        predicted_valid_water = pred_water & finite_valid
        denominator = predicted_valid_water.sum().float()
        if denominator <= 0:
            return {
                "violation_rate_topo": 0.0,
                "slope_threshold": float(slope_threshold.detach().cpu()),
            }

        violations = predicted_valid_water & (slope > slope_threshold)
        rate = violations.sum().float() / denominator.clamp_min(eps)
        if not torch.isfinite(rate):
            rate = rate.new_zeros(())
        return {
            "violation_rate_topo": float(rate.detach().cpu()),
            "slope_threshold": float(slope_threshold.detach().cpu()),
        }


def topographic_violation_rate(
    pred_mask: Tensor,
    dem: Tensor,
    ignore_mask: Tensor | None = None,
    pixel_reliability: Tensor | None = None,
    *,
    use_eight_neighbors: bool = True,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> Tensor:
    """Compute the rate of high-water / low-dry violations over valid high-low pairs."""
    if pred_mask.ndim != 3:
        raise ValueError(f"pred_mask must have shape [B, H, W], got {tuple(pred_mask.shape)}.")

    dem_bhw = _as_bhw(dem, "dem").to(device=pred_mask.device)
    if tuple(dem_bhw.shape) != tuple(pred_mask.shape):
        raise ValueError(
            f"dem must match pred_mask shape {tuple(pred_mask.shape)}, got {tuple(dem_bhw.shape)}."
        )

    if pred_mask.is_floating_point():
        water = pred_mask >= threshold
    else:
        water = pred_mask == 1

    ignore = torch.zeros_like(water, dtype=torch.bool)
    if ignore_mask is not None:
        if ignore_mask.ndim != 3 or tuple(ignore_mask.shape) != tuple(pred_mask.shape):
            raise ValueError("ignore_mask must have shape [B, H, W] matching pred_mask.")
        ignore = ignore_mask.to(device=pred_mask.device, dtype=torch.bool)

    reliability_bhw = (
        _as_bhw(pixel_reliability, "pixel_reliability").to(device=pred_mask.device)
        if pixel_reliability is not None
        else None
    )
    if reliability_bhw is not None and tuple(reliability_bhw.shape) != tuple(pred_mask.shape):
        raise ValueError("pixel_reliability must match pred_mask spatial shape.")

    numerator = dem_bhw.new_zeros(())
    denominator = dem_bhw.new_zeros(())

    for dy, dx in _neighbor_offsets(use_eight_neighbors):
        w_a, w_b = _crop_pair(water, dy, dx)
        z_a, z_b = _crop_pair(dem_bhw, dy, dx)
        i_a, i_b = _crop_pair(ignore, dy, dx)
        valid_pair = torch.isfinite(z_a) & torch.isfinite(z_b) & ~i_a & ~i_b

        if reliability_bhw is not None:
            q_a, q_b = _crop_pair(reliability_bhw, dy, dx)
            valid_pair = valid_pair & torch.isfinite(q_a) & torch.isfinite(q_b)
            pair_weight = torch.clamp(q_a, min=0) * torch.clamp(q_b, min=0)
        else:
            pair_weight = torch.ones_like(z_a)

        forward_num, forward_den = _directional_violations(
            z_high=z_a,
            z_low=z_b,
            water_high=w_a,
            water_low=w_b,
            valid_pair=valid_pair,
            pair_weight=pair_weight,
        )
        backward_num, backward_den = _directional_violations(
            z_high=z_b,
            z_low=z_a,
            water_high=w_b,
            water_low=w_a,
            valid_pair=valid_pair,
            pair_weight=pair_weight,
        )
        numerator = numerator + forward_num + backward_num
        denominator = denominator + forward_den + backward_den

    return numerator / denominator.clamp_min(eps)


def _directional_violations(
    *,
    z_high: Tensor,
    z_low: Tensor,
    water_high: Tensor,
    water_low: Tensor,
    valid_pair: Tensor,
    pair_weight: Tensor,
) -> tuple[Tensor, Tensor]:
    high_low = z_high > z_low
    valid_weight = (valid_pair & high_low).to(dtype=z_high.dtype) * pair_weight
    violation = (water_high & ~water_low).to(dtype=z_high.dtype) * valid_weight
    return violation.sum(), valid_weight.sum()
