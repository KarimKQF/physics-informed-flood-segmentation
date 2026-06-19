from __future__ import annotations

import pytest
import torch

from urban_runoff.metrics import topographic_violation_rate, violation_rate_topo


def test_topographic_violation_rate_is_between_zero_and_one() -> None:
    pred_mask = torch.randint(0, 2, (2, 8, 8))
    dem = torch.randn(2, 1, 8, 8)
    ignore_mask = torch.zeros(2, 8, 8, dtype=torch.bool)
    reliability = torch.rand(2, 8, 8)

    rate = topographic_violation_rate(
        pred_mask=pred_mask,
        dem=dem,
        ignore_mask=ignore_mask,
        pixel_reliability=reliability,
    )

    assert rate.ndim == 0
    assert 0.0 <= rate.item() <= 1.0


def test_violation_rate_topo_detects_water_on_high_slope() -> None:
    dem = torch.tensor([[[[0.0, 0.0, 0.0, 10.0], [0.0, 0.0, 0.0, 10.0], [0.0, 0.0, 0.0, 10.0]]]])
    valid_mask = torch.ones_like(dem)
    high_slope_logits = torch.full_like(dem, -10.0)
    low_slope_logits = torch.full_like(dem, -10.0)
    high_slope_logits[..., 3] = 10.0
    low_slope_logits[..., 0] = 10.0

    high = violation_rate_topo(
        logits=high_slope_logits,
        dem=dem,
        valid_mask=valid_mask,
        slope_quantile=0.5,
    )
    low = violation_rate_topo(
        logits=low_slope_logits,
        dem=dem,
        valid_mask=valid_mask,
        slope_quantile=0.5,
    )

    assert high["violation_rate_topo"] > 0
    assert low["violation_rate_topo"] < high["violation_rate_topo"]


def test_violation_rate_topo_handles_no_predicted_water_and_constant_dem() -> None:
    dem = torch.zeros(1, 1, 3, 3)
    valid_mask = torch.ones_like(dem)
    logits = torch.full_like(dem, -10.0)

    result = violation_rate_topo(logits=logits, dem=dem, valid_mask=valid_mask)

    assert result["violation_rate_topo"] == pytest.approx(0.0)
    assert result["slope_threshold"] == pytest.approx(0.0)
