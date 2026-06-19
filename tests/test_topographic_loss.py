import torch

from urban_runoff.losses import SegmentationWithTopographicLoss, TopographicLoss


def _sample_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = torch.randn(2, 2, 8, 8, requires_grad=True)
    dem = torch.randn(2, 8, 8)
    target = torch.randint(0, 2, (2, 8, 8))
    return logits, dem, target


def test_topographic_loss_returns_scalar() -> None:
    logits, dem, target = _sample_tensors()
    loss = TopographicLoss()(logits=logits, dem=dem, target=target)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_topographic_loss_accepts_bhw_and_b1hw_dem() -> None:
    logits, dem, target = _sample_tensors()
    criterion = TopographicLoss()
    loss_bhw = criterion(logits=logits, dem=dem, target=target)
    loss_b1hw = criterion(logits=logits, dem=dem[:, None], target=target)
    assert loss_bhw.ndim == 0
    assert loss_b1hw.ndim == 0


def test_topographic_loss_supports_ignore_index() -> None:
    logits, dem, target = _sample_tensors()
    target[:, 0, 0] = 255
    loss = TopographicLoss(ignore_index=255)(logits=logits, dem=dem, target=target)
    assert torch.isfinite(loss)


def test_topographic_loss_supports_no_reliability_and_reliability_mask() -> None:
    logits, dem, target = _sample_tensors()
    criterion = TopographicLoss()
    loss_none = criterion(logits=logits, dem=dem, target=target, pixel_reliability=None)
    reliability = torch.rand(2, 1, 8, 8)
    loss_reliable = criterion(
        logits=logits,
        dem=dem,
        target=target,
        pixel_reliability=reliability,
    )
    assert torch.isfinite(loss_none)
    assert torch.isfinite(loss_reliable)


def test_combined_loss_produces_gradient_on_logits() -> None:
    logits, dem, target = _sample_tensors()
    criterion = SegmentationWithTopographicLoss(class_weights=[1.0, 8.0])
    out = criterion(logits=logits, target=target, dem=dem)
    out["loss"].backward()
    assert out["loss"].ndim == 0
    assert out["loss_seg"].ndim == 0
    assert out["loss_topo"].ndim == 0
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0
