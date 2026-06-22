from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from losses.combined_loss import CombinedSegmentationPhysicsLoss
from losses.physics_topographic_loss import TopographicInconsistencyLoss


def _logits_from_water_scores(scores: torch.Tensor) -> torch.Tensor:
    background = torch.zeros_like(scores)
    return torch.stack([background, scores], dim=1)


def test_topographic_loss_basic_shape() -> None:
    logits = torch.randn(2, 2, 5, 6, requires_grad=True)
    target = torch.randint(0, 2, (2, 5, 6))
    topography = torch.randn(2, 5, 6)

    loss = TopographicInconsistencyLoss()(logits, target, topography)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_topographic_loss_backward_differentiability() -> None:
    logits = torch.randn(1, 2, 4, 4, requires_grad=True)
    target = torch.randint(0, 2, (1, 4, 4))
    topography = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)

    loss = TopographicInconsistencyLoss(neighborhood="8")(logits, target, topography)
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_topographic_loss_ignore_index_removes_pairs() -> None:
    scores = torch.tensor([[[4.0, -4.0]]], requires_grad=True)
    logits = _logits_from_water_scores(scores)
    target = torch.tensor([[[-1, 0]]])
    topography = torch.tensor([[[2.0, 1.0]]])

    loss = TopographicInconsistencyLoss()(logits, target, topography)

    assert loss.item() == pytest.approx(0.0)


def test_topographic_loss_returns_zero_when_no_valid_pairs() -> None:
    logits = torch.randn(1, 2, 3, 3, requires_grad=True)
    target = torch.zeros(1, 3, 3, dtype=torch.long)
    topography = torch.ones(1, 3, 3)

    loss = TopographicInconsistencyLoss()(logits, target, topography)
    loss.backward()

    assert loss.item() == pytest.approx(0.0)
    assert logits.grad is not None


def test_physically_inconsistent_case_gives_positive_penalty() -> None:
    scores = torch.tensor([[[4.0, -4.0]]], requires_grad=True)
    logits = _logits_from_water_scores(scores)
    target = torch.zeros(1, 1, 2, dtype=torch.long)
    topography = torch.tensor([[[2.0, 1.0]]])

    loss = TopographicInconsistencyLoss()(logits, target, topography)

    assert loss.item() > 0.9


def test_physically_coherent_case_gives_lower_penalty() -> None:
    target = torch.zeros(1, 1, 2, dtype=torch.long)
    topography = torch.tensor([[[2.0, 1.0]]])
    inconsistent = _logits_from_water_scores(torch.tensor([[[4.0, -4.0]]]))
    coherent = _logits_from_water_scores(torch.tensor([[[-4.0, 4.0]]]))
    criterion = TopographicInconsistencyLoss()

    inconsistent_loss = criterion(inconsistent, target, topography)
    coherent_loss = criterion(coherent, target, topography)

    assert coherent_loss < inconsistent_loss
    assert coherent_loss.item() < 0.01


def test_non_finite_topography_is_safe() -> None:
    logits = torch.randn(1, 2, 2, 3, requires_grad=True)
    target = torch.zeros(1, 2, 3, dtype=torch.long)
    topography = torch.tensor([[[float("nan"), 2.0, 1.0], [float("inf"), 1.0, 0.0]]])

    loss = TopographicInconsistencyLoss()(logits, target, topography)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_topographic_loss_cpu_cuda_consistency() -> None:
    logits = torch.randn(1, 2, 4, 4)
    target = torch.randint(0, 2, (1, 4, 4))
    topography = torch.randn(1, 4, 4)
    criterion = TopographicInconsistencyLoss(neighborhood="8")

    cpu_loss = criterion(logits, target, topography)
    cuda_loss = criterion(logits.cuda(), target.cuda(), topography.cuda()).cpu()

    assert torch.allclose(cpu_loss, cuda_loss, atol=1e-6)


def test_combined_loss_correctness() -> None:
    logits = torch.randn(2, 2, 4, 4, requires_grad=True)
    target = torch.randint(0, 2, (2, 4, 4))
    target[:, 0, 0] = -1
    topography = torch.randn(2, 4, 4)
    class_weights = torch.tensor([1.0, 2.0])
    criterion = CombinedSegmentationPhysicsLoss(
        lambda_topo=0.05,
        class_weights=class_weights,
        ignore_index=-1,
    )

    out = criterion(logits=logits, target=target, topography=topography)
    expected_seg = F.cross_entropy(logits, target, weight=class_weights, ignore_index=-1)
    expected_total = expected_seg + 0.05 * out["loss_topo"]
    out["loss_total"].backward()

    assert set(out) == {"loss_total", "loss_seg", "loss_topo", "lambda_topo"}
    assert torch.allclose(out["loss_seg"], expected_seg)
    assert torch.allclose(out["loss_total"], expected_total)
    assert out["lambda_topo"].item() == pytest.approx(0.05)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
