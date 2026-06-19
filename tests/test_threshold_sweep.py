from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_threshold_module():
    script_path = PROJECT_ROOT / "experiments" / "loss_comparison" / "threshold_sweep.py"
    spec = importlib.util.spec_from_file_location("threshold_sweep", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FixedLogitModel(nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.tensor([[[[2.0, 0.0, -2.0, -2.0]]]], dtype=image.dtype, device=image.device)


def one_batch_loader():
    image = torch.zeros(1, 1, 1, 4)
    mask = torch.tensor([[[[1, 0, 1, 0]]]], dtype=torch.float32)
    valid_mask = torch.ones_like(mask)
    dem = torch.zeros_like(mask)
    return [{"image": image, "mask": mask, "valid_mask": valid_mask, "dem": dem}]


def test_threshold_sweep_changes_predicted_positive_rate() -> None:
    module = load_threshold_module()
    model = FixedLogitModel()

    low = module.evaluate_checkpoint(
        model=model,
        loader=one_batch_loader(),
        threshold=0.5,
        device=torch.device("cpu"),
    )
    high = module.evaluate_checkpoint(
        model=model,
        loader=one_batch_loader(),
        threshold=0.9,
        device=torch.device("cpu"),
    )

    assert low["predicted_positive_rate"] == pytest.approx(0.5)
    assert high["predicted_positive_rate"] == pytest.approx(0.0)
    assert low["recall"] > high["recall"]
