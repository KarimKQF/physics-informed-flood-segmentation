from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_trivial_module():
    script_path = PROJECT_ROOT / "experiments" / "loss_comparison" / "evaluate_trivial_baselines.py"
    spec = importlib.util.spec_from_file_location("evaluate_trivial_baselines", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def one_batch_loader():
    mask = torch.tensor([[[[1, 0, 1, 0]]]], dtype=torch.float32)
    valid_mask = torch.tensor([[[[1, 1, 0, 1]]]], dtype=torch.float32)
    dem = torch.zeros_like(mask)
    return [{"mask": mask, "valid_mask": valid_mask, "dem": dem}]


def test_all_water_baseline_matches_target_rate_and_recall() -> None:
    module = load_trivial_module()

    result = module.evaluate_baseline(
        one_batch_loader(),
        baseline="all_water",
        positive_probability=1.0,
        seed=42,
    )

    assert result["recall"] == pytest.approx(1.0)
    assert result["target_positive_rate"] == pytest.approx(1 / 3)
    assert result["precision"] == pytest.approx(1 / 3)
    assert result["predicted_positive_rate"] == pytest.approx(1.0)


def test_all_background_baseline_has_zero_predicted_positive_rate() -> None:
    module = load_trivial_module()

    result = module.evaluate_baseline(
        one_batch_loader(),
        baseline="all_background",
        positive_probability=0.0,
        seed=42,
    )

    assert result["predicted_positive_rate"] == pytest.approx(0.0)
    assert result["recall"] == pytest.approx(0.0)
    assert result["precision"] == pytest.approx(0.0)
