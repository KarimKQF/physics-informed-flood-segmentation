from pathlib import Path

import numpy as np
import pytest

from experiments.qualitative_demo.run_qualitative_demo import calculate_metrics, generate_figure


def test_calculate_metrics():
    counts = {
        "tp": 10.0,
        "fp": 5.0,
        "fn": 2.0,
        "tn": 83.0,
        "valid_pixel_count": 100.0,
        "positive_pixel_count": 12.0,
        "negative_pixel_count": 88.0,
        "predicted_positive_pixel_count": 15.0,
    }

    metrics = calculate_metrics(counts)
    assert metrics["iou"] == pytest.approx(10.0 / (10 + 5 + 2))
    assert metrics["dice"] == pytest.approx(20.0 / (20 + 5 + 2))
    assert metrics["target_positive_rate"] == pytest.approx(0.12)
    assert metrics["predicted_positive_rate"] == pytest.approx(0.15)


def test_generate_figure(tmp_path: Path):
    image = np.random.rand(64, 64).astype(np.float32)
    mask = np.random.randint(0, 2, size=(64, 64)).astype(np.int32)
    valid_mask = np.ones((64, 64), dtype=np.int32)
    valid_mask[0, 0] = 0
    prediction = np.random.randint(0, 2, size=(64, 64)).astype(np.int32)
    dem = np.random.rand(64, 64).astype(np.float32) * 100

    output_path = tmp_path / "test_fig.png"

    generate_figure("test", image, mask, valid_mask, prediction, dem, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
