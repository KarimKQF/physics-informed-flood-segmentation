from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from metrics.confusion import compute_confusion_matrix
from metrics.segmentation_metrics import (
    accuracy as step4_accuracy,
    compute_all_metrics,
    f1_score,
    iou_background,
    iou_per_class,
    iou_water,
    precision,
    recall,
    support_per_class,
)
from urban_runoff.metrics import masked_binary_confusion_counts, masked_segmentation_metrics


def logits_from_binary(predictions: torch.Tensor) -> torch.Tensor:
    return torch.where(
        predictions.bool(),
        torch.full_like(predictions, 10.0),
        torch.full_like(predictions, -10.0),
    )


def test_masked_binary_metrics_match_known_counts() -> None:
    predictions = torch.tensor([[[[1, 1, 0, 0]]]], dtype=torch.float32)
    targets = torch.tensor([[[[1, 0, 1, 0]]]], dtype=torch.float32)
    valid_mask = torch.ones_like(targets)
    logits = logits_from_binary(predictions)

    counts = masked_binary_confusion_counts(logits, targets, valid_mask)
    metrics = masked_segmentation_metrics(logits, targets, valid_mask)

    assert counts["tp"] == 1.0
    assert counts["fp"] == 1.0
    assert counts["fn"] == 1.0
    assert counts["tn"] == 1.0
    assert counts["valid_pixel_count"] == 4.0
    assert counts["positive_pixel_count"] == 2.0
    assert counts["negative_pixel_count"] == 2.0
    assert counts["predicted_positive_pixel_count"] == 2.0
    assert metrics["iou"] == pytest.approx(1 / 3)
    assert metrics["dice"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["target_positive_rate"] == pytest.approx(0.5)
    assert metrics["predicted_positive_rate"] == pytest.approx(0.5)


def test_masked_binary_metrics_ignore_invalid_pixels() -> None:
    predictions = torch.tensor([[[[1, 1, 0, 0]]]], dtype=torch.float32)
    targets = torch.tensor([[[[1, 0, 1, 0]]]], dtype=torch.float32)
    valid_mask = torch.tensor([[[[1, 0, 1, 1]]]], dtype=torch.float32)
    logits = logits_from_binary(predictions)

    counts = masked_binary_confusion_counts(logits, targets, valid_mask)
    metrics = masked_segmentation_metrics(logits, targets, valid_mask)

    assert counts["tp"] == 1.0
    assert counts["fp"] == 0.0
    assert counts["fn"] == 1.0
    assert counts["tn"] == 1.0
    assert counts["valid_pixel_count"] == 3.0
    assert counts["positive_pixel_count"] == 2.0
    assert counts["negative_pixel_count"] == 1.0
    assert counts["predicted_positive_pixel_count"] == 1.0
    assert metrics["iou"] == pytest.approx(0.5)
    assert metrics["dice"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["target_positive_rate"] == pytest.approx(2 / 3)
    assert metrics["predicted_positive_rate"] == pytest.approx(1 / 3)


def test_step4_perfect_prediction() -> None:
    y_true = np.array([[0, 1], [0, 1]])
    y_pred = np.array([[0, 1], [0, 1]])

    matrix = compute_confusion_matrix(y_true, y_pred)
    metrics = compute_all_metrics(confusion_matrix=matrix)

    assert matrix.tolist() == [[2, 0], [0, 2]]
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1_score"] == pytest.approx(1.0)
    assert metrics["iou_background"] == pytest.approx(1.0)
    assert metrics["iou_water"] == pytest.approx(1.0)
    assert metrics["mean_iou"] == pytest.approx(1.0)


def test_step4_completely_wrong_prediction() -> None:
    y_true = np.array([[0, 0], [1, 1]])
    y_pred = np.array([[1, 1], [0, 0]])

    matrix = compute_confusion_matrix(y_true, y_pred)
    metrics = compute_all_metrics(confusion_matrix=matrix)

    assert matrix.tolist() == [[0, 2], [2, 0]]
    assert metrics["accuracy"] == pytest.approx(0.0)
    assert metrics["precision"] == pytest.approx(0.0)
    assert metrics["recall"] == pytest.approx(0.0)
    assert metrics["f1_score"] == pytest.approx(0.0)
    assert metrics["iou_background"] == pytest.approx(0.0)
    assert metrics["iou_water"] == pytest.approx(0.0)


def test_step4_mixed_prediction() -> None:
    y_true = np.array([[0, 1, 1]])
    y_pred = np.array([[0, 1, 0]])

    matrix = compute_confusion_matrix(y_true, y_pred)

    assert matrix.tolist() == [[1, 0], [1, 1]]
    assert step4_accuracy(confusion_matrix=matrix) == pytest.approx(2 / 3)
    assert precision(confusion_matrix=matrix) == pytest.approx(1.0)
    assert recall(confusion_matrix=matrix) == pytest.approx(0.5)
    assert f1_score(confusion_matrix=matrix) == pytest.approx(2 / 3)
    assert iou_background(confusion_matrix=matrix) == pytest.approx(0.5)
    assert iou_water(confusion_matrix=matrix) == pytest.approx(0.5)


def test_step4_ignore_index_pixels() -> None:
    y_true = np.array([[-1, 0, 1, 1]])
    y_pred = np.array([[1, 0, 1, 0]])

    matrix = compute_confusion_matrix(y_true, y_pred, ignore_index=-1)

    assert matrix.tolist() == [[1, 0], [1, 1]]
    assert compute_all_metrics(confusion_matrix=matrix)["valid_pixel_count"] == 3


def test_step4_all_pixels_ignored_returns_nan_metrics() -> None:
    y_true = np.array([[-1, -1]])
    y_pred = np.array([[0, 1]])

    matrix = compute_confusion_matrix(y_true, y_pred, ignore_index=-1)
    metrics = compute_all_metrics(confusion_matrix=matrix)

    assert matrix.tolist() == [[0, 0], [0, 0]]
    assert metrics["valid_pixel_count"] == 0
    assert math.isnan(metrics["accuracy"])
    assert math.isnan(metrics["iou_water"])
    assert math.isnan(metrics["mean_iou"])


def test_step4_no_water_ground_truth_keeps_false_positive_penalty() -> None:
    y_true = np.array([[0, 0, 0, 0]])
    y_pred = np.array([[0, 1, 0, 0]])

    matrix = compute_confusion_matrix(y_true, y_pred)
    metrics = compute_all_metrics(confusion_matrix=matrix)

    assert matrix.tolist() == [[3, 1], [0, 0]]
    assert metrics["support_water"] == 0
    assert metrics["precision"] == pytest.approx(0.0)
    assert math.isnan(metrics["recall"])
    assert metrics["iou_water"] == pytest.approx(0.0)
    assert metrics["iou_background"] == pytest.approx(0.75)


def test_step4_no_water_prediction() -> None:
    y_true = np.array([[0, 1, 1, 0]])
    y_pred = np.array([[0, 0, 0, 0]])

    matrix = compute_confusion_matrix(y_true, y_pred)
    metrics = compute_all_metrics(confusion_matrix=matrix)

    assert matrix.tolist() == [[2, 0], [2, 0]]
    assert math.isnan(metrics["precision"])
    assert metrics["recall"] == pytest.approx(0.0)
    assert metrics["f1_score"] == pytest.approx(0.0)
    assert metrics["iou_water"] == pytest.approx(0.0)


def test_step4_all_water_ground_truth() -> None:
    y_true = np.array([[1, 1, 1]])
    y_pred = np.array([[1, 0, 1]])

    matrix = compute_confusion_matrix(y_true, y_pred)
    metrics = compute_all_metrics(confusion_matrix=matrix)

    assert matrix.tolist() == [[0, 0], [1, 2]]
    assert metrics["support_background"] == 0
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["iou_background"] == pytest.approx(0.0)
    assert metrics["iou_water"] == pytest.approx(2 / 3)


def test_step4_absent_class_behavior_is_explicit_nan() -> None:
    y_true = np.array([[0, 0]])
    y_pred = np.array([[0, 0]])

    metrics = compute_all_metrics(y_true, y_pred)

    assert metrics["support_per_class"] == [2, 0]
    assert metrics["iou_background"] == pytest.approx(1.0)
    assert math.isnan(metrics["iou_water"])
    assert math.isnan(metrics["precision"])
    assert math.isnan(metrics["recall"])
    assert math.isnan(metrics["f1_score"])


def test_step4_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same shape"):
        compute_confusion_matrix(np.zeros((2, 2)), np.zeros((2, 3)))


def test_step4_numpy_input() -> None:
    y_true = np.array([[0, 1, -1]])
    y_pred = np.array([[0, 0, 1]])

    assert compute_confusion_matrix(y_true, y_pred).tolist() == [[1, 0], [1, 0]]
    assert support_per_class(y_true, y_pred).tolist() == [1, 1]
    assert iou_per_class(y_true, y_pred).tolist() == [0.5, 0.0]


def test_step4_torch_input_if_torch_is_installed() -> None:
    y_true = torch.tensor([[0, 1, -1]])
    y_pred = torch.tensor([[0, 1, 1]])

    matrix = compute_confusion_matrix(y_true, y_pred)

    assert matrix.tolist() == [[1, 0], [0, 1]]
    assert compute_all_metrics(confusion_matrix=matrix)["accuracy"] == pytest.approx(1.0)


def test_step4_manual_synthetic_sanity_check() -> None:
    y_true = np.array([[-1, 0, 0, 1, 1, 1]])
    y_pred = np.array([[1, 0, 1, 1, 0, 1]])

    matrix = compute_confusion_matrix(y_true, y_pred)
    metrics = compute_all_metrics(confusion_matrix=matrix)

    assert matrix.tolist() == [[1, 1], [1, 2]]
    assert metrics["accuracy"] == pytest.approx(3 / 5)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["f1_score"] == pytest.approx(2 / 3)
    assert metrics["iou_water"] == pytest.approx(0.5)
