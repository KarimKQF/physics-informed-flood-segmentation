from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .confusion import compute_confusion_matrix

ZeroDivision = str | float | int | None


def _zero_value(zero_division: ZeroDivision) -> float:
    if zero_division is None:
        return float("nan")
    if isinstance(zero_division, str):
        normalized = zero_division.lower()
        if normalized in {"nan", "none"}:
            return float("nan")
        if normalized == "0":
            return 0.0
        if normalized == "1":
            return 1.0
        raise ValueError("zero_division must be one of 'nan', 0, or 1.")
    value = float(zero_division)
    if np.isnan(value):
        return float("nan")
    if value in {0.0, 1.0}:
        return value
    raise ValueError("zero_division must be one of 'nan', 0, or 1.")


def _safe_divide(numerator: float, denominator: float, *, zero_division: ZeroDivision) -> float:
    if denominator == 0:
        return _zero_value(zero_division)
    return float(numerator / denominator)


def _matrix_from_inputs(
    y_true: Any | None,
    y_pred: Any | None,
    *,
    confusion_matrix: Any | None,
    num_classes: int,
    ignore_index: int | None,
) -> NDArray[np.int64]:
    if confusion_matrix is not None:
        matrix = np.asarray(confusion_matrix, dtype=np.int64)
        if matrix.shape != (num_classes, num_classes):
            raise ValueError(
                f"confusion_matrix must have shape ({num_classes}, {num_classes}), "
                f"got {matrix.shape}."
            )
        return matrix
    if y_true is None or y_pred is None:
        raise ValueError("Provide y_true and y_pred, or provide confusion_matrix.")
    return compute_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )


def _class_counts(matrix: NDArray[np.int64], class_index: int) -> tuple[int, int, int, int]:
    if class_index < 0 or class_index >= matrix.shape[0]:
        raise ValueError(f"class_index must be in [0, {matrix.shape[0] - 1}].")
    true_positive = int(matrix[class_index, class_index])
    false_positive = int(matrix[:, class_index].sum() - true_positive)
    false_negative = int(matrix[class_index, :].sum() - true_positive)
    true_negative = int(matrix.sum() - true_positive - false_positive - false_negative)
    return true_positive, false_positive, false_negative, true_negative


def accuracy(
    y_true: Any | None = None,
    y_pred: Any | None = None,
    *,
    confusion_matrix: Any | None = None,
    num_classes: int = 2,
    ignore_index: int | None = -1,
    zero_division: ZeroDivision = "nan",
) -> float:
    matrix = _matrix_from_inputs(
        y_true,
        y_pred,
        confusion_matrix=confusion_matrix,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )
    return _safe_divide(float(np.trace(matrix)), float(matrix.sum()), zero_division=zero_division)


def precision(
    y_true: Any | None = None,
    y_pred: Any | None = None,
    *,
    class_index: int = 1,
    confusion_matrix: Any | None = None,
    num_classes: int = 2,
    ignore_index: int | None = -1,
    zero_division: ZeroDivision = "nan",
) -> float:
    matrix = _matrix_from_inputs(
        y_true,
        y_pred,
        confusion_matrix=confusion_matrix,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )
    true_positive, false_positive, _, _ = _class_counts(matrix, class_index)
    return _safe_divide(
        true_positive,
        true_positive + false_positive,
        zero_division=zero_division,
    )


def recall(
    y_true: Any | None = None,
    y_pred: Any | None = None,
    *,
    class_index: int = 1,
    confusion_matrix: Any | None = None,
    num_classes: int = 2,
    ignore_index: int | None = -1,
    zero_division: ZeroDivision = "nan",
) -> float:
    matrix = _matrix_from_inputs(
        y_true,
        y_pred,
        confusion_matrix=confusion_matrix,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )
    true_positive, _, false_negative, _ = _class_counts(matrix, class_index)
    return _safe_divide(
        true_positive,
        true_positive + false_negative,
        zero_division=zero_division,
    )


def f1_score(
    y_true: Any | None = None,
    y_pred: Any | None = None,
    *,
    class_index: int = 1,
    confusion_matrix: Any | None = None,
    num_classes: int = 2,
    ignore_index: int | None = -1,
    zero_division: ZeroDivision = "nan",
) -> float:
    matrix = _matrix_from_inputs(
        y_true,
        y_pred,
        confusion_matrix=confusion_matrix,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )
    true_positive, false_positive, false_negative, _ = _class_counts(matrix, class_index)
    denominator = (2 * true_positive) + false_positive + false_negative
    return _safe_divide(2 * true_positive, denominator, zero_division=zero_division)


def iou_per_class(
    y_true: Any | None = None,
    y_pred: Any | None = None,
    *,
    confusion_matrix: Any | None = None,
    num_classes: int = 2,
    ignore_index: int | None = -1,
    zero_division: ZeroDivision = "nan",
) -> NDArray[np.float64]:
    matrix = _matrix_from_inputs(
        y_true,
        y_pred,
        confusion_matrix=confusion_matrix,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )
    values = []
    for class_index in range(num_classes):
        true_positive, false_positive, false_negative, _ = _class_counts(matrix, class_index)
        values.append(
            _safe_divide(
                true_positive,
                true_positive + false_positive + false_negative,
                zero_division=zero_division,
            )
        )
    return np.asarray(values, dtype=np.float64)


def iou_water(
    y_true: Any | None = None,
    y_pred: Any | None = None,
    *,
    confusion_matrix: Any | None = None,
    ignore_index: int | None = -1,
    zero_division: ZeroDivision = "nan",
) -> float:
    return float(
        iou_per_class(
            y_true,
            y_pred,
            confusion_matrix=confusion_matrix,
            num_classes=2,
            ignore_index=ignore_index,
            zero_division=zero_division,
        )[1]
    )


def iou_background(
    y_true: Any | None = None,
    y_pred: Any | None = None,
    *,
    confusion_matrix: Any | None = None,
    ignore_index: int | None = -1,
    zero_division: ZeroDivision = "nan",
) -> float:
    return float(
        iou_per_class(
            y_true,
            y_pred,
            confusion_matrix=confusion_matrix,
            num_classes=2,
            ignore_index=ignore_index,
            zero_division=zero_division,
        )[0]
    )


def mean_iou(
    y_true: Any | None = None,
    y_pred: Any | None = None,
    *,
    confusion_matrix: Any | None = None,
    num_classes: int = 2,
    ignore_index: int | None = -1,
    zero_division: ZeroDivision = "nan",
) -> float:
    values = iou_per_class(
        y_true,
        y_pred,
        confusion_matrix=confusion_matrix,
        num_classes=num_classes,
        ignore_index=ignore_index,
        zero_division=zero_division,
    )
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return _zero_value(zero_division)
    return float(finite_values.mean())


def support_per_class(
    y_true: Any | None = None,
    y_pred: Any | None = None,
    *,
    confusion_matrix: Any | None = None,
    num_classes: int = 2,
    ignore_index: int | None = -1,
) -> NDArray[np.int64]:
    matrix = _matrix_from_inputs(
        y_true,
        y_pred,
        confusion_matrix=confusion_matrix,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )
    return matrix.sum(axis=1).astype(np.int64, copy=False)


def compute_all_metrics(
    y_true: Any | None = None,
    y_pred: Any | None = None,
    *,
    confusion_matrix: Any | None = None,
    num_classes: int = 2,
    ignore_index: int | None = -1,
    zero_division: ZeroDivision = "nan",
) -> dict[str, Any]:
    matrix = _matrix_from_inputs(
        y_true,
        y_pred,
        confusion_matrix=confusion_matrix,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )
    if num_classes != 2:
        raise ValueError("compute_all_metrics currently expects binary segmentation.")

    support = support_per_class(confusion_matrix=matrix, num_classes=num_classes)
    ious = iou_per_class(
        confusion_matrix=matrix,
        num_classes=num_classes,
        zero_division=zero_division,
    )
    return {
        "accuracy": accuracy(confusion_matrix=matrix, num_classes=num_classes, zero_division=zero_division),
        "precision": precision(confusion_matrix=matrix, num_classes=num_classes, zero_division=zero_division),
        "recall": recall(confusion_matrix=matrix, num_classes=num_classes, zero_division=zero_division),
        "f1_score": f1_score(confusion_matrix=matrix, num_classes=num_classes, zero_division=zero_division),
        "iou_per_class": ious.tolist(),
        "iou_background": float(ious[0]),
        "iou_water": float(ious[1]),
        "mean_iou": mean_iou(
            confusion_matrix=matrix,
            num_classes=num_classes,
            zero_division=zero_division,
        ),
        "support_per_class": support.tolist(),
        "support_background": int(support[0]),
        "support_water": int(support[1]),
        "valid_pixel_count": int(matrix.sum()),
        "confusion_matrix": matrix.astype(int).tolist(),
    }
