from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def _to_numpy(array: Any) -> NDArray[Any]:
    """Convert numpy-like or torch tensors to a CPU numpy array."""
    if hasattr(array, "detach") and hasattr(array, "cpu") and hasattr(array, "numpy"):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _validate_label_values(values: NDArray[Any], *, name: str, num_classes: int) -> None:
    if values.size == 0:
        return

    numeric = values.astype(np.float64, copy=False)
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} contains NaN or Inf labels.")
    if not np.all(numeric == np.floor(numeric)):
        raise ValueError(f"{name} must contain integer class labels.")

    min_value = int(numeric.min())
    max_value = int(numeric.max())
    if min_value < 0 or max_value >= num_classes:
        raise ValueError(
            f"{name} labels must be in [0, {num_classes - 1}] after ignore masking; "
            f"got range [{min_value}, {max_value}]."
        )


def compute_confusion_matrix(
    y_true: Any,
    y_pred: Any,
    num_classes: int = 2,
    ignore_index: int | None = -1,
) -> NDArray[np.int64]:
    """Return a confusion matrix with rows=true labels and columns=predicted labels.

    Pixels whose ground-truth value equals ``ignore_index`` are excluded before any
    metric is computed. For binary segmentation, the returned matrix is:

    ``[[TN, FP], [FN, TP]]``.
    """
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")

    true_array = _to_numpy(y_true)
    pred_array = _to_numpy(y_pred)

    if true_array.shape != pred_array.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape, "
            f"got {true_array.shape} and {pred_array.shape}."
        )

    true_flat = true_array.reshape(-1)
    pred_flat = pred_array.reshape(-1)
    if ignore_index is not None:
        valid_mask = true_flat != ignore_index
        true_flat = true_flat[valid_mask]
        pred_flat = pred_flat[valid_mask]

    if true_flat.size == 0:
        return np.zeros((num_classes, num_classes), dtype=np.int64)

    _validate_label_values(true_flat, name="y_true", num_classes=num_classes)
    _validate_label_values(pred_flat, name="y_pred", num_classes=num_classes)

    true_labels = true_flat.astype(np.int64, copy=False)
    pred_labels = pred_flat.astype(np.int64, copy=False)
    encoded = (num_classes * true_labels) + pred_labels
    counts = np.bincount(encoded, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes).astype(np.int64, copy=False)


def binary_confusion_counts(confusion_matrix: Any) -> dict[str, int]:
    """Return binary TP/FP/FN/TN counts from a 2x2 confusion matrix."""
    matrix = np.asarray(confusion_matrix, dtype=np.int64)
    if matrix.shape != (2, 2):
        raise ValueError(f"binary confusion counts require a 2x2 matrix, got {matrix.shape}.")

    true_negative = int(matrix[0, 0])
    false_positive = int(matrix[0, 1])
    false_negative = int(matrix[1, 0])
    true_positive = int(matrix[1, 1])
    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "tn": true_negative,
    }
