from __future__ import annotations

import torch
from torch import Tensor


def _validate_inputs(
    logits: Tensor,
    targets: Tensor,
    valid_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    if logits.shape != targets.shape:
        raise ValueError(
            "logits and targets must have the same shape, "
            f"got {tuple(logits.shape)} and {tuple(targets.shape)}."
        )
    if logits.shape != valid_mask.shape:
        raise ValueError(
            "logits and valid_mask must have the same shape, "
            f"got {tuple(logits.shape)} and {tuple(valid_mask.shape)}."
        )
    targets = targets.to(device=logits.device, dtype=torch.bool)
    valid = valid_mask.to(device=logits.device, dtype=torch.bool)
    return logits, targets, valid


def masked_binary_confusion_counts(
    logits: Tensor,
    targets: Tensor,
    valid_mask: Tensor,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    logits, targets, valid = _validate_inputs(logits, targets, valid_mask)
    probabilities = torch.sigmoid(logits)
    predictions = probabilities >= threshold

    true_positive = (predictions & targets & valid).sum().float()
    false_positive = (predictions & ~targets & valid).sum().float()
    false_negative = (~predictions & targets & valid).sum().float()
    true_negative = (~predictions & ~targets & valid).sum().float()
    valid_pixel_count = valid.sum().float()
    positive_pixel_count = (targets & valid).sum().float()
    negative_pixel_count = (~targets & valid).sum().float()
    predicted_positive_pixel_count = (predictions & valid).sum().float()
    return {
        "tp": float(true_positive.detach().cpu()),
        "fp": float(false_positive.detach().cpu()),
        "fn": float(false_negative.detach().cpu()),
        "tn": float(true_negative.detach().cpu()),
        "valid_pixel_count": float(valid_pixel_count.detach().cpu()),
        "positive_pixel_count": float(positive_pixel_count.detach().cpu()),
        "negative_pixel_count": float(negative_pixel_count.detach().cpu()),
        "predicted_positive_pixel_count": float(predicted_positive_pixel_count.detach().cpu()),
    }


def masked_segmentation_metrics(
    logits: Tensor,
    targets: Tensor,
    valid_mask: Tensor,
    *,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> dict[str, float]:
    if eps <= 0:
        raise ValueError("eps must be positive.")
    counts = masked_binary_confusion_counts(
        logits=logits,
        targets=targets,
        valid_mask=valid_mask,
        threshold=threshold,
    )
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]
    valid_pixel_count = counts["valid_pixel_count"]
    positive_pixel_count = counts["positive_pixel_count"]
    negative_pixel_count = counts["negative_pixel_count"]
    predicted_positive_pixel_count = counts["predicted_positive_pixel_count"]

    iou = tp / (tp + fp + fn + eps)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    recall = tp / (tp + fn + eps)
    precision = tp / (tp + fp + eps)
    target_positive_rate = positive_pixel_count / (valid_pixel_count + eps)
    predicted_positive_rate = predicted_positive_pixel_count / (valid_pixel_count + eps)
    return {
        "iou": float(iou),
        "dice": float(dice),
        "f1": float(dice),
        "recall": float(recall),
        "precision": float(precision),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "target_positive_rate": float(target_positive_rate),
        "predicted_positive_rate": float(predicted_positive_rate),
        "valid_pixel_count": float(valid_pixel_count),
        "positive_pixel_count": float(positive_pixel_count),
        "negative_pixel_count": float(negative_pixel_count),
    }
