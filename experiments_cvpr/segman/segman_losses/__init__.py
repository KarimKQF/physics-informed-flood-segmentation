"""Config-driven loss selector for SegMAN flood-segmentation experiments."""

from .segman_loss import MODES, SegManCombinedLoss, build_loss, lambda_for_epoch

__all__ = ["MODES", "SegManCombinedLoss", "build_loss", "lambda_for_epoch"]
