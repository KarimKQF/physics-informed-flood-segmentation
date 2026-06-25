from __future__ import annotations

from collections.abc import Iterable

import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from losses.physics_topographic_loss import TopographicInconsistencyLoss


def _soft_dice_loss(
    logits: Tensor,
    target: Tensor,
    ignore_index: int = -1,
    smooth: float = 1.0,
) -> Tensor:
    """Legacy hand-rolled multiclass soft Dice loss (kept for reference only).

    WARNING: This is NOT bit-exact with terratorch's STEP 5S-A Dice loss.
    terratorch's ``loss: dice`` builds ``smp.losses.DiceLoss("multiclass",
    ignore_index=...)`` which uses ``smooth=0.0``, ``eps=1e-7`` AND zeroes the
    contribution of classes that are entirely absent from a batch. This function
    uses ``smooth=1.0`` and no empty-class masking, so it differs from terratorch
    by ~1e-5 on dense batches and by more on no-water batches.

    The STEP 6C v2 combined loss therefore uses ``smp.losses.DiceLoss`` directly
    (see ``CombinedDicePhysicsLoss``) to guarantee exact Dice parity with STEP 5S-A.
    This function is retained only so older diagnostics that import it keep working.
    """
    num_classes = logits.shape[1]
    probs = F.softmax(logits, dim=1)                          # [B, C, H, W]
    mask = (target != ignore_index).float()                   # [B, H, W]
    safe_target = target.clamp(min=0)
    target_oh = F.one_hot(safe_target, num_classes=num_classes)  # [B, H, W, C]
    target_oh = target_oh.permute(0, 3, 1, 2).float()            # [B, C, H, W]
    mask4 = mask.unsqueeze(1)
    probs = probs * mask4
    target_oh = target_oh * mask4
    dims = (0, 2, 3)
    intersection = (probs * target_oh).sum(dim=dims)          # [C]
    cardinality = probs.sum(dim=dims) + target_oh.sum(dim=dims)
    dice_per_class = (2.0 * intersection + smooth) / (cardinality + smooth)
    return 1.0 - dice_per_class.mean()


class CombinedDicePhysicsLoss(nn.Module):
    """Dice segmentation loss plus the STEP 6A topographic prior.

    This is the primary combined loss for STEP 6C. Using Dice (not CE) keeps
    the segmentation component identical to the STEP 5S-A baseline, so the only
    experimental variable is the topographic regularization term.

    Total loss:  L = Dice(logits, target) + lambda_topo * L_topo
    """

    def __init__(
        self,
        *,
        lambda_topo: float = 0.05,
        ignore_index: int = -1,
        water_class: int = 1,
        dice_smooth: float = 0.0,
        elevation_margin: float = 0.0,
        elevation_scale: float = 1.0,
        use_elevation_weight: bool = True,
        neighborhood: str = "4",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if lambda_topo < 0:
            raise ValueError("lambda_topo must be non-negative.")
        self.lambda_topo = float(lambda_topo)
        self.ignore_index = ignore_index
        self.dice_smooth = dice_smooth
        # EXACT Dice parity with STEP 5S-A: terratorch's `loss: dice` constructs
        # smp.losses.DiceLoss("multiclass", ignore_index=ignore_index). We build the
        # identical object here (smooth defaults to 0.0, eps=1e-7, empty-class masking),
        # so CombinedDicePhysicsLoss(lambda_topo=0) is bit-exact with the 5S-A loss.
        self.loss_dice = smp.losses.DiceLoss(
            mode="multiclass",
            ignore_index=ignore_index,
            smooth=dice_smooth,
        )
        self.loss_topo = TopographicInconsistencyLoss(
            ignore_index=ignore_index,
            water_class=water_class,
            elevation_margin=elevation_margin,
            elevation_scale=elevation_scale,
            use_elevation_weight=use_elevation_weight,
            neighborhood=neighborhood,
            eps=eps,
            reduction="mean",
        )

    def set_lambda_topo(self, value: float) -> None:
        """Update the topographic weight in place (used by epoch-wise lambda schedules)."""
        if value < 0:
            raise ValueError("lambda_topo must be non-negative.")
        self.lambda_topo = float(value)

    def forward(self, logits: Tensor, target: Tensor, topography: Tensor) -> dict[str, Tensor]:
        target = target.to(device=logits.device, dtype=torch.long)
        loss_dice = self.loss_dice(logits, target)
        loss_topo = self.loss_topo(logits=logits, target=target, topography=topography)
        lambda_topo = logits.new_tensor(self.lambda_topo)
        loss_total = loss_dice + lambda_topo * loss_topo
        return {
            "loss_total": loss_total,
            "loss_dice": loss_dice,
            "loss_topo": loss_topo,
            "lambda_topo": lambda_topo,
        }


class CombinedDiceCELoss(nn.Module):
    """Dice + alpha * CrossEntropy (no topographic term).

    Drop-in replacement for CombinedDicePhysicsLoss when lambda_topo=0 and a CE
    term is needed to counter all-background collapse. Keeps the same forward
    signature and dict keys so the training loop requires no changes.

    Total loss:  L = DiceLoss(logits, target) + ce_alpha * CE(logits, target)
    """

    def __init__(
        self,
        *,
        ce_alpha: float = 1.0,
        ignore_index: int = -1,
        dice_smooth: float = 0.0,
        class_weights: "Iterable[float] | Tensor | None" = None,
    ) -> None:
        super().__init__()
        self.lambda_topo = 0.0  # compatibility with training loop
        self.ce_alpha = float(ce_alpha)
        self.loss_dice = smp.losses.DiceLoss(
            mode="multiclass",
            ignore_index=ignore_index,
            smooth=dice_smooth,
        )
        weight = None
        if class_weights is not None:
            weight = torch.as_tensor(class_weights, dtype=torch.float32)
        self.loss_ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def set_lambda_topo(self, value: float) -> None:
        """No-op: compatibility with CombinedDicePhysicsLoss interface."""

    def forward(self, logits: Tensor, target: Tensor, topography: Tensor) -> "dict[str, Tensor]":
        target = target.to(device=logits.device, dtype=torch.long)
        loss_dice = self.loss_dice(logits, target)
        loss_ce = self.loss_ce(logits, target)
        loss_total = loss_dice + self.ce_alpha * loss_ce
        zero = logits.new_tensor(0.0)
        return {
            "loss_total": loss_total,
            "loss_dice": loss_dice,
            "loss_topo": zero,
            "lambda_topo": zero,
        }


class CombinedSegmentationPhysicsLoss(nn.Module):
    """Cross-entropy segmentation loss plus the STEP 6A topographic prior.

    The module is intentionally model-agnostic. Any segmentation model that
    returns class logits shaped ``[B, C, H, W]`` can use it, including the
    TerraMind-L + UPerNet target and the TerraMind base + UNetDecoder control.
    ``topography`` must already be aligned to the logits/target grid; STEP 6A
    does not validate or create those rasters.
    """

    def __init__(
        self,
        *,
        lambda_topo: float = 0.1,
        class_weights: Iterable[float] | Tensor | None = None,
        ignore_index: int = -1,
        water_class: int = 1,
        elevation_margin: float = 0.0,
        elevation_scale: float = 1.0,
        use_elevation_weight: bool = True,
        neighborhood: str = "4",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if lambda_topo < 0:
            raise ValueError("lambda_topo must be non-negative.")

        weight = None
        if class_weights is not None:
            weight = torch.as_tensor(class_weights, dtype=torch.float32)

        self.lambda_topo = float(lambda_topo)
        self.loss_seg = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        self.loss_topo = TopographicInconsistencyLoss(
            ignore_index=ignore_index,
            water_class=water_class,
            elevation_margin=elevation_margin,
            elevation_scale=elevation_scale,
            use_elevation_weight=use_elevation_weight,
            neighborhood=neighborhood,
            eps=eps,
            reduction="mean",
        )

    def forward(self, logits: Tensor, target: Tensor, topography: Tensor) -> dict[str, Tensor]:
        target = target.to(device=logits.device, dtype=torch.long)
        loss_seg = self.loss_seg(logits, target)
        loss_topo = self.loss_topo(logits=logits, target=target, topography=topography)
        lambda_topo = logits.new_tensor(self.lambda_topo)
        loss_total = loss_seg + lambda_topo * loss_topo
        return {
            "loss_total": loss_total,
            "loss_seg": loss_seg,
            "loss_topo": loss_topo,
            "lambda_topo": lambda_topo,
        }
