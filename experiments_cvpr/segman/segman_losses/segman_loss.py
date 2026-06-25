"""Config-driven loss selector for the SegMAN flood-segmentation experiments.

Four mutually exclusive modes, selected from config (never hardcoded):

    ce                          -> L = CE
    dice_ce                     -> L = Dice + alpha * CE
    dice_ce_topo                -> L = Dice + alpha * CE + lambda_topo * Topo
    dice_ce_topo_dem_shuffled   -> identical loss to dice_ce_topo; the DEM is
                                   spatially shuffled across samples by the
                                   datamodule (a reproducible derangement), so
                                   only the *data* differs, not the loss math.

Formulation (matches the project convention):
    L_DiceCE = L_Dice + alpha * L_CE
    L_total  = L_DiceCE + lambda_topo * L_topo

Components are reused from the existing repository:
    * Dice : ``segmentation_models_pytorch.losses.DiceLoss`` (exact parity with
      the TerraMind baselines' ``loss: dice``).
    * CE   : ``torch.nn.CrossEntropyLoss`` with ``ignore_index``.
    * Topo : ``losses.physics_topographic_loss.TopographicInconsistencyLoss``.

Every component is logged separately (loss_ce / loss_dice / loss_topo /
loss_total). For ``ce`` no Dice or Topo is computed; for ``dice_ce`` Topo is
disabled (loss_topo == 0).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import segmentation_models_pytorch as smp
import torch
from torch import Tensor, nn

# Reuse the repository's topographic loss from src/.
_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from losses.physics_topographic_loss import TopographicInconsistencyLoss  # noqa: E402

MODES = ("ce", "dice_ce", "dice_ce_topo", "dice_ce_topo_dem_shuffled")
_TOPO_MODES = ("dice_ce_topo", "dice_ce_topo_dem_shuffled")


class SegManCombinedLoss(nn.Module):
    def __init__(
        self,
        mode: str,
        *,
        ce_alpha: float = 1.0,
        lambda_topo: float = 0.0,
        ignore_index: int = -1,
        water_class: int = 1,
        dice_smooth: float = 0.0,
        class_weights: "list[float] | None" = None,
        elevation_margin: float = 0.0,
        elevation_scale: float = 1.0,
        use_elevation_weight: bool = True,
        neighborhood: str = "4",
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"Unknown loss mode {mode!r}; choose from {MODES}")
        self.mode = mode
        self.use_dice = mode != "ce"
        self.use_topo = mode in _TOPO_MODES
        self.ce_alpha = float(ce_alpha)
        self.lambda_topo = float(lambda_topo)
        self.ignore_index = int(ignore_index)

        weight = torch.as_tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.loss_ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        if self.use_dice:
            self.loss_dice = smp.losses.DiceLoss(
                mode="multiclass", ignore_index=ignore_index, smooth=dice_smooth
            )
        if self.use_topo:
            self.loss_topo = TopographicInconsistencyLoss(
                ignore_index=ignore_index,
                water_class=water_class,
                elevation_margin=elevation_margin,
                elevation_scale=elevation_scale,
                use_elevation_weight=use_elevation_weight,
                neighborhood=neighborhood,
                reduction="mean",
            )

    def set_lambda_topo(self, value: float) -> None:
        if value < 0:
            raise ValueError("lambda_topo must be non-negative.")
        self.lambda_topo = float(value)

    def forward(
        self, logits: Tensor, target: Tensor, topography: Tensor | None = None
    ) -> dict[str, Tensor]:
        target = target.to(device=logits.device, dtype=torch.long)
        zero = logits.new_tensor(0.0)

        loss_ce = self.loss_ce(logits, target)
        loss_dice = self.loss_dice(logits, target) if self.use_dice else zero

        if self.use_topo:
            if topography is None:
                raise ValueError("topography is required for topographic loss modes.")
            loss_topo = self.loss_topo(logits=logits, target=target, topography=topography)
        else:
            loss_topo = zero

        if self.mode == "ce":
            loss_total = loss_ce
        else:
            loss_total = loss_dice + self.ce_alpha * loss_ce + self.lambda_topo * loss_topo

        return {
            "loss_total": loss_total,
            "loss_ce": loss_ce,
            "loss_dice": loss_dice,
            "loss_topo": loss_topo,
            "lambda_topo": logits.new_tensor(self.lambda_topo),
        }


def build_loss(config: dict[str, Any]) -> SegManCombinedLoss:
    loss_cfg = config["loss"]
    topo_cfg = loss_cfg.get("topo", {})
    return SegManCombinedLoss(
        mode=str(loss_cfg["mode"]),
        ce_alpha=float(loss_cfg.get("ce_alpha", 1.0)),
        lambda_topo=float(loss_cfg.get("lambda_topo", 0.0)),
        ignore_index=int(loss_cfg.get("ignore_index", -1)),
        water_class=int(loss_cfg.get("water_class", 1)),
        dice_smooth=float(loss_cfg.get("dice_smooth", 0.0)),
        class_weights=loss_cfg.get("class_weights"),
        elevation_margin=float(topo_cfg.get("elevation_margin", 0.0)),
        elevation_scale=float(topo_cfg.get("elevation_scale", 1.0)),
        use_elevation_weight=bool(topo_cfg.get("use_elevation_weight", True)),
        neighborhood=str(topo_cfg.get("neighborhood", "4")),
    )


def lambda_for_epoch(config: dict[str, Any], epoch: int) -> float:
    """Optional epoch-wise lambda schedule (constant or warmup_linear)."""
    loss_cfg = config["loss"]
    base = float(loss_cfg.get("lambda_topo", 0.0))
    schedule = loss_cfg.get("lambda_schedule") or {"type": "constant"}
    kind = str(schedule.get("type", "constant")).lower()
    if kind == "constant":
        return base
    if kind == "warmup_linear":
        warmup = int(schedule.get("warmup_epochs", 0))
        ramp = int(schedule.get("ramp_epochs", 0))
        if epoch <= warmup:
            return 0.0
        if ramp <= 0 or epoch >= warmup + ramp:
            return base
        return base * ((epoch - warmup) / ramp)
    raise ValueError(f"Unsupported lambda schedule: {kind}")
