from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn

from losses.physics_topographic_loss import TopographicInconsistencyLoss


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
