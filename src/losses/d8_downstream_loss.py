"""D8 downstream consistency loss for physics-informed flood segmentation.

Encodes the hydrological prior that flood water cannot pool upstream of dry land:
if pixel i drains downhill to pixel d(i) via the steepest-descent (D8) direction,
then predicting i as water while d(i) is dry is penalized.

Loss formula:
    p_i = softmax(logits_i)[water]
    d(i) = argmax_{j in N8(i)} (h_i - h_j) / dist(i,j)   # D8 steepest descent
    s_i  = max(0, h_i - h_{d(i)})                          # positive drop only
    w_i  = min(1, s_i / s0)                                 # slope weight (flat=0)
    M_i  = valid_i AND valid_{d(i)} AND s_i > 0            # active pixels
    L_D8 = sum_i(M_i * w_i * max(0, p_i - p_{d(i)} - tau)^2)
           / (sum_i(M_i * w_i) + eps)

Key properties:
- D8 direction is NOT differentiable and is treated as fixed conditioning data.
- Gradient flows only through p_i and p_{d(i)}.
- All-dry (p=0 everywhere) satisfies the loss (loss -> 0); Dice+CE anchors segmentation.
- Flat areas (drop ~ 0) are weighted near zero.
- NaN / +/-Inf DEM values are masked out.
- Border pixels whose downstream direction falls outside the tile are excluded.
- Shuffled DEM reroutes d(i), changing which pairs are active -- this is the
  DEM-specific signal that V1 (local 4-neighbor) lacked.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# 8 neighbor directions (dy, dx) and their Euclidean distances.
_D8_DY   = (-1, -1, -1,  0,  0,  1, 1, 1)
_D8_DX   = (-1,  0,  1, -1,  1, -1, 0, 1)
_D8_DIST = (math.sqrt(2), 1.0, math.sqrt(2),
            1.0,          1.0,
            math.sqrt(2), 1.0, math.sqrt(2))
_N_DIR   = 8

# Sentinel: fills padded border positions of the DEM so those neighbors always
# have a more negative slope than any real neighbor and are never selected as D8.
_DEM_PAD_VAL = 1e9


class D8DownstreamLoss(nn.Module):
    """Slope-weighted D8 downstream consistency loss.

    Args:
        ignore_index:  Label value to exclude (-1 for Sen1Floods11).
        water_class:   Index in the class dimension that corresponds to water.
        s0:            Reference drop (m) for slope weight normalisation.
                       Pixels with drop >= s0 get weight 1.0; flat areas (~0).
        tau:           Hinge margin -- penalises only when p_upstream exceeds
                       p_downstream by more than tau.  Prevents penalising tiny
                       random probability differences.
        eps:           Numerical stability denominator floor.
        reduction:     'mean' (normalised by sum of active weights) or 'sum'.
    """

    def __init__(
        self,
        *,
        ignore_index: int = -1,
        water_class: int = 1,
        s0: float = 1.0,
        tau: float = 0.05,
        eps: float = 1e-6,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if s0 <= 0:
            raise ValueError("s0 must be strictly positive.")
        if eps <= 0:
            raise ValueError("eps must be strictly positive.")
        if reduction not in {"mean", "sum"}:
            raise ValueError('reduction must be "mean" or "sum".')

        self.ignore_index = int(ignore_index)
        self.water_class  = int(water_class)
        self.s0           = float(s0)
        self.tau          = float(tau)
        self.eps          = float(eps)
        self.reduction    = reduction

    # ---------------------------------------------------------------------- #
    def forward(
        self,
        logits: Tensor,
        target: Tensor,
        topography: Tensor,
    ) -> Tensor:
        """
        Args:
            logits:     [B, C, H, W] — raw class logits.
            target:     [B, H, W] or [B, 1, H, W] — integer label map.
            topography: [B, H, W] or [B, 1, H, W] — DEM values (metres).

        Returns:
            Scalar loss tensor (differentiable w.r.t. logits).
        """
        if logits.ndim != 4:
            raise ValueError(f"logits must be [B, C, H, W], got {tuple(logits.shape)}")
        B, C, H, W = logits.shape
        if self.water_class >= C:
            raise ValueError(
                f"water_class={self.water_class} out of range for C={C} logits."
            )

        target = _to_bhw_long(target, "target", logits.device)
        topo   = _to_bhw_float(topography, "topography", logits)

        # Water probability map [B, H, W].
        p_water = F.softmax(logits, dim=1)[:, self.water_class]

        # Valid-pixel mask: labelled AND finite DEM.
        valid = (target != self.ignore_index) & torch.isfinite(topo)

        # Build padded tensors [B, H+2, W+2] for the 1-pixel border ring.
        # Border ring of DEM = _DEM_PAD_VAL  → slope to border is very negative.
        # Border ring of p   = 0.0           → safe fallback (never selected).
        # Border ring of valid = False        → border downstream pixels excluded.
        topo_pad  = F.pad(topo,                       (1, 1, 1, 1), value=_DEM_PAD_VAL)
        p_pad     = F.pad(p_water,                    (1, 1, 1, 1), value=0.0)
        valid_pad = F.pad(valid.to(dtype=topo.dtype), (1, 1, 1, 1), value=0.0)

        # Stack per-direction tensors: [B, 8, H, W].
        slope_list  : list[Tensor] = []
        p_nbr_list  : list[Tensor] = []
        v_nbr_list  : list[Tensor] = []

        for k in range(_N_DIR):
            dy, dx, dist = _D8_DY[k], _D8_DX[k], _D8_DIST[k]
            # Slice for direction (dy, dx): neighbor at (y+dy, x+dx) in original
            # corresponds to padded position (y+1+dy, x+1+dx), so the window is
            #   padded[:, dy+1 : dy+1+H, dx+1 : dx+1+W]
            h_nbr = topo_pad  [:, dy + 1 : dy + 1 + H, dx + 1 : dx + 1 + W]
            p_n   = p_pad     [:, dy + 1 : dy + 1 + H, dx + 1 : dx + 1 + W]
            v_n   = valid_pad [:, dy + 1 : dy + 1 + H, dx + 1 : dx + 1 + W]

            # Slope = (h_center - h_neighbor) / dist  (positive = downhill to nbr).
            slope_list.append((topo - h_nbr) / dist)
            p_nbr_list.append(p_n)
            v_nbr_list.append(v_n)

        slopes_stack = torch.stack(slope_list, dim=1)   # [B, 8, H, W]
        p_nbr_stack  = torch.stack(p_nbr_list, dim=1)  # [B, 8, H, W]
        v_nbr_stack  = torch.stack(v_nbr_list, dim=1)  # [B, 8, H, W]

        # NaN slopes arise when center DEM or a neighbour DEM is NaN.  Replace
        # with -1e9 so they are never selected as steepest-descent direction.
        slopes_stack = slopes_stack.nan_to_num(nan=-1e9, posinf=-1e9, neginf=-1e9)

        # D8: steepest-descent direction for each pixel.
        d8_slope, d8_dir = slopes_stack.max(dim=1)   # [B, H, W] each

        # Gather downstream probability and validity at D8 direction.
        idx          = d8_dir.unsqueeze(1)                                    # [B, 1, H, W]
        p_downstream = p_nbr_stack.gather(1, idx).squeeze(1)                 # [B, H, W]
        v_downstream = v_nbr_stack.gather(1, idx).squeeze(1).bool()          # [B, H, W]

        # Drop = positive D8 slope (0 for sinks / flat areas).
        drop = d8_slope.clamp(min=0.0)

        # Slope weight in [0, 1].
        w = (drop / self.s0).clamp(0.0, 1.0)

        # Active: centre valid, downstream valid, positive drop.
        active = valid & v_downstream & (d8_slope > 0)

        # Hinge-squared violation: penalise p_upstream >> p_downstream.
        hinge = torch.relu(p_water - p_downstream - self.tau).pow(2)

        # Weighted sum.
        wm = w * active.to(dtype=topo.dtype)

        if self.reduction == "sum":
            return (wm * hinge).sum()

        denom = wm.sum() + self.eps
        return (wm * hinge).sum() / denom


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #

def _to_bhw_long(t: Tensor, name: str, device: torch.device) -> Tensor:
    t = t.to(device=device, dtype=torch.long)
    if t.ndim == 4 and t.shape[1] == 1:
        return t[:, 0]
    if t.ndim == 3:
        return t
    raise ValueError(f"{name} must be [B,H,W] or [B,1,H,W], got {tuple(t.shape)}")


def _to_bhw_float(t: Tensor, name: str, ref: Tensor) -> Tensor:
    t = t.to(device=ref.device, dtype=ref.dtype)
    if t.ndim == 4 and t.shape[1] == 1:
        return t[:, 0]
    if t.ndim == 3:
        return t
    raise ValueError(f"{name} must be [B,H,W] or [B,1,H,W], got {tuple(t.shape)}")
