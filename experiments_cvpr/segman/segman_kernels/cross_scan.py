"""Pure-PyTorch VMamba cross-scan / cross-merge (replaces ``csm_triton``).

Four scan directions: row-major, column-major (transpose), and their reverses.
All ops are differentiable, so no custom autograd Function is required; we expose
``.apply`` to match SegMAN's ``CrossScanTriton.apply`` / ``CrossMergeTriton.apply``
call convention.
"""

from __future__ import annotations

import torch
from torch import Tensor


def cross_scan(x: Tensor) -> Tensor:
    """[B, C, H, W] -> [B, 4, C, H*W] (4 directional flattenings)."""
    B, C, H, W = x.shape
    x0 = x.flatten(2)                       # row-major
    x1 = x.transpose(2, 3).flatten(2)       # column-major
    return torch.stack([x0, x1, torch.flip(x0, dims=[-1]), torch.flip(x1, dims=[-1])], dim=1)


def cross_merge(ys: Tensor) -> Tensor:
    """[B, 4, C, H, W] -> [B, C, H*W] (sum the 4 directions back to a single map)."""
    B, K, C, H, W = ys.shape
    L = H * W
    ys = ys.view(B, K, C, L)
    a = ys[:, 0] + torch.flip(ys[:, 2], dims=[-1])
    b = ys[:, 1] + torch.flip(ys[:, 3], dims=[-1])
    y = a + b.view(B, C, W, H).transpose(2, 3).reshape(B, C, L)
    return y


class CrossScanTriton:
    @staticmethod
    def apply(x: Tensor) -> Tensor:
        return cross_scan(x)


class CrossMergeTriton:
    @staticmethod
    def apply(ys: Tensor) -> Tensor:
        return cross_merge(ys)
