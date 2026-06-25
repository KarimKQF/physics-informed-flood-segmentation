"""Pure-PyTorch selective scan for SegMAN's SS2D / VSSM token mixer.

SegMAN's official encoder/decoder import a compiled CUDA kernel
(``selective_scan_cuda_oflex``) that cannot be built on this Windows host
(no CUDA toolkit / MSVC). This module provides a mathematically equivalent
pure-PyTorch implementation that runs on the GPU without any custom kernels.

Two implementations are provided:

* :func:`selective_scan_seq` -- an obviously-correct sequential reference.
* :func:`selective_scan_ref` -- a chunked parallel scan (default) that is
  numerically stable and fast enough for training; it is unit-tested for
  parity against the sequential reference in the smoke tests.

Tensor layout (matching SegMAN ``VSSM.forward`` -> ``SelectiveScanOflex``):

    u           : [B, K*D, L]   input features (the cross-scanned tensor ``xs``)
    delta       : [B, K*D, L]   raw dt (softplus + bias applied inside here)
    A           : [K*D, N]      negative state matrix (already ``-exp(A_log)``)
    B (Bmat)    : [B, K, N, L]   grouped input projection (K groups)
    C (Cmat)    : [B, K, N, L]   grouped output projection (K groups)
    D           : [K*D]          skip connection (optional)
    delta_bias  : [K*D]          dt bias (optional)

The K scan directions are folded into the channel dimension as ``K`` groups,
each group owning its own ``B``/``C`` of shape ``[N, L]`` shared across the
``D`` channels of that group. ``N = d_state`` is 1 for SegMAN, so this is cheap.

Returns ``y`` of shape ``[B, K*D, L]`` (the SSM output ``+ D * u``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _prepare(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor | None,
    delta_bias: Tensor | None,
    delta_softplus: bool,
):
    """Reshape grouped inputs to [Bb, K, Dg, L] / [Bb, K, N, L] and apply dt bias."""
    Bb, KD, L = u.shape
    K = B.shape[1]
    N = B.shape[2]
    Dg = KD // K

    # Work in fp32 for numerical stability regardless of autocast.
    u = u.float()
    delta = delta.float()
    A = A.float()
    Bm = B.float()
    Cm = C.float()

    delta = delta.view(Bb, K, Dg, L)
    u = u.view(Bb, K, Dg, L)
    A = A.view(K, Dg, N)
    if delta_bias is not None:
        delta = delta + delta_bias.float().view(1, K, Dg, 1)
    if delta_softplus:
        delta = F.softplus(delta)

    return Bb, K, Dg, N, L, u, delta, A, Bm, Cm


def selective_scan_seq(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor | None = None,
    delta_bias: Tensor | None = None,
    delta_softplus: bool = False,
) -> Tensor:
    """Sequential reference implementation (slow, obviously correct)."""
    Bb, K, Dg, N, L, u, delta, A, Bm, Cm = _prepare(
        u, delta, A, B, C, D, delta_bias, delta_softplus
    )
    deltaA = torch.exp(delta.unsqueeze(-1) * A.view(1, K, Dg, 1, N))  # [B,K,Dg,L,N]
    # deltaB_u[b,k,d,l,n] = delta * B[b,k,n,l] * u
    deltaB_u = (
        delta.unsqueeze(-1)
        * Bm.permute(0, 1, 3, 2).unsqueeze(2)  # [B,K,1,L,N]
        * u.unsqueeze(-1)
    )
    h = u.new_zeros(Bb, K, Dg, N)
    ys = []
    for t in range(L):
        h = deltaA[:, :, :, t] * h + deltaB_u[:, :, :, t]  # [B,K,Dg,N]
        # y_t = sum_n C[b,k,n,t] * h
        y_t = (h * Cm[:, :, :, t].unsqueeze(2)).sum(-1)  # [B,K,Dg]
        ys.append(y_t)
    y = torch.stack(ys, dim=-1)  # [B,K,Dg,L]
    if D is not None:
        y = y + D.float().view(1, K, Dg, 1) * u
    return y.reshape(Bb, K * Dg, L).to(dtype=torch.float32)


def selective_scan_ref(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor | None = None,
    delta_bias: Tensor | None = None,
    delta_softplus: bool = False,
    chunk_size: int = 128,
) -> Tensor:
    """Chunked parallel scan (Mamba-2 / SSD segment-sum form). Default path.

    The sequence is split into chunks of length ``c``. Inside a chunk every
    decay is expressed as ``exp(a_t - a_s)`` with ``s <= t`` and ``a`` the
    cumulative log-decay, so each exponent is non-positive and cannot overflow.
    Only the cheap chunk-to-chunk carry recurrence loops in Python (``L/c``
    iterations). The result is exact (up to float error) versus
    :func:`selective_scan_seq`.
    """
    Bb, K, Dg, N, L, u, delta, A, Bm, Cm = _prepare(
        u, delta, A, B, C, D, delta_bias, delta_softplus
    )

    dA = delta.unsqueeze(-1) * A.view(1, K, Dg, 1, N)          # [B,K,Dg,L,N]
    dBu = (
        delta.unsqueeze(-1)
        * Bm.permute(0, 1, 3, 2).unsqueeze(2)                  # [B,K,1,L,N]
        * u.unsqueeze(-1)
    )                                                          # [B,K,Dg,L,N]
    Ct = Cm.permute(0, 1, 3, 2)                                # [B,K,L,N]

    c = int(chunk_size)
    pad = (c - L % c) % c
    if pad:
        dA = F.pad(dA, (0, 0, 0, pad))      # pad along L (dim=3)
        dBu = F.pad(dBu, (0, 0, 0, pad))
        Ct = F.pad(Ct, (0, 0, 0, pad))
    Lp = L + pad
    nc = Lp // c

    # [B,K,Dg,nc,c,N]
    dA_c = dA.view(Bb, K, Dg, nc, c, N)
    dBu_c = dBu.view(Bb, K, Dg, nc, c, N)
    C_c = Ct.view(Bb, K, nc, c, N)

    a = torch.cumsum(dA_c, dim=4)                              # [B,K,Dg,nc,c,N]
    # segment sum matrix: Lmat[t,s] = exp(a_t - a_s) for s<=t, else 0
    a_t = a.unsqueeze(5)                                       # [...,c,1,N]
    a_s = a.unsqueeze(4)                                       # [...,1,c,N]
    # Mask the exponent (not the result): upper triangle -> -inf so exp == 0.
    # (exp(large_positive)*0 would be inf*0 == nan, hence masking before exp.)
    tri_mask = torch.tril(a.new_ones(c, c, dtype=torch.bool)).view(1, 1, 1, 1, c, c, 1)
    diff = torch.where(tri_mask, a_t - a_s, a.new_full((), float("-inf")))
    Lmat = torch.exp(diff)                                     # [B,K,Dg,nc,c,c,N]

    # intra-chunk output (no carry): h_intra[t] = sum_{s<=t} Lmat[t,s] * dBu_s
    h_intra = (Lmat * dBu_c.unsqueeze(4)).sum(dim=5)           # [B,K,Dg,nc,c,N]

    # carry decay to each position within its chunk = exp(a_t)
    decay_to_t = torch.exp(a)                                  # [B,K,Dg,nc,c,N]
    chunk_final = h_intra[:, :, :, :, -1, :]                   # [B,K,Dg,nc,N]
    chunk_decay = decay_to_t[:, :, :, :, -1, :]                # [B,K,Dg,nc,N]

    # inter-chunk carry recurrence (loop over nc chunks only)
    states = []
    state = u.new_zeros(Bb, K, Dg, N)
    for j in range(nc):
        states.append(state)
        state = chunk_decay[:, :, :, j] * state + chunk_final[:, :, :, j]
    state_in = torch.stack(states, dim=3)                     # [B,K,Dg,nc,N]

    h = decay_to_t * state_in.unsqueeze(4) + h_intra          # [B,K,Dg,nc,c,N]
    y = (h * C_c.unsqueeze(2)).sum(-1)                         # [B,K,Dg,nc,c]
    y = y.reshape(Bb, K, Dg, Lp)[:, :, :, :L]                 # drop padding

    if D is not None:
        y = y + D.float().view(1, K, Dg, 1) * u
    return y.reshape(Bb, K * Dg, L).to(dtype=torch.float32)
