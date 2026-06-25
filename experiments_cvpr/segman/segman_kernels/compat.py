"""Pure-PyTorch compatibility shims so SegMAN's *original* source runs without
its compiled CUDA / Triton / mmcv / mmseg dependencies.

The official SegMAN encoder and MMSCopE decoder import:
    * ``natten`` (neighborhood attention CUDA op)
    * ``selective_scan_cuda_oflex`` (Mamba SS2D CUDA op)
    * ``csm_triton`` (Triton cross-scan)
    * ``mmcv.cnn.ConvModule`` and ``mmseg`` decode-head infrastructure
    * ``fvcore`` (only used by ``.flops()`` helpers)

None of these can be compiled on the host (no CUDA toolkit / MSVC). We register
drop-in pure-PyTorch modules into ``sys.modules`` *before* importing the
unmodified SegMAN files, then load those files by path. The selective scan and
neighborhood attention are mathematically faithful (scan is unit-tested against
a sequential reference; NATTEN is an unfold-based equivalent with valid-neighbor
masking + central-block relative position bias — exact on interior pixels).

Provenance: external/SegMAN @ commit 9ced66a (CVPR 2025).
"""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .cross_scan import CrossMergeTriton, CrossScanTriton
from .selective_scan import selective_scan_ref

EXTERNAL_SEGMAN = Path(__file__).resolve().parents[3] / "external" / "SegMAN"


# --------------------------------------------------------------------------- #
# NATTEN (neighborhood attention) -- pure-torch via unfold                     #
# --------------------------------------------------------------------------- #
def _pair(x):
    return (x, x) if isinstance(x, int) else tuple(x)


def _unfold_neighbors(x_bchw: Tensor, kernel_size, dilation):
    """[B, C, H, W] -> ([B, C, k*k, H*W] neighbors, [1, 1, k*k, H*W] valid mask)."""
    kh, kw = _pair(kernel_size)
    dh, dw = _pair(dilation)
    ph, pw = (kh - 1) // 2 * dh, (kw - 1) // 2 * dw
    B, C, H, W = x_bchw.shape
    unf = F.unfold(x_bchw, (kh, kw), dilation=(dh, dw), padding=(ph, pw))
    unf = unf.view(B, C, kh * kw, H * W)
    ones = x_bchw.new_ones(1, 1, H, W)
    mask = F.unfold(ones, (kh, kw), dilation=(dh, dw), padding=(ph, pw)).view(1, 1, kh * kw, H * W)
    return unf, mask


def _central_rpb(rpb: Tensor, kh: int, kw: int) -> Tensor:
    """Map NATTEN RPB table [heads, 2kh-1, 2kw-1] to interior [heads, kh*kw]."""
    rh, rw = (kh - 1) // 2, (kw - 1) // 2
    center = rpb[:, kh - 1 - rh: kh - 1 - rh + kh, kw - 1 - rw: kw - 1 - rw + kw]
    return center.reshape(rpb.shape[0], kh * kw)


def na2d_qk(q: Tensor, k: Tensor, kernel_size, dilation=1, rpb: Tensor | None = None) -> Tensor:
    """q, k: [B, heads, H, W, dim] -> attn scores [B, heads, H, W, k*k]."""
    B, nh, H, W, d = q.shape
    kh, kw = _pair(kernel_size)
    qb = q.permute(0, 1, 4, 2, 3).reshape(B * nh, d, H * W)
    kb = k.permute(0, 1, 4, 2, 3).reshape(B * nh, d, H, W)
    k_unf, mask = _unfold_neighbors(kb, (kh, kw), dilation)          # [B*nh,d,k2,HW]
    attn = torch.einsum("bdl,bdkl->blk", qb, k_unf)                  # [B*nh,HW,k2]
    attn = attn.view(B, nh, H * W, kh * kw)
    if rpb is not None:
        attn = attn + _central_rpb(rpb, kh, kw).view(1, nh, 1, kh * kw)
    # mask is [1, 1, k*k, HW]; transpose (not view) to align with [.., HW, k*k]
    mask = mask.transpose(-1, -2).reshape(1, 1, H * W, kh * kw)
    attn = attn.masked_fill(mask == 0, float("-inf"))
    return attn.view(B, nh, H, W, kh * kw)


def na2d_av(attn: Tensor, v: Tensor, kernel_size, dilation=1) -> Tensor:
    """attn: [B, heads, H, W, k*k] (post-softmax), v: [B, heads, H, W, dim]."""
    B, nh, H, W, d = v.shape
    kh, kw = _pair(kernel_size)
    vb = v.permute(0, 1, 4, 2, 3).reshape(B * nh, d, H, W)
    v_unf, _ = _unfold_neighbors(vb, (kh, kw), dilation)             # [B*nh,d,k2,HW]
    a = attn.reshape(B * nh, H * W, kh * kw)
    out = torch.einsum("blk,bdkl->bdl", a, v_unf)                   # [B*nh,d,HW]
    return out.view(B, nh, d, H, W).permute(0, 1, 3, 4, 2).contiguous()


def na2d(q, k, v, kernel_size, dilation=1, rpb=None, scale=None, **kw):
    """Fused-style entry (not used by the default SegMAN path)."""
    attn = na2d_qk(q, k, kernel_size, dilation, rpb)
    if scale is not None:
        attn = attn * scale
    attn = torch.softmax(attn, dim=-1)
    return na2d_av(attn, v, kernel_size, dilation)


def _build_natten_modules():
    nat = types.ModuleType("natten")
    nat.use_fused_na = lambda *a, **k: None
    nat.use_gemm_na = lambda *a, **k: None

    class NeighborhoodAttention2D(nn.Module):  # imported but unused by encoder
        def __init__(self, *a, **k):
            super().__init__()

    nat.NeighborhoodAttention2D = NeighborhoodAttention2D

    func = types.ModuleType("natten.functional")
    func.na2d = na2d
    func.na2d_qk = na2d_qk
    func.na2d_av = na2d_av
    func.natten2dqkrpb = na2d_qk
    func.natten2dav = na2d_av
    nat.functional = func

    flops = types.ModuleType("natten.flops")
    flops.qk_2d_rpb_flop = lambda *a, **k: 0
    flops.av_2d_flop = lambda *a, **k: 0
    flops.add_natten_handle = lambda *a, **k: None
    nat.flops = flops
    return {"natten": nat, "natten.functional": func, "natten.flops": flops}


# --------------------------------------------------------------------------- #
# mmcv.cnn shim                                                                #
# --------------------------------------------------------------------------- #
def _build_norm(norm_cfg, num_features):
    t = (norm_cfg or {}).get("type", "BN")
    if t in ("BN", "BN2d", "SyncBN"):
        return nn.BatchNorm2d(num_features)
    if t == "GN":
        return nn.GroupNorm(norm_cfg.get("num_groups", 1), num_features)
    if t in ("LN", "LN2d"):
        return nn.GroupNorm(1, num_features)
    return nn.BatchNorm2d(num_features)


def _build_act(act_cfg):
    if act_cfg is None:
        return None
    t = act_cfg.get("type", "ReLU")
    return {"ReLU": nn.ReLU, "GELU": nn.GELU, "SiLU": nn.SiLU, "LeakyReLU": nn.LeakyReLU}.get(t, nn.ReLU)()


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias="auto", conv_cfg=None, norm_cfg=None,
                 act_cfg=None, order=("conv", "norm", "act"), **kwargs):
        super().__init__()
        if act_cfg is None and "act_cfg" not in kwargs:
            act_cfg = {"type": "ReLU"}
        use_norm = norm_cfg is not None
        use_bias = (not use_norm) if bias == "auto" else bias
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                              dilation, groups, bias=use_bias)
        self.norm = _build_norm(norm_cfg, out_channels) if use_norm else None
        self.activate = _build_act(act_cfg)

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activate is not None:
            x = self.activate(x)
        return x


class DepthwiseSeparableConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, norm_cfg=None, act_cfg=None, **kwargs):
        super().__init__()
        self.depthwise = ConvModule(in_channels, in_channels, kernel_size, stride, padding,
                                    dilation, groups=in_channels, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.pointwise = ConvModule(in_channels, out_channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


# --------------------------------------------------------------------------- #
# mmseg decode-head infrastructure shim                                        #
# --------------------------------------------------------------------------- #
def _resize(input, size=None, scale_factor=None, mode="nearest", align_corners=None, warning=False):
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class _Registry:
    def register_module(self, *args, **kwargs):
        def deco(cls):
            return cls
        if args and callable(args[0]):
            return args[0]
        return deco


class BaseDecodeHead(nn.Module):
    """Minimal stand-in for mmseg BaseDecodeHead (inference-time attributes only)."""

    def __init__(self, in_channels, channels, *, num_classes, dropout_ratio=0.1,
                 conv_cfg=None, norm_cfg=None, act_cfg=None, in_index=-1,
                 input_transform=None, align_corners=False, loss_decode=None,
                 ignore_index=255, sampler=None, **kwargs):
        super().__init__()
        if act_cfg is None:
            act_cfg = {"type": "ReLU"}
        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index
        self.input_transform = input_transform
        self.align_corners = align_corners
        self.ignore_index = ignore_index
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio and dropout_ratio > 0 else None

    def _transform_inputs(self, inputs):
        if self.input_transform == "multiple_select":
            return [inputs[i] for i in self.in_index]
        if self.input_transform == "resize_concat":
            sel = [inputs[i] for i in self.in_index]
            sel = [_resize(x, size=sel[0].shape[2:], mode="bilinear",
                           align_corners=self.align_corners) for x in sel]
            return torch.cat(sel, dim=1)
        return inputs[self.in_index]

    def cls_seg(self, feat):
        if self.dropout is not None:
            feat = self.dropout(feat)
        return self.conv_seg(feat)


def _build_mmcv_mmseg_modules():
    mods = {}

    mmcv = types.ModuleType("mmcv")
    mmcv.__path__ = []
    cnn = types.ModuleType("mmcv.cnn")
    cnn.ConvModule = ConvModule
    cnn.DepthwiseSeparableConvModule = DepthwiseSeparableConvModule
    mmcv.cnn = cnn
    mods["mmcv"] = mmcv
    mods["mmcv.cnn"] = cnn

    mmseg = types.ModuleType("mmseg")
    mmseg.__path__ = []
    ops = types.ModuleType("mmseg.ops")
    ops.resize = _resize
    models = types.ModuleType("mmseg.models")
    models.__path__ = []
    builder = types.ModuleType("mmseg.models.builder")
    builder.HEADS = _Registry()
    utils = types.ModuleType("mmseg.models.utils")          # for `import *`
    decode_heads = types.ModuleType("mmseg.models.decode_heads")
    decode_heads.__path__ = []
    decode_head = types.ModuleType("mmseg.models.decode_heads.decode_head")
    decode_head.BaseDecodeHead = BaseDecodeHead

    mmseg.ops = ops
    mmseg.models = models
    models.builder = builder
    models.utils = utils
    models.decode_heads = decode_heads
    decode_heads.decode_head = decode_head

    mods.update({
        "mmseg": mmseg,
        "mmseg.ops": ops,
        "mmseg.models": models,
        "mmseg.models.builder": builder,
        "mmseg.models.utils": utils,
        "mmseg.models.decode_heads": decode_heads,
        "mmseg.models.decode_heads.decode_head": decode_head,
    })
    return mods


# --------------------------------------------------------------------------- #
# selective_scan_cuda_oflex / csm_triton / fvcore stubs                        #
# --------------------------------------------------------------------------- #
def _build_misc_modules():
    mods = {}

    ssc = types.ModuleType("selective_scan_cuda_oflex")

    def _unused(*a, **k):
        raise RuntimeError("selective_scan_cuda_oflex stub should not be called; "
                           "VSSM._selective_scan is monkeypatched to pure-torch.")

    ssc.fwd = _unused
    ssc.bwd = _unused
    mods["selective_scan_cuda_oflex"] = ssc

    csm = types.ModuleType("csm_triton")
    csm.CrossScanTriton = CrossScanTriton
    csm.CrossMergeTriton = CrossMergeTriton
    mods["csm_triton"] = csm

    fvcore = types.ModuleType("fvcore")
    fvcore.__path__ = []
    fvnn = types.ModuleType("fvcore.nn")
    for name in ("FlopCountAnalysis", "flop_count_table", "flop_count_str", "flop_count",
                 "parameter_count"):
        setattr(fvnn, name, lambda *a, **k: {})
    fvcore.nn = fvnn
    mods["fvcore"] = fvcore
    mods["fvcore.nn"] = fvnn
    return mods


_INSTALLED = False


def install_shims():
    """Register all pure-torch shim modules into ``sys.modules`` (idempotent)."""
    global _INSTALLED
    if _INSTALLED:
        return
    for name, mod in {
        **_build_natten_modules(),
        **_build_mmcv_mmseg_modules(),
        **_build_misc_modules(),
    }.items():
        sys.modules.setdefault(name, mod)
    _INSTALLED = True


def _load_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _patch_vssm(module):
    """Replace VSSM._selective_scan (CUDA) with the pure-torch reference scan."""
    vssm = getattr(module, "VSSM", None)
    if vssm is None:
        return

    def _selective_scan(self, u, delta, A, B, C, D=None, delta_bias=None,
                        delta_softplus=True, nrows=None, backnrows=None, ssoflex=False):
        return selective_scan_ref(u, delta, A, B, C, D, delta_bias, bool(delta_softplus))

    vssm._selective_scan = _selective_scan


def load_segman_encoder_module():
    """Import the unmodified SegMAN classification encoder (returns the module)."""
    install_shims()
    path = EXTERNAL_SEGMAN / "models" / "segman_encoder.py"
    mod = _load_by_path("segman_encoder_vendored", path)
    _patch_vssm(mod)
    return mod


def load_segman_decoder_module():
    """Import the unmodified SegMAN MMSCopE decoder (returns the module)."""
    install_shims()
    path = EXTERNAL_SEGMAN / "segmentation" / "mmseg" / "models" / "decode_heads" / "segman_decoder.py"
    mod = _load_by_path("mmseg.models.decode_heads.segman_decoder", path)
    _patch_vssm(mod)
    return mod
