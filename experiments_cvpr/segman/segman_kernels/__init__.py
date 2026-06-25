"""Pure-PyTorch kernels and compatibility shims for running SegMAN without
its compiled CUDA/Triton/mmcv/mmseg dependencies."""

from .compat import (
    install_shims,
    load_segman_decoder_module,
    load_segman_encoder_module,
)
from .selective_scan import selective_scan_ref, selective_scan_seq

__all__ = [
    "install_shims",
    "load_segman_encoder_module",
    "load_segman_decoder_module",
    "selective_scan_ref",
    "selective_scan_seq",
]
