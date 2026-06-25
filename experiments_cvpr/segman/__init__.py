"""SegMAN (CVPR 2025) physics-informed-loss experimental subproject.

Self-contained pipeline that runs the official SegMAN architecture on the
Sen1Floods11 flood/water segmentation dataset, reusing the repository's existing
losses, metrics, DEM topographic-loss and DEM-shuffle machinery. Only the loss
changes across the four experimental variants; the architecture is fixed.
"""
