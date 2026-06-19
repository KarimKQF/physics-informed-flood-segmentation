from __future__ import annotations

from torch import Tensor, nn


class SimpleSegmentationCNN(nn.Module):
    """Tiny binary segmentation CNN for technical smoke tests."""

    def __init__(self, in_channels: int = 3, hidden_channels: int = 16) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive.")

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, image: Tensor) -> Tensor:
        if image.ndim != 4:
            raise ValueError(f"image must have shape [B, C, H, W], got {tuple(image.shape)}.")
        return self.net(image)
