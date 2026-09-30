from __future__ import annotations

from torch import Tensor, nn


class MotionProbe(nn.Module):
    """Small supervised temporal sanity probe for the deterministic M1 renderer."""

    def __init__(self, classes: int = 4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(16, 24, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d((2, 4, 4)),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(24 * 2 * 4 * 4, 128),
            nn.SiLU(),
        )
        self.direction_head = nn.Linear(128, classes)
        self.color_head = nn.Linear(128, classes)

    def forward(self, video: Tensor) -> tuple[Tensor, Tensor]:
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError("video must have shape [B, 3, T, H, W]")
        hidden = self.projection(self.features(video))
        return self.direction_head(hidden), self.color_head(hidden)
