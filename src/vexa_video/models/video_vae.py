from __future__ import annotations

import torch
from torch import Tensor, nn


class Residual3DBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(8, channels)
        while channels % groups != 0:
            groups -= 1
        self.net = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


class TinyVideoVAE(nn.Module):
    """Small deterministic autoencoder scaffold.

    This starter behaves like an autoencoder rather than a probabilistic VAE; a learned
    posterior (mu/logvar + KL objective) is introduced in the dedicated VAE milestone.
    """

    def __init__(self, in_channels: int = 3, latent_channels: int = 4, base_channels: int = 16) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1),
            Residual3DBlock(base_channels),
            nn.Conv3d(
                base_channels,
                base_channels * 2,
                kernel_size=(3, 4, 4),
                stride=(1, 2, 2),
                padding=(1, 1, 1),
            ),
            nn.SiLU(),
            nn.Conv3d(
                base_channels * 2,
                latent_channels,
                kernel_size=4,
                stride=(2, 2, 2),
                padding=1,
            ),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(
                latent_channels,
                base_channels * 2,
                kernel_size=4,
                stride=(2, 2, 2),
                padding=1,
            ),
            nn.SiLU(),
            nn.ConvTranspose3d(
                base_channels * 2,
                base_channels,
                kernel_size=(3, 4, 4),
                stride=(1, 2, 2),
                padding=(1, 1, 1),
            ),
            Residual3DBlock(base_channels),
            nn.SiLU(),
            nn.Conv3d(base_channels, in_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def encode(self, video: Tensor) -> Tensor:
        self._validate(video)
        return self.encoder(video)

    def decode(self, latents: Tensor) -> Tensor:
        if latents.ndim != 5:
            raise ValueError(f"latents must be [B, C, T, H, W], got {tuple(latents.shape)}")
        return self.decoder(latents)

    def forward(self, video: Tensor) -> tuple[Tensor, Tensor]:
        latents = self.encode(video)
        reconstruction = self.decode(latents)
        return reconstruction, latents

    @staticmethod
    def _validate(video: Tensor) -> None:
        if video.ndim != 5:
            raise ValueError(f"video must be [B, C, T, H, W], got {tuple(video.shape)}")
        _, _, frames, height, width = video.shape
        if frames % 2 or height % 4 or width % 4:
            raise ValueError("starter VAE requires T divisible by 2 and H/W divisible by 4")
