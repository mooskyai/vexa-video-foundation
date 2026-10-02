from __future__ import annotations

from typing import cast

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
        return x + cast(Tensor, self.net(x))


def _compression_stages(factor: int, *, name: str) -> int:
    if factor <= 0 or factor & (factor - 1):
        raise ValueError(f"{name} must be a positive power of two")
    return factor.bit_length() - 1


class TinyVideoVAE(nn.Module):
    """Small deterministic video autoencoder with configurable power-of-two compression.

    This scaffold behaves like an autoencoder rather than a probabilistic VAE; a learned
    posterior (mu/logvar + KL objective) is introduced in the dedicated VAE milestone.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 16,
        spatial_downsample: int = 4,
        temporal_downsample: int = 2,
    ) -> None:
        super().__init__()
        spatial_stages = _compression_stages(spatial_downsample, name="spatial_downsample")
        temporal_stages = _compression_stages(temporal_downsample, name="temporal_downsample")
        stages = max(spatial_stages, temporal_stages)
        if stages == 0:
            raise ValueError("at least one spatial or temporal compression stage is required")

        self.spatial_downsample = spatial_downsample
        self.temporal_downsample = temporal_downsample

        encoder_layers: list[nn.Module] = [
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1),
            Residual3DBlock(base_channels),
        ]
        stage_strides: list[tuple[int, int, int]] = []
        stage_channels: list[int] = [base_channels]
        current_channels = base_channels
        for stage in range(stages):
            temporal_stride = 2 if stage < temporal_stages else 1
            spatial_stride = 2 if stage < spatial_stages else 1
            stride = (temporal_stride, spatial_stride, spatial_stride)
            stage_strides.append(stride)
            is_last = stage == stages - 1
            next_channels = latent_channels if is_last else base_channels * min(2 ** (stage + 1), 4)
            kernel = (
                4 if stride[0] == 2 else 3,
                4 if stride[1] == 2 else 3,
                4 if stride[2] == 2 else 3,
            )
            encoder_layers.append(
                nn.Conv3d(
                    current_channels,
                    next_channels,
                    kernel_size=kernel,
                    stride=stride,
                    padding=1,
                )
            )
            if not is_last:
                encoder_layers.extend([nn.SiLU(), Residual3DBlock(next_channels)])
            current_channels = next_channels
            stage_channels.append(next_channels)
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
        current_channels = latent_channels
        for stage in reversed(range(stages)):
            stride = stage_strides[stage]
            target_channels = stage_channels[stage]
            kernel = (
                4 if stride[0] == 2 else 3,
                4 if stride[1] == 2 else 3,
                4 if stride[2] == 2 else 3,
            )
            decoder_layers.append(
                nn.ConvTranspose3d(
                    current_channels,
                    target_channels,
                    kernel_size=kernel,
                    stride=stride,
                    padding=1,
                )
            )
            decoder_layers.extend([nn.SiLU(), Residual3DBlock(target_channels)])
            current_channels = target_channels
        decoder_layers.extend(
            [
                nn.SiLU(),
                nn.Conv3d(base_channels, in_channels, kernel_size=3, padding=1),
                nn.Tanh(),
            ]
        )
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, video: Tensor) -> Tensor:
        self._validate(video)
        return cast(Tensor, self.encoder(video))

    def decode(self, latents: Tensor) -> Tensor:
        if latents.ndim != 5:
            raise ValueError(f"latents must be [B, C, T, H, W], got {tuple(latents.shape)}")
        return cast(Tensor, self.decoder(latents))

    def forward(self, video: Tensor) -> tuple[Tensor, Tensor]:
        latents = self.encode(video)
        reconstruction = self.decode(latents)
        return reconstruction, latents

    def _validate(self, video: Tensor) -> None:
        if video.ndim != 5:
            raise ValueError(f"video must be [B, C, T, H, W], got {tuple(video.shape)}")
        _, _, frames, height, width = video.shape
        if frames % self.temporal_downsample:
            raise ValueError("frames must be divisible by temporal_downsample")
        if height % self.spatial_downsample or width % self.spatial_downsample:
            raise ValueError("height/width must be divisible by spatial_downsample")
