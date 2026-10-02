import pytest
import torch

from vexa_video.models.video_vae import TinyVideoVAE


def test_vae_shapes_and_gradients() -> None:
    model = TinyVideoVAE(in_channels=3, latent_channels=4, base_channels=8)
    video = torch.randn(2, 3, 8, 32, 32)
    reconstruction, latent = model(video)
    assert reconstruction.shape == video.shape
    assert latent.shape == (2, 4, 4, 8, 8)
    reconstruction.mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_vae_two_x_spatial_compression_preserves_larger_latent_grid() -> None:
    model = TinyVideoVAE(
        in_channels=3,
        latent_channels=4,
        base_channels=8,
        spatial_downsample=2,
        temporal_downsample=2,
    )
    video = torch.randn(2, 3, 8, 32, 32)
    reconstruction, latent = model(video)
    assert reconstruction.shape == video.shape
    assert latent.shape == (2, 4, 4, 16, 16)


def test_vae_rejects_non_power_of_two_compression() -> None:
    with pytest.raises(ValueError, match="power of two"):
        TinyVideoVAE(spatial_downsample=3)
