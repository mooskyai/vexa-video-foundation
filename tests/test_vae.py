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
