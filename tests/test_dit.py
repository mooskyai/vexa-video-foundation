import torch

from vexa_video.models.dit import VideoDiT


def test_dit_preserves_latent_shape_and_backpropagates() -> None:
    model = VideoDiT(
        latent_channels=4,
        text_dim=32,
        hidden_size=64,
        layers=2,
        heads=4,
        patch_size=(1, 2, 2),
    )
    latents = torch.randn(2, 4, 4, 8, 8)
    timesteps = torch.tensor([10, 20])
    text = torch.randn(2, 12, 32)
    mask = torch.ones(2, 12, dtype=torch.bool)
    output = model(latents, timesteps, text, mask)
    assert output.shape == latents.shape
    output.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
