from __future__ import annotations

import torch

from vexa_video.models.dit import VideoDiT
from vexa_video.training.m1_conditioning_diagnostics import diagnostic_dit_forward


def _fixture() -> tuple[VideoDiT, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    dit = VideoDiT(
        latent_channels=4,
        text_dim=64,
        hidden_size=48,
        layers=2,
        heads=4,
        patch_size=(1, 2, 2),
    ).eval()
    latents = torch.randn(2, 4, 4, 8, 8)
    timesteps = torch.tensor([500, 245], dtype=torch.long)
    text = torch.randn(2, 12, 64)
    mask = torch.ones(2, 12, dtype=torch.bool)
    return dit, latents, timesteps, text, mask


def test_diagnostic_forward_matches_production_forward() -> None:
    dit, latents, timesteps, text, mask = _fixture()
    shape_mask = torch.zeros_like(mask)
    shape_mask[:, 3:9] = True

    with torch.no_grad():
        expected = dit(latents, timesteps, text, mask)
        actual, trace = diagnostic_dit_forward(
            dit,
            latents,
            timesteps,
            text,
            mask,
            shape_token_mask=shape_mask,
        )

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert trace.pooled.shape == (2, 1, 48)
    assert trace.first_attention.shape[0] == 2
    assert trace.final_attention.shape == trace.first_attention.shape
    assert bool(torch.isfinite(trace.first_entropy).all().item())
    assert bool(torch.isfinite(trace.final_entropy).all().item())
    assert bool(((trace.first_shape_mass >= 0.0) & (trace.first_shape_mass <= 1.0)).all().item())
    assert bool(((trace.final_shape_mass >= 0.0) & (trace.final_shape_mass <= 1.0)).all().item())


def test_disabling_all_text_branches_removes_text_dependence() -> None:
    dit, latents, timesteps, text, mask = _fixture()
    alternate_text = torch.randn_like(text)

    with torch.no_grad():
        first, _ = diagnostic_dit_forward(
            dit,
            latents,
            timesteps,
            text,
            mask,
            disable_pool=True,
            disable_first_attention=True,
            disable_final_attention=True,
        )
        second, _ = diagnostic_dit_forward(
            dit,
            latents,
            timesteps,
            alternate_text,
            mask,
            disable_pool=True,
            disable_first_attention=True,
            disable_final_attention=True,
        )

    assert torch.allclose(first, second, atol=1e-6, rtol=1e-5)
