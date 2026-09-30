from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor, nn


def timestep_embedding(timesteps: Tensor, dim: int, max_period: int = 10_000) -> Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half, 1)
    )
    args = timesteps.float()[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class VideoDiT(nn.Module):
    def __init__(
        self,
        latent_channels: int,
        text_dim: int,
        hidden_size: int,
        layers: int,
        heads: int,
        patch_size: tuple[int, int, int] = (1, 2, 2),
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.patch_size = patch_size
        self.patch_embed = nn.Conv3d(
            latent_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        block = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            block,
            num_layers=layers,
            enable_nested_tensor=False,
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.text_proj = nn.Linear(text_dim, hidden_size)
        patch_volume = math.prod(patch_size)
        self.out = nn.Linear(hidden_size, latent_channels * patch_volume)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self, latents: Tensor, timesteps: Tensor, text_tokens: Tensor, text_mask: Tensor
    ) -> Tensor:
        if latents.ndim != 5:
            raise ValueError("latents must have shape [B, C, T, H, W]")
        if text_tokens.ndim != 3:
            raise ValueError("text_tokens must have shape [B, L, D]")
        batch, channels, frames, height, width = latents.shape
        if channels != self.latent_channels:
            raise ValueError(f"expected {self.latent_channels} latent channels, got {channels}")

        pt, ph, pw = self.patch_size
        if frames % pt or height % ph or width % pw:
            raise ValueError("latent dimensions must be divisible by patch size")

        patches = self.patch_embed(latents)
        grid = patches.shape[2:]
        tokens = patches.flatten(2).transpose(1, 2)

        time_cond = self.time_mlp(timestep_embedding(timesteps, tokens.shape[-1])).unsqueeze(1)
        mask = text_mask.to(dtype=text_tokens.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled_text = (text_tokens * mask).sum(dim=1) / denom
        text_cond = self.text_proj(pooled_text).unsqueeze(1)
        tokens = tokens + time_cond + text_cond
        tokens = self.blocks(tokens)
        tokens = self.out(self.norm(tokens))

        tokens = tokens.view(batch, grid[0], grid[1], grid[2], channels, pt, ph, pw)
        tokens = tokens.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return cast(Tensor, tokens.view(batch, channels, frames, height, width))
