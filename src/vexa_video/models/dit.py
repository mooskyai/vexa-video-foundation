from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn.functional as F
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


def spatiotemporal_position_embedding(
    grid: tuple[int, int, int],
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Create deterministic 3D Fourier features for temporal/spatial patch positions."""
    frames, height, width = grid
    axes = [
        torch.linspace(-1.0, 1.0, steps=size, device=device, dtype=torch.float32)
        for size in (frames, height, width)
    ]
    tt, yy, xx = torch.meshgrid(*axes, indexing="ij")
    coordinates = torch.stack((tt, yy, xx), dim=-1).reshape(-1, 3)
    bands = max(1, math.ceil(dim / 6))
    frequencies = math.pi * torch.pow(
        2.0,
        torch.arange(bands, device=device, dtype=torch.float32),
    )
    phases = coordinates.unsqueeze(-1) * frequencies
    features = torch.cat((torch.sin(phases), torch.cos(phases)), dim=-1).reshape(
        coordinates.shape[0], -1
    )
    if features.shape[1] < dim:
        padding = torch.zeros(
            features.shape[0],
            dim - features.shape[1],
            device=device,
            dtype=features.dtype,
        )
        features = torch.cat((features, padding), dim=-1)
    return features[:, :dim].to(dtype=dtype).unsqueeze(0)


def token_text_attention(video_tokens: Tensor, text_tokens: Tensor, text_mask: Tensor) -> Tensor:
    """Parameter-free token attention using the existing learned text projection space."""
    if video_tokens.ndim != 3 or text_tokens.ndim != 3:
        raise ValueError("video_tokens and text_tokens must be rank-3 tensors")
    if text_mask.shape != text_tokens.shape[:2]:
        raise ValueError("text_mask must have shape [B, L]")
    if video_tokens.shape[0] != text_tokens.shape[0]:
        raise ValueError("video/text batch sizes must match")
    if not bool(text_mask.any(dim=1).all().item()):
        raise ValueError("every text sequence must contain at least one unmasked token")

    query = F.normalize(video_tokens, dim=-1)
    key = F.normalize(text_tokens, dim=-1)
    scores = torch.matmul(query, key.transpose(1, 2)) * 4.0
    scores = scores.masked_fill(~text_mask.unsqueeze(1), torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, text_tokens)


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
        grid = cast(tuple[int, int, int], tuple(patches.shape[2:]))
        tokens = patches.flatten(2).transpose(1, 2)
        positions = spatiotemporal_position_embedding(
            grid,
            tokens.shape[-1],
            device=tokens.device,
            dtype=tokens.dtype,
        )

        time_cond = self.time_mlp(timestep_embedding(timesteps, tokens.shape[-1])).unsqueeze(1)
        projected_text = F.layer_norm(
            self.text_proj(text_tokens),
            (tokens.shape[-1],),
        )
        mask = text_mask.to(dtype=projected_text.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled_text = (projected_text * mask).sum(dim=1) / denom
        text_cond = pooled_text.unsqueeze(1)
        token_cond = token_text_attention(tokens, projected_text, text_mask)
        tokens = tokens + positions + time_cond + text_cond + token_cond
        tokens = self.blocks(tokens)
        tokens = tokens + token_text_attention(tokens, projected_text, text_mask)
        tokens = self.out(self.norm(tokens))

        tokens = tokens.view(batch, grid[0], grid[1], grid[2], channels, pt, ph, pw)
        tokens = tokens.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return cast(Tensor, tokens.view(batch, channels, frames, height, width))
