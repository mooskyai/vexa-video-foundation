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
    """Projection-driven cross-attention over text tokens without a separate QKV module."""
    if video_tokens.ndim != 3 or text_tokens.ndim != 3:
        raise ValueError("video_tokens and text_tokens must be rank-3 tensors")
    if text_mask.shape != text_tokens.shape[:2]:
        raise ValueError("text_mask must have shape [B, L]")
    if video_tokens.shape[0] != text_tokens.shape[0]:
        raise ValueError("video/text batch sizes must match")
    if video_tokens.shape[-1] != text_tokens.shape[-1]:
        raise ValueError("video/text feature dimensions must match")
    if not bool(text_mask.any(dim=1).all().item()):
        raise ValueError("every text sequence must contain at least one unmasked token")

    scale = 1.0 / math.sqrt(max(video_tokens.shape[-1], 1))
    scores = torch.matmul(video_tokens, text_tokens.transpose(1, 2)) * scale
    scores = scores.masked_fill(~text_mask.unsqueeze(1), torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1)
    attended = torch.matmul(weights, text_tokens)
    return F.layer_norm(attended, (attended.shape[-1],))


def importance_weighted_text_pool(text_tokens: Tensor, text_mask: Tensor) -> Tensor:
    """Pool text while letting the learned projection amplify informative tokens."""
    if text_tokens.ndim != 3:
        raise ValueError("text_tokens must have shape [B, L, D]")
    if text_mask.shape != text_tokens.shape[:2]:
        raise ValueError("text_mask must have shape [B, L]")
    if not bool(text_mask.any(dim=1).all().item()):
        raise ValueError("every text sequence must contain at least one unmasked token")

    energy = text_tokens.square().mean(dim=-1).sqrt()
    scores = energy / math.sqrt(max(text_tokens.shape[-1], 1))
    scores = scores.masked_fill(~text_mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1)
    pooled = (weights.unsqueeze(-1) * text_tokens).sum(dim=1)
    return F.layer_norm(pooled, (pooled.shape[-1],))


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
        projected_text = self.text_proj(text_tokens)
        text_cond = importance_weighted_text_pool(projected_text, text_mask).unsqueeze(1)

        # Shape words need to bind to spatial tokens, not only to raw patch content.
        # Build the query state from patch + position + timestep + global text first,
        # then apply token-level text attention so its routing can depend on where and
        # when a patch exists in the denoising trajectory.
        hidden = tokens + positions + time_cond + text_cond
        hidden = hidden + token_text_attention(hidden, projected_text, text_mask)

        # The second text injection used to happen after every Transformer block,
        # immediately before the output projection. That gave the model no spatial
        # processing stage in which to turn the final text response into boundary
        # geometry. Inject it before the final block instead, so the last self-attn/
        # MLP stage can propagate and refine text-conditioned local structure.
        layers = self.blocks.layers
        if len(layers) == 0:
            raise RuntimeError("VideoDiT requires at least one Transformer block")
        for block in layers[:-1]:
            hidden = block(hidden)
        hidden = hidden + token_text_attention(hidden, projected_text, text_mask)
        hidden = layers[-1](hidden)
        if self.blocks.norm is not None:
            hidden = self.blocks.norm(hidden)
        tokens = self.out(self.norm(hidden))

        tokens = tokens.view(batch, grid[0], grid[1], grid[2], channels, pt, ph, pw)
        tokens = tokens.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return cast(Tensor, tokens.view(batch, channels, frames, height, width))
