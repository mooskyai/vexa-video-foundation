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


class GatedTextCrossAttention(nn.Module):
    """Learned text cross-attention with Q/K RMS normalization and a gentle residual gate."""

    def __init__(self, hidden_size: int, heads: int, *, gate_init: float = 0.05) -> None:
        super().__init__()
        if hidden_size % heads != 0:
            raise ValueError("hidden_size must be divisible by heads")
        if not 0.0 < gate_init < 1.0:
            raise ValueError("gate_init must be in (0, 1)")
        self.hidden_size = hidden_size
        self.heads = heads
        self.head_dim = hidden_size // heads
        self.query_norm = nn.LayerNorm(hidden_size)
        self.text_norm = nn.LayerNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        gate_logit = math.log(gate_init / (1.0 - gate_init))
        self.gate_logit = nn.Parameter(torch.tensor(gate_logit, dtype=torch.float32))

    def _split_heads(self, tensor: Tensor) -> Tensor:
        batch, tokens, _ = tensor.shape
        return tensor.view(batch, tokens, self.heads, self.head_dim).transpose(1, 2)

    @staticmethod
    def _rms_unit(tensor: Tensor, eps: float = 1e-6) -> Tensor:
        scale = torch.rsqrt(tensor.square().mean(dim=-1, keepdim=True) + eps)
        return tensor * scale

    def normalized_qk(self, video_tokens: Tensor, text_tokens: Tensor) -> tuple[Tensor, Tensor]:
        query = self._split_heads(self.q_proj(self.query_norm(video_tokens)))
        key = self._split_heads(self.k_proj(self.text_norm(text_tokens)))
        return self._rms_unit(query), self._rms_unit(key)

    def forward(self, video_tokens: Tensor, text_tokens: Tensor, text_mask: Tensor) -> Tensor:
        if video_tokens.ndim != 3 or text_tokens.ndim != 3:
            raise ValueError("video_tokens and text_tokens must be rank-3 tensors")
        if text_mask.shape != text_tokens.shape[:2]:
            raise ValueError("text_mask must have shape [B, L]")
        if video_tokens.shape[0] != text_tokens.shape[0]:
            raise ValueError("video/text batch sizes must match")
        if video_tokens.shape[-1] != self.hidden_size or text_tokens.shape[-1] != self.hidden_size:
            raise ValueError("cross-attention feature dimensions must match hidden_size")
        if not bool(text_mask.any(dim=1).all().item()):
            raise ValueError("every text sequence must contain at least one unmasked token")

        query, key = self.normalized_qk(video_tokens, text_tokens)
        value = self._split_heads(self.v_proj(self.text_norm(text_tokens)))
        allowed = text_mask[:, None, None, :]
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=allowed,
            dropout_p=0.0,
            is_causal=False,
        )
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(video_tokens.shape[0], video_tokens.shape[1], self.hidden_size)
        )
        gate = torch.sigmoid(self.gate_logit).to(dtype=attended.dtype)
        return cast(Tensor, gate * self.out_proj(attended))


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
        self.text_cross_attention = nn.ModuleList(
            GatedTextCrossAttention(hidden_size, heads) for _ in range(layers)
        )
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

        # Keep pooled text as the global control path for color/direction, then use
        # learned token-level cross-attention before every spatial Transformer block.
        # Q/K RMS normalization keeps logits controlled while SDPA can dispatch to
        # fused CUDA kernels; the small learned gate prevents a random local branch
        # from overwhelming the global path at the start of diffusion training.
        hidden = tokens + positions + time_cond + text_cond
        layers = self.blocks.layers
        if len(layers) == 0:
            raise RuntimeError("VideoDiT requires at least one Transformer block")
        if len(layers) != len(self.text_cross_attention):
            raise RuntimeError("Transformer/cross-attention layer counts must match")
        for block, cross_attention in zip(layers, self.text_cross_attention, strict=True):
            hidden = hidden + cross_attention(hidden, projected_text, text_mask)
            hidden = block(hidden)
        if self.blocks.norm is not None:
            hidden = self.blocks.norm(hidden)
        tokens = self.out(self.norm(hidden))

        tokens = tokens.view(batch, grid[0], grid[1], grid[2], channels, pt, ph, pw)
        tokens = tokens.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return cast(Tensor, tokens.view(batch, channels, frames, height, width))
