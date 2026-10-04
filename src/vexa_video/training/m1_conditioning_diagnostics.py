from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor

from vexa_video.models.dit import (
    VideoDiT,
    importance_weighted_text_pool,
    spatiotemporal_position_embedding,
    timestep_embedding,
)


@dataclass(frozen=True, slots=True)
class AttentionTrace:
    output: Tensor
    normalized_entropy: Tensor
    shape_token_mass: Tensor
    valid_logit_std: Tensor


@dataclass(frozen=True, slots=True)
class DiTConditioningTrace:
    pooled: Tensor
    first_attention: Tensor
    final_attention: Tensor
    first_entropy: Tensor
    first_shape_mass: Tensor
    first_logit_std: Tensor
    final_entropy: Tensor
    final_shape_mass: Tensor
    final_logit_std: Tensor


def _attention_trace(
    video_tokens: Tensor,
    text_tokens: Tensor,
    text_mask: Tensor,
    shape_token_mask: Tensor | None,
) -> AttentionTrace:
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
    if shape_token_mask is not None and shape_token_mask.shape != text_mask.shape:
        raise ValueError("shape_token_mask must match text_mask")

    scale = 1.0 / math.sqrt(max(video_tokens.shape[-1], 1))
    scores = torch.matmul(video_tokens, text_tokens.transpose(1, 2)) * scale
    valid = text_mask.unsqueeze(1)
    masked_scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
    weights = torch.softmax(masked_scores, dim=-1)
    attended = torch.matmul(weights, text_tokens)
    output = F.layer_norm(attended, (attended.shape[-1],))

    valid_float = valid.to(dtype=scores.dtype)
    valid_count = valid_float.sum(dim=-1).clamp_min(1.0)
    mean = (scores * valid_float).sum(dim=-1) / valid_count
    centered = (scores - mean.unsqueeze(-1)) * valid_float
    logit_std = (centered.square().sum(dim=-1) / valid_count).sqrt().mean(dim=1)

    entropy = -(weights * weights.clamp_min(torch.finfo(weights.dtype).tiny).log()).sum(dim=-1)
    entropy_denominator = text_mask.sum(dim=-1).to(dtype=entropy.dtype).clamp_min(2.0).log()
    normalized_entropy = (entropy / entropy_denominator.unsqueeze(1)).mean(dim=1)

    if shape_token_mask is None:
        shape_mass = torch.zeros(
            (video_tokens.shape[0],),
            device=video_tokens.device,
            dtype=video_tokens.dtype,
        )
    else:
        effective_shape_mask = (shape_token_mask & text_mask).to(dtype=weights.dtype)
        shape_mass = (weights * effective_shape_mask.unsqueeze(1)).sum(dim=-1).mean(dim=1)

    return AttentionTrace(
        output=output,
        normalized_entropy=normalized_entropy,
        shape_token_mass=shape_mass,
        valid_logit_std=logit_std,
    )


def diagnostic_dit_forward(
    dit: VideoDiT,
    latents: Tensor,
    timesteps: Tensor,
    text_tokens: Tensor,
    text_mask: Tensor,
    *,
    shape_token_mask: Tensor | None = None,
    disable_pool: bool = False,
    disable_first_attention: bool = False,
    disable_final_attention: bool = False,
) -> tuple[Tensor, DiTConditioningTrace]:
    """Mirror VideoDiT.forward while exposing/ablating only existing conditioning branches.

    With every disable flag false, this function must remain numerically equivalent to
    VideoDiT.forward. It is diagnostic-only: it does not mutate the model or register
    additional parameters.
    """
    if latents.ndim != 5:
        raise ValueError("latents must have shape [B, C, T, H, W]")
    if text_tokens.ndim != 3:
        raise ValueError("text_tokens must have shape [B, L, D]")
    batch, channels, frames, height, width = latents.shape
    if channels != dit.latent_channels:
        raise ValueError(f"expected {dit.latent_channels} latent channels, got {channels}")

    pt, ph, pw = dit.patch_size
    if frames % pt or height % ph or width % pw:
        raise ValueError("latent dimensions must be divisible by patch size")

    patches = dit.patch_embed(latents)
    grid = cast(tuple[int, int, int], tuple(patches.shape[2:]))
    tokens = patches.flatten(2).transpose(1, 2)
    positions = spatiotemporal_position_embedding(
        grid,
        tokens.shape[-1],
        device=tokens.device,
        dtype=tokens.dtype,
    )
    time_cond = dit.time_mlp(timestep_embedding(timesteps, tokens.shape[-1])).unsqueeze(1)
    projected_text = dit.text_proj(text_tokens)
    pooled = importance_weighted_text_pool(projected_text, text_mask).unsqueeze(1)

    hidden = tokens + positions + time_cond
    if not disable_pool:
        hidden = hidden + pooled

    # Production queries the first text-attention branch only after spatial,
    # timestep and pooled-text context are present.
    first = _attention_trace(hidden, projected_text, text_mask, shape_token_mask)
    if not disable_first_attention:
        hidden = hidden + first.output

    # Mirror production's final text injection before the final Transformer
    # block, leaving one learned spatial processing stage after conditioning.
    layers = dit.blocks.layers
    if len(layers) == 0:
        raise RuntimeError("VideoDiT requires at least one Transformer block")
    for block in layers[:-1]:
        hidden = block(hidden)
    final = _attention_trace(hidden, projected_text, text_mask, shape_token_mask)
    if not disable_final_attention:
        hidden = hidden + final.output
    hidden = layers[-1](hidden)
    if dit.blocks.norm is not None:
        hidden = dit.blocks.norm(hidden)
    hidden = dit.out(dit.norm(hidden))

    hidden = hidden.view(batch, grid[0], grid[1], grid[2], channels, pt, ph, pw)
    hidden = hidden.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
    prediction = cast(Tensor, hidden.view(batch, channels, frames, height, width))
    return prediction, DiTConditioningTrace(
        pooled=pooled,
        first_attention=first.output,
        final_attention=final.output,
        first_entropy=first.normalized_entropy,
        first_shape_mass=first.shape_token_mass,
        first_logit_std=first.valid_logit_std,
        final_entropy=final.normalized_entropy,
        final_shape_mass=final.shape_token_mass,
        final_logit_std=final.valid_logit_std,
    )
