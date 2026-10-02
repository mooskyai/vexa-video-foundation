from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor

from vexa_video.config import ProjectConfig
from vexa_video.data import (
    COLORS,
    DIRECTIONS,
    SHAPES,
    StageBSyntheticDataset,
    SyntheticControl,
    render_motion_sample,
)
from vexa_video.data.controlled_motion import controlled_motion_shape_size
from vexa_video.diffusion import LinearNoiseSchedule
from vexa_video.inference.sampler import guided_ddim_rollout, sample_video
from vexa_video.models import ByteTokenizer, TinyVideoVAE, TransformerTextEncoder, VideoDiT
from vexa_video.training.checkpoint import build_training_checkpoint
from vexa_video.training.m1_research import (
    M1ResearchSettings,
    causal_latent_delta_loss,
    constraint_rank_score,
    gradient_cosine_matrix,
    latent_svd_diagnostics,
    latent_target_ranking_loss,
    merge_primary_and_protected_gradients,
    min_snr_epsilon_weights,
    sinkhorn_causal_gate,
    sinkhorn_shape_geometry_loss,
)


@dataclass(slots=True)
class StageBComponents:
    tokenizer: ByteTokenizer
    text_encoder: TransformerTextEncoder
    vae: TinyVideoVAE
    dit: VideoDiT
    schedule: LinearNoiseSchedule


@dataclass(frozen=True, slots=True)
class StageBStepMetrics:
    reconstruction_loss: float
    diffusion_loss: float
    total_loss: float
    gradient_norm: float
    unconditional_loss: float = 0.0
    counterfactual_loss: float = 0.0
    prompt_contrast_loss: float = 0.0
    color_prompt_gap: float = 0.0
    direction_prompt_gap: float = 0.0
    shape_prompt_gap: float = 0.0
    semantic_direction_loss: float = 0.0
    semantic_color_loss: float = 0.0
    semantic_shape_loss: float = 0.0
    full_rollout_direction_loss: float = 0.0
    full_rollout_color_loss: float = 0.0
    full_rollout_shape_loss: float = 0.0
    causal_shape_loss: float = 0.0
    latent_shape_rank_loss: float = 0.0
    latent_shape_rank_margin: float = 0.0
    sinkhorn_shape_loss: float = 0.0
    sinkhorn_shape_gate: float = 0.0
    latent_shape_delta_cosine: float = 0.0
    latent_shape_delta_magnitude_ratio: float = 0.0
    sinkhorn_shape_margin: float = 0.0
    shape_gradient_cosine: float = 0.0
    shape_gradient_projection_active: float = 0.0
    grad_cos_diffusion_direction: float = 0.0
    grad_cos_diffusion_color: float = 0.0
    grad_cos_diffusion_shape: float = 0.0
    grad_cos_direction_color: float = 0.0
    grad_cos_direction_shape: float = 0.0
    grad_cos_color_shape: float = 0.0


@dataclass(frozen=True, slots=True)
class StageBTrainResult:
    latest_checkpoint: Path
    best_checkpoint: Path | None
    diffusion_step: int
    vae_validation_reconstruction_loss: float
    best_validation_score: float


_COLOR_TARGETS: dict[str, tuple[float, float, float]] = {
    "red": (1.0, -0.65, -0.65),
    "green": (-0.65, 1.0, -0.65),
    "blue": (-0.65, -0.65, 1.0),
    "yellow": (1.0, 1.0, -0.65),
}
_DIRECTION_TARGETS: dict[str, tuple[float, float]] = {
    "right": (1.0, 0.0),
    "left": (-1.0, 0.0),
    "down": (0.0, 1.0),
    "up": (0.0, -1.0),
}


def build_stage_b_components(cfg: ProjectConfig, *, device: torch.device) -> StageBComponents:
    tokenizer = ByteTokenizer()
    text_encoder = TransformerTextEncoder(
        vocab_size=cfg.text.vocab_size,
        max_length=cfg.text.max_length,
        d_model=cfg.text.d_model,
        layers=cfg.text.layers,
        heads=cfg.text.heads,
        ff_mult=cfg.text.ff_mult,
    ).to(device)
    vae = TinyVideoVAE(
        in_channels=cfg.vae.in_channels,
        latent_channels=cfg.vae.latent_channels,
        base_channels=cfg.vae.base_channels,
        spatial_downsample=cfg.vae.spatial_downsample,
        temporal_downsample=cfg.vae.temporal_downsample,
    ).to(device)
    dit = VideoDiT(
        latent_channels=cfg.vae.latent_channels,
        text_dim=cfg.text.d_model,
        hidden_size=cfg.dit.hidden_size,
        layers=cfg.dit.layers,
        heads=cfg.dit.heads,
        patch_size=(cfg.dit.patch_t, cfg.dit.patch_h, cfg.dit.patch_w),
    ).to(device)
    schedule = LinearNoiseSchedule(
        cfg.diffusion.timesteps,
        cfg.diffusion.beta_start,
        cfg.diffusion.beta_end,
    )
    return StageBComponents(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        dit=dit,
        schedule=schedule,
    )


def reconstruction_loss(
    reconstruction: Tensor,
    target: Tensor,
    *,
    silhouette_weight: float = 0.0,
) -> Tensor:
    """Pixel/color reconstruction plus balanced foreground-silhouette preservation."""
    if reconstruction.shape != target.shape:
        raise ValueError("reconstruction and target must have identical shapes")
    if silhouette_weight < 0:
        raise ValueError("silhouette_weight must be non-negative")
    pixel_loss = F.mse_loss(reconstruction, target)
    foreground = target.amax(dim=1, keepdim=True).gt(-0.9)
    expanded = foreground.expand_as(target).to(dtype=target.dtype)
    foreground_count = expanded.sum()
    foreground_loss = pixel_loss.new_zeros(())
    if float(foreground_count.item()) > 0.0:
        foreground_loss = ((reconstruction - target).square() * expanded).sum() / foreground_count

    target_silhouette = foreground.to(dtype=target.dtype)
    predicted_silhouette = ((reconstruction + 1.0) * 0.5).clamp(0.0, 1.0).amax(dim=1, keepdim=True)
    silhouette_error = (predicted_silhouette - target_silhouette).square()
    background_silhouette = 1.0 - target_silhouette
    silhouette_loss = pixel_loss.new_zeros(())
    target_count = target_silhouette.sum()
    background_count = background_silhouette.sum()
    if float(target_count.item()) > 0.0:
        silhouette_loss = (
            silhouette_loss + (silhouette_error * target_silhouette).sum() / target_count
        )
    if float(background_count.item()) > 0.0:
        silhouette_loss = (
            silhouette_loss + (silhouette_error * background_silhouette).sum() / background_count
        )
    return pixel_loss + foreground_loss + silhouette_weight * silhouette_loss


def _sample_batch(
    dataset: StageBSyntheticDataset,
    indices: list[int],
    *,
    device: torch.device,
) -> tuple[Tensor, list[str]]:
    samples = [dataset.sample(index) for index in indices]
    videos = torch.stack([sample.video for sample in samples]).to(device)
    captions = [sample.caption for sample in samples]
    return videos, captions


def _sample_shape_counterfactual_batch(
    dataset: StageBSyntheticDataset,
    indices: list[int],
    *,
    device: torch.device,
) -> tuple[Tensor, list[str]]:
    paired_videos: list[Tensor] = []
    paired_captions: list[str] = []
    for index in indices:
        sample = dataset.sample(index)
        paired_shape = SHAPES[(SHAPES.index(sample.control.shape) + 1) % len(SHAPES)]
        paired_control = SyntheticControl(
            direction=sample.control.direction,
            color=sample.control.color,
            shape=paired_shape,
            speed_bucket=sample.control.speed_bucket,
        )
        paired = render_motion_sample(
            frames=dataset.frames,
            size=dataset.size,
            seed=sample.seed,
            control=paired_control,
            shape_size=controlled_motion_shape_size(dataset.size),
        )
        paired_videos.append(paired.video)
        paired_captions.append(paired.caption)
    return torch.stack(paired_videos).to(device), paired_captions


def _random_indices(
    *,
    dataset_size: int,
    batch_size: int,
    generator: torch.Generator,
) -> list[int]:
    values = torch.randint(0, dataset_size, (batch_size,), generator=generator)
    return [int(value.item()) for value in values]


def _counterfactual_caption(caption: str, vocabulary: tuple[str, ...]) -> str:
    for index, word in enumerate(vocabulary):
        marker = f" {word} "
        if marker in caption:
            replacement = vocabulary[(index + 1) % len(vocabulary)]
            return caption.replace(marker, f" {replacement} ", 1)
    raise ValueError(f"caption does not contain any expected term from {vocabulary}: {caption}")


def _caption_with_direction(caption: str, direction: str) -> str:
    if direction not in DIRECTIONS:
        raise ValueError(f"unknown direction: {direction}")
    for word in DIRECTIONS:
        marker = f" {word} "
        if marker in caption:
            return caption.replace(marker, f" {direction} ", 1)
    raise ValueError(f"caption does not contain a direction term: {caption}")


def _caption_with_control(caption: str, value: str, vocabulary: tuple[str, ...]) -> str:
    if value not in vocabulary:
        raise ValueError(f"unknown control value: {value}")
    for word in vocabulary:
        marker = f" {word} "
        if marker in caption:
            return caption.replace(marker, f" {value} ", 1)
    raise ValueError(f"caption does not contain a control term from {vocabulary}: {caption}")


def _soft_shape_template_loss(video: Tensor, captions: list[str], size: int) -> Tensor:
    """Match decoded foreground geometry to the caption's square/circle template."""
    if video.ndim != 5 or video.shape[1] != 3:
        raise ValueError("video must have shape [B, 3, T, H, W]")
    if video.shape[0] != len(captions):
        raise ValueError("video/caption counts must match")

    signal = ((video + 1.0) * 0.5).clamp(0.0, 1.0).amax(dim=1)
    weights = signal.square()
    _, frames, height, width = signal.shape
    y_grid, x_grid = torch.meshgrid(
        torch.arange(height, device=video.device, dtype=video.dtype),
        torch.arange(width, device=video.device, dtype=video.dtype),
        indexing="ij",
    )
    mass = weights.sum(dim=(2, 3)).clamp_min(1e-6)
    center_x = (weights * x_grid.view(1, 1, height, width)).sum(dim=(2, 3)) / mass
    center_y = (weights * y_grid.view(1, 1, height, width)).sum(dim=(2, 3)) / mass

    shape_size = controlled_motion_shape_size(size)
    half_extent = (shape_size - 1) / 2.0
    center_x = (torch.floor(center_x.detach()) + 0.5).clamp(half_extent, (width - 1) - half_extent)
    center_y = (torch.floor(center_y.detach()) + 0.5).clamp(half_extent, (height - 1) - half_extent)
    dx = x_grid.view(1, 1, height, width) - center_x[:, :, None, None]
    dy = y_grid.view(1, 1, height, width) - center_y[:, :, None, None]
    square_template = ((dx.abs() <= half_extent) & (dy.abs() <= half_extent)).to(dtype=video.dtype)
    radius = max(1.0, shape_size / 2.0)
    circle_template = (dx.square() + dy.square() <= radius**2).to(dtype=video.dtype)

    square_selector: list[float] = []
    for caption in captions:
        if " square " in caption:
            square_selector.append(1.0)
        elif " circle " in caption:
            square_selector.append(0.0)
        else:
            raise ValueError(f"caption does not contain a shape term: {caption}")
    selector = torch.tensor(square_selector, device=video.device, dtype=video.dtype).view(
        -1, 1, 1, 1
    )
    target = selector * square_template + (1.0 - selector) * circle_template
    target_area = target.sum(dim=(2, 3)).clamp_min(1.0)
    frame_loss = (signal - target).square().sum(dim=(2, 3)) / target_area
    if frame_loss.shape != (video.shape[0], frames):
        raise RuntimeError("unexpected shape-template loss dimensions")
    return frame_loss.mean()


def _shape_square_selector(
    captions: list[str], *, device: torch.device, dtype: torch.dtype
) -> Tensor:
    values: list[float] = []
    for caption in captions:
        if " square " in caption:
            values.append(1.0)
        elif " circle " in caption:
            values.append(0.0)
        else:
            raise ValueError(f"caption does not contain a shape term: {caption}")
    return torch.tensor(values, device=device, dtype=dtype)


def _soft_shape_corner_occupancy(video: Tensor, size: int) -> Tensor:
    """Return per-frame occupancy of pixels present only in the square template."""
    if video.ndim != 5 or video.shape[1] != 3:
        raise ValueError("video must have shape [B, 3, T, H, W]")

    signal = ((video + 1.0) * 0.5).clamp(0.0, 1.0).amax(dim=1)
    weights = signal.square()
    _, _, height, width = signal.shape
    y_grid, x_grid = torch.meshgrid(
        torch.arange(height, device=video.device, dtype=video.dtype),
        torch.arange(width, device=video.device, dtype=video.dtype),
        indexing="ij",
    )
    mass = weights.sum(dim=(2, 3)).clamp_min(1e-6)
    center_x = (weights * x_grid.view(1, 1, height, width)).sum(dim=(2, 3)) / mass
    center_y = (weights * y_grid.view(1, 1, height, width)).sum(dim=(2, 3)) / mass

    shape_size = controlled_motion_shape_size(size)
    half_extent = (shape_size - 1) / 2.0
    center_x = (torch.floor(center_x.detach()) + 0.5).clamp(half_extent, (width - 1) - half_extent)
    center_y = (torch.floor(center_y.detach()) + 0.5).clamp(half_extent, (height - 1) - half_extent)
    dx = x_grid.view(1, 1, height, width) - center_x[:, :, None, None]
    dy = y_grid.view(1, 1, height, width) - center_y[:, :, None, None]
    square_template = (dx.abs() <= half_extent) & (dy.abs() <= half_extent)
    radius = max(1.0, shape_size / 2.0)
    circle_template = dx.square() + dy.square() <= radius**2
    corner_mask = (square_template & ~circle_template).to(dtype=video.dtype)
    corner_count = corner_mask.sum(dim=(2, 3)).clamp_min(1.0)
    return (signal * corner_mask).sum(dim=(2, 3)) / corner_count


def _soft_shape_corner_loss(video: Tensor, captions: list[str], size: int) -> Tensor:
    """Supervise only the pixels that distinguish square from circle."""
    if video.shape[0] != len(captions):
        raise ValueError("video/caption counts must match")
    occupancy = _soft_shape_corner_occupancy(video, size)
    square_target = _shape_square_selector(
        captions, device=video.device, dtype=video.dtype
    ).unsqueeze(1)
    return F.mse_loss(occupancy, square_target.expand_as(occupancy))


def _soft_shape_pair_margin_loss(
    first_video: Tensor,
    first_captions: list[str],
    second_video: Tensor,
    second_captions: list[str],
    size: int,
    *,
    margin: float = 0.5,
) -> Tensor:
    """Force same-noise square/circle outputs to separate in corner occupancy."""
    if first_video.shape != second_video.shape:
        raise ValueError("paired shape videos must have identical shapes")
    if first_video.shape[0] != len(first_captions) or second_video.shape[0] != len(second_captions):
        raise ValueError("paired video/caption counts must match")
    if margin <= 0.0:
        raise ValueError("shape pair margin must be positive")

    first_target = _shape_square_selector(
        first_captions, device=first_video.device, dtype=first_video.dtype
    )
    second_target = _shape_square_selector(
        second_captions, device=second_video.device, dtype=second_video.dtype
    )
    orientation = first_target - second_target
    if not bool(orientation.abs().eq(1.0).all().item()):
        raise ValueError("paired captions must contain opposite square/circle targets")

    first_occupancy = _soft_shape_corner_occupancy(first_video, size).mean(dim=1)
    second_occupancy = _soft_shape_corner_occupancy(second_video, size).mean(dim=1)
    separation = orientation * (first_occupancy - second_occupancy)
    return F.relu(margin - separation).mean()


def _shape_fill_targets(
    captions: list[str],
    size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Return evaluator-aligned square/circle fill-ratio targets."""
    shape_size = controlled_motion_shape_size(size)
    coords = torch.arange(shape_size, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    center = (shape_size - 1) / 2.0
    radius = max(1.0, shape_size / 2.0)
    circle_area = ((xx - center).square() + (yy - center).square() <= radius**2).sum()
    circle_fill = circle_area.to(dtype=dtype) / float(shape_size * shape_size)

    values: list[Tensor] = []
    for caption in captions:
        if " square " in caption:
            values.append(torch.ones((), device=device, dtype=dtype))
        elif " circle " in caption:
            values.append(circle_fill)
        else:
            raise ValueError(f"caption does not contain a shape term: {caption}")
    return torch.stack(values)


def _soft_evaluator_shape_fill(video: Tensor, size: int) -> Tensor:
    """Approximate the frozen evaluator's fill ratio on the expected object box."""
    if video.ndim != 5 or video.shape[1] != 3:
        raise ValueError("video must have shape [B, 3, T, H, W]")

    strength = (video + 1.0).abs().mean(dim=1)
    _, _, height, width = strength.shape
    y_grid, x_grid = torch.meshgrid(
        torch.arange(height, device=video.device, dtype=video.dtype),
        torch.arange(width, device=video.device, dtype=video.dtype),
        indexing="ij",
    )
    center_weights = strength.square()
    mass = center_weights.sum(dim=(2, 3)).clamp_min(1e-6)
    center_x = (center_weights * x_grid.view(1, 1, height, width)).sum(dim=(2, 3)) / mass
    center_y = (center_weights * y_grid.view(1, 1, height, width)).sum(dim=(2, 3)) / mass

    shape_size = controlled_motion_shape_size(size)
    half_extent = (shape_size - 1) / 2.0
    center_x = (torch.floor(center_x.detach()) + 0.5).clamp(half_extent, (width - 1) - half_extent)
    center_y = (torch.floor(center_y.detach()) + 0.5).clamp(half_extent, (height - 1) - half_extent)
    dx = x_grid.view(1, 1, height, width) - center_x[:, :, None, None]
    dy = y_grid.view(1, 1, height, width) - center_y[:, :, None, None]
    expected_box = ((dx.abs() <= half_extent) & (dy.abs() <= half_extent)).to(dtype=video.dtype)

    maximum = strength.amax(dim=(2, 3), keepdim=True)
    threshold = torch.maximum(torch.full_like(maximum, 0.10), maximum * 0.25)
    scale = (maximum - threshold).clamp_min(1e-4)
    occupancy = ((strength - threshold) / scale).clamp(0.0, 1.0)
    box_area = expected_box.sum(dim=(2, 3)).clamp_min(1.0)
    return (occupancy * expected_box).sum(dim=(2, 3)) / box_area


def _soft_evaluator_shape_fill_loss(video: Tensor, captions: list[str], size: int) -> Tensor:
    """Match square/circle geometry using the evaluator's fill-ratio targets."""
    if video.shape[0] != len(captions):
        raise ValueError("video/caption counts must match")
    fill = _soft_evaluator_shape_fill(video, size)
    targets = _shape_fill_targets(captions, size, device=video.device, dtype=video.dtype).unsqueeze(
        1
    )
    return F.mse_loss(fill, targets.expand_as(fill))


def _soft_evaluator_shape_pair_margin_loss(
    first_video: Tensor,
    first_captions: list[str],
    second_video: Tensor,
    second_captions: list[str],
    size: int,
) -> Tensor:
    """Require same-noise shape pairs to span the evaluator's target fill gap."""
    if first_video.shape != second_video.shape:
        raise ValueError("paired shape videos must have identical shapes")
    if first_video.shape[0] != len(first_captions) or second_video.shape[0] != len(second_captions):
        raise ValueError("paired video/caption counts must match")

    first_targets = _shape_fill_targets(
        first_captions, size, device=first_video.device, dtype=first_video.dtype
    )
    second_targets = _shape_fill_targets(
        second_captions, size, device=second_video.device, dtype=second_video.dtype
    )
    target_gap = first_targets - second_targets
    if not bool(target_gap.abs().gt(0.0).all().item()):
        raise ValueError("paired captions must contain opposite square/circle targets")

    first_fill = _soft_evaluator_shape_fill(first_video, size).mean(dim=1)
    second_fill = _soft_evaluator_shape_fill(second_video, size).mean(dim=1)
    separation = target_gap.sign() * (first_fill - second_fill)
    normalized_separation = separation / target_gap.abs().clamp_min(1e-6)
    return F.relu(1.0 - normalized_separation).mean()


def _conditioning_predictions(
    *,
    components: StageBComponents,
    noisy_latents: Tensor,
    timesteps: Tensor,
    captions: list[str],
    cfg: ProjectConfig,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    color_counterfactuals = [_counterfactual_caption(caption, COLORS) for caption in captions]
    direction_counterfactuals = [
        _counterfactual_caption(caption, DIRECTIONS) for caption in captions
    ]
    shape_counterfactuals = [_counterfactual_caption(caption, SHAPES) for caption in captions]
    unconditional = [""] * len(captions)
    all_captions = [
        *captions,
        *color_counterfactuals,
        *direction_counterfactuals,
        *shape_counterfactuals,
        *unconditional,
    ]
    tokens = components.tokenizer.batch(all_captions, cfg.text.max_length, device=device)
    text_tokens = components.text_encoder(tokens.input_ids, tokens.attention_mask)
    predicted = components.dit(
        torch.cat((noisy_latents,) * 5, dim=0),
        timesteps.repeat(5),
        text_tokens,
        tokens.attention_mask,
    )
    return cast(tuple[Tensor, Tensor, Tensor, Tensor, Tensor], predicted.chunk(5, dim=0))


def _per_sample_mse(prediction: Tensor, target: Tensor) -> Tensor:
    return (prediction - target).square().flatten(start_dim=1).mean(dim=1)


def _sample_training_timesteps(
    *,
    batch_size: int,
    timesteps: int,
    high_noise_fraction: float,
    device: torch.device,
) -> Tensor:
    sampled = torch.randint(0, timesteps, (batch_size,), device=device)
    if high_noise_fraction == 0.0:
        return sampled
    high_noise = torch.randint(timesteps // 2, timesteps, (batch_size,), device=device)
    choose_high = torch.rand(batch_size, device=device) < high_noise_fraction
    return torch.where(choose_high, high_noise, sampled)


def _caption_target(
    caption: str,
    targets: Mapping[str, tuple[float, ...]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    for word, values in targets.items():
        if f" {word} " in caption:
            return torch.tensor(values, device=device, dtype=dtype)
    raise ValueError(f"caption does not contain an expected semantic target: {caption}")


def _soft_video_features(video: Tensor) -> tuple[Tensor, Tensor]:
    if video.ndim != 5 or video.shape[1] != 3:
        raise ValueError("video must have shape [B, 3, T, H, W]")
    signal = ((video + 1.0) * 0.5).clamp(0.0, 1.0)
    weights = signal.amax(dim=1).square()
    _, _, frames, height, width = video.shape
    x = torch.linspace(-1.0, 1.0, width, device=video.device, dtype=video.dtype)
    y = torch.linspace(-1.0, 1.0, height, device=video.device, dtype=video.dtype)
    mass = weights.sum(dim=(2, 3)).clamp_min(1e-6)
    center_x = (weights * x.view(1, 1, 1, width)).sum(dim=(2, 3)) / mass
    center_y = (weights * y.view(1, 1, height, 1)).sum(dim=(2, 3)) / mass
    motion = torch.stack(
        (center_x[:, frames - 1] - center_x[:, 0], center_y[:, frames - 1] - center_y[:, 0]),
        dim=-1,
    )
    pixel_weights = weights.unsqueeze(1)
    color_mass = pixel_weights.sum(dim=(2, 3, 4)).clamp_min(1e-6)
    mean_color = (video * pixel_weights).sum(dim=(2, 3, 4)) / color_mass
    return motion, mean_color


def _soft_shape_score(video: Tensor) -> Tensor:
    """Differentiable corner-occupancy score: square ~= 1, circle < 1."""
    if video.ndim != 5 or video.shape[1] != 3:
        raise ValueError("video must have shape [B, 3, T, H, W]")
    weights = ((video + 1.0) * 0.5).clamp(0.0, 1.0).amax(dim=1).square()
    _, _, height, width = weights.shape
    x = torch.linspace(-1.0, 1.0, width, device=video.device, dtype=video.dtype)
    y = torch.linspace(-1.0, 1.0, height, device=video.device, dtype=video.dtype)
    y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")
    mass = weights.sum(dim=(2, 3)).clamp_min(1e-6)
    center_x = (weights * x_grid.view(1, 1, height, width)).sum(dim=(2, 3)) / mass
    center_y = (weights * y_grid.view(1, 1, height, width)).sum(dim=(2, 3)) / mass
    dx = x_grid.view(1, 1, height, width) - center_x[:, :, None, None]
    dy = y_grid.view(1, 1, height, width) - center_y[:, :, None, None]
    variance_x = (weights * dx.square()).sum(dim=(2, 3)) / mass
    variance_y = (weights * dy.square()).sum(dim=(2, 3)) / mass
    corner_moment = (weights * dx.square() * dy.square()).sum(dim=(2, 3)) / mass
    score = corner_moment / (variance_x * variance_y).clamp_min(1e-6)
    return score.mean(dim=1)


def _shape_template_target(
    shape: str,
    size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    shape_size = controlled_motion_shape_size(size)
    coords = torch.arange(shape_size, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    if shape == "square":
        mask = torch.ones((shape_size, shape_size), device=device, dtype=dtype)
    elif shape == "circle":
        center = (shape_size - 1) / 2.0
        radius = max(1.0, shape_size / 2.0)
        mask = (((xx - center).square() + (yy - center).square()) <= radius**2).to(dtype)
    else:
        raise ValueError(f"unknown shape: {shape}")
    weights = mask
    mass = weights.sum().clamp_min(1e-6)
    x = torch.linspace(-1.0, 1.0, shape_size, device=device, dtype=dtype)
    y = torch.linspace(-1.0, 1.0, shape_size, device=device, dtype=dtype)
    y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")
    center_x = (weights * x_grid).sum() / mass
    center_y = (weights * y_grid).sum() / mass
    dx = x_grid - center_x
    dy = y_grid - center_y
    variance_x = (weights * dx.square()).sum() / mass
    variance_y = (weights * dy.square()).sum() / mass
    corner_moment = (weights * dx.square() * dy.square()).sum() / mass
    return corner_moment / (variance_x * variance_y).clamp_min(1e-6)


def _shape_target(
    caption: str,
    size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    for shape in SHAPES:
        if f" {shape} " in caption:
            return _shape_template_target(shape, size, device=device, dtype=dtype)
    raise ValueError(f"caption does not contain a shape term: {caption}")


def _expected_foreground_area(
    caption: str,
    size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    shape_size = controlled_motion_shape_size(size)
    for shape in SHAPES:
        if f" {shape} " not in caption:
            continue
        if shape == "square":
            area = float(shape_size * shape_size)
        else:
            coords = torch.arange(shape_size, device=device, dtype=dtype)
            yy, xx = torch.meshgrid(coords, coords, indexing="ij")
            center = (shape_size - 1) / 2.0
            radius = max(1.0, shape_size / 2.0)
            area = float(
                (((xx - center).square() + (yy - center).square()) <= radius**2).sum().item()
            )
        return torch.tensor(area, device=device, dtype=dtype)
    raise ValueError(f"caption does not contain a shape term: {caption}")


def _soft_foreground_area_loss(video: Tensor, captions: list[str], size: int) -> Tensor:
    """Penalize foreground expansion and disappearance on every generated frame."""
    if video.ndim != 5 or video.shape[1] != 3:
        raise ValueError("video must have shape [B, 3, T, H, W]")
    if video.shape[0] != len(captions):
        raise ValueError("video/caption counts must match")
    signal = ((video + 1.0) * 0.5).clamp(0.0, 1.0).amax(dim=1)
    soft_area = signal.sum(dim=(2, 3)).clamp_min(1e-4)
    expected_area = torch.stack(
        [
            _expected_foreground_area(
                caption,
                size,
                device=video.device,
                dtype=video.dtype,
            )
            for caption in captions
        ]
    ).unsqueeze(1)
    log_area_ratio = torch.log(soft_area / expected_area)
    return 0.25 * F.smooth_l1_loss(
        log_area_ratio,
        torch.zeros_like(log_area_ratio),
        beta=0.5,
    )


def _motion_target_loss(motion: Tensor, target: Tensor, minimum_motion: float) -> Tensor:
    alignment = 1.0 - F.cosine_similarity(motion, target, dim=-1, eps=1e-6)
    magnitude = motion.norm(dim=-1)
    return (alignment + F.relu(minimum_motion - magnitude)).mean()


def _sampler_aligned_losses(
    *,
    components: StageBComponents,
    clean_latents: Tensor,
    captions: list[str],
    cfg: ProjectConfig,
    device: torch.device,
    initial_noise: Tensor,
    sampling_steps: int | None = None,
    minimum_motion: float | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    batch = min(cfg.m1.rollout_batch_size, clean_latents.shape[0], len(captions))
    if batch <= 0:
        zero = clean_latents.new_zeros(())
        return zero, zero, zero

    base_captions = captions[:batch]
    direction_captions = [
        _caption_with_direction(caption, direction)
        for caption in base_captions
        for direction in DIRECTIONS
    ]
    color_captions = [
        _caption_with_control(caption, color, COLORS)
        for caption in base_captions
        for color in COLORS
    ]
    shape_captions = [
        _caption_with_control(caption, shape, SHAPES)
        for caption in base_captions
        for shape in SHAPES
    ]
    rollout_captions = [*direction_captions, *color_captions, *shape_captions]
    conditional_tokens = components.tokenizer.batch(
        rollout_captions,
        cfg.text.max_length,
        device=device,
    )
    unconditional_tokens = components.tokenizer.batch(
        [""] * len(rollout_captions),
        cfg.text.max_length,
        device=device,
    )
    conditional_text = components.text_encoder(
        conditional_tokens.input_ids,
        conditional_tokens.attention_mask,
    )
    unconditional_text = components.text_encoder(
        unconditional_tokens.input_ids,
        unconditional_tokens.attention_mask,
    )
    base_noise = initial_noise[:batch]
    rollout_noise = torch.cat(
        (
            base_noise.repeat_interleave(len(DIRECTIONS), dim=0),
            base_noise.repeat_interleave(len(COLORS), dim=0),
            base_noise.repeat_interleave(len(SHAPES), dim=0),
        ),
        dim=0,
    )
    steps = sampling_steps if sampling_steps is not None else cfg.m1.rollout_steps
    rollout_latents = guided_ddim_rollout(
        latents=rollout_noise,
        dit=components.dit,
        schedule=components.schedule,
        conditional_text=conditional_text,
        conditional_mask=conditional_tokens.attention_mask,
        unconditional_text=unconditional_text,
        unconditional_mask=unconditional_tokens.attention_mask,
        sampling_timesteps=components.schedule.sampling_timesteps(
            steps,
            device=device,
        ),
        guidance_scale=cfg.m1.guidance_scale,
    )
    rollout_video = components.vae.decode(rollout_latents)
    direction_count = batch * len(DIRECTIONS)
    color_count = batch * len(COLORS)
    direction_video = rollout_video[:direction_count]
    color_video = rollout_video[direction_count : direction_count + color_count]
    shape_video = rollout_video[direction_count + color_count :]

    rollout_motion, _ = _soft_video_features(direction_video)
    _, rollout_color = _soft_video_features(color_video)
    direction_targets = torch.tensor(
        [_DIRECTION_TARGETS[direction] for _ in base_captions for direction in DIRECTIONS],
        device=device,
        dtype=clean_latents.dtype,
    )
    color_targets = torch.stack(
        [
            _caption_target(caption, _COLOR_TARGETS, device=device, dtype=clean_latents.dtype)
            for caption in color_captions
        ]
    )
    motion_floor = cfg.m1.semantic_min_motion if minimum_motion is None else minimum_motion
    direction_loss = _motion_target_loss(
        rollout_motion,
        direction_targets,
        motion_floor,
    )
    color_loss = F.mse_loss(rollout_color, color_targets)
    shape_loss = _soft_evaluator_shape_fill_loss(
        shape_video,
        shape_captions,
        cfg.data.height,
    )
    paired_shape_video = shape_video.reshape(batch, len(SHAPES), *shape_video.shape[1:])
    shape_loss = shape_loss + _soft_evaluator_shape_pair_margin_loss(
        paired_shape_video[:, 0],
        shape_captions[0 :: len(SHAPES)],
        paired_shape_video[:, 1],
        shape_captions[1 :: len(SHAPES)],
        cfg.data.height,
    )
    shape_loss = shape_loss + _soft_foreground_area_loss(
        rollout_video,
        rollout_captions,
        cfg.data.height,
    )
    return direction_loss, color_loss, shape_loss


def _semantic_conditioning_losses(
    *,
    components: StageBComponents,
    clean_latents: Tensor,
    captions: list[str],
    cfg: ProjectConfig,
    device: torch.device,
    noise: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    batch = clean_latents.shape[0]
    semantic_timesteps = torch.full(
        (batch,), cfg.m1.semantic_timestep, device=device, dtype=torch.long
    )
    semantic_noise = torch.randn_like(clean_latents) if noise is None else noise
    noisy = components.schedule.add_noise(clean_latents, semantic_noise, semantic_timesteps)
    (
        conditioned,
        color_counterfactual,
        direction_counterfactual,
        shape_counterfactual,
        _,
    ) = _conditioning_predictions(
        components=components,
        noisy_latents=noisy,
        timesteps=semantic_timesteps,
        captions=captions,
        cfg=cfg,
        device=device,
    )
    color_captions = [_counterfactual_caption(caption, COLORS) for caption in captions]
    direction_captions = [_counterfactual_caption(caption, DIRECTIONS) for caption in captions]
    shape_captions = [_counterfactual_caption(caption, SHAPES) for caption in captions]
    predicted_clean = torch.cat(
        [
            components.schedule.predict_clean(noisy, conditioned, cfg.m1.semantic_timestep),
            components.schedule.predict_clean(
                noisy, color_counterfactual, cfg.m1.semantic_timestep
            ),
            components.schedule.predict_clean(
                noisy, direction_counterfactual, cfg.m1.semantic_timestep
            ),
            components.schedule.predict_clean(
                noisy, shape_counterfactual, cfg.m1.semantic_timestep
            ),
        ],
        dim=0,
    )
    decoded = components.vae.decode(predicted_clean)
    correct_video, color_video, direction_video, shape_video = decoded.chunk(4, dim=0)
    correct_motion, correct_color = _soft_video_features(correct_video)
    _, counterfactual_color = _soft_video_features(color_video)
    counterfactual_motion, _ = _soft_video_features(direction_video)

    direction_target = torch.stack(
        [
            _caption_target(caption, _DIRECTION_TARGETS, device=device, dtype=clean_latents.dtype)
            for caption in captions
        ]
    )
    direction_counterfactual_target = torch.stack(
        [
            _caption_target(caption, _DIRECTION_TARGETS, device=device, dtype=clean_latents.dtype)
            for caption in direction_captions
        ]
    )
    color_target = torch.stack(
        [
            _caption_target(caption, _COLOR_TARGETS, device=device, dtype=clean_latents.dtype)
            for caption in captions
        ]
    )
    color_counterfactual_target = torch.stack(
        [
            _caption_target(caption, _COLOR_TARGETS, device=device, dtype=clean_latents.dtype)
            for caption in color_captions
        ]
    )
    direction_loss = 0.5 * (
        _motion_target_loss(correct_motion, direction_target, cfg.m1.semantic_min_motion)
        + _motion_target_loss(
            counterfactual_motion, direction_counterfactual_target, cfg.m1.semantic_min_motion
        )
    )
    color_loss = 0.5 * (
        F.mse_loss(correct_color, color_target)
        + F.mse_loss(counterfactual_color, color_counterfactual_target)
    )
    shape_loss = 0.5 * (
        _soft_evaluator_shape_fill_loss(correct_video, captions, cfg.data.height)
        + _soft_evaluator_shape_fill_loss(shape_video, shape_captions, cfg.data.height)
    )
    shape_loss = shape_loss + _soft_evaluator_shape_pair_margin_loss(
        correct_video,
        captions,
        shape_video,
        shape_captions,
        cfg.data.height,
    )
    shape_loss = shape_loss + 0.5 * (
        _soft_foreground_area_loss(correct_video, captions, cfg.data.height)
        + _soft_foreground_area_loss(shape_video, shape_captions, cfg.data.height)
    )
    rollout_direction_loss, _, rollout_shape_loss = _sampler_aligned_losses(
        components=components,
        clean_latents=clean_latents,
        captions=captions,
        cfg=cfg,
        device=device,
        initial_noise=semantic_noise,
    )
    direction_loss = direction_loss + cfg.m1.rollout_direction_weight * rollout_direction_loss
    shape_loss = shape_loss + cfg.m1.rollout_shape_weight * rollout_shape_loss
    return direction_loss, color_loss, shape_loss


def _causal_shape_research_losses(
    *,
    components: StageBComponents,
    clean_latents: Tensor,
    paired_latents: Tensor,
    captions: list[str],
    paired_captions: list[str],
    cfg: ProjectConfig,
    device: torch.device,
    research: M1ResearchSettings,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    if clean_latents.shape != paired_latents.shape:
        raise ValueError("causal shape latent pairs must have identical shapes")
    if len(captions) != clean_latents.shape[0] or len(paired_captions) != len(captions):
        raise ValueError("causal shape caption counts must match the latent batch")

    batch = clean_latents.shape[0]
    timestep_value = cfg.m1.semantic_timestep
    timesteps = torch.full((batch,), timestep_value, device=device, dtype=torch.long)
    common_clean = 0.5 * (clean_latents + paired_latents)
    common_noise = torch.randn_like(common_clean)
    noisy = components.schedule.add_noise(common_clean, common_noise, timesteps)
    all_captions = [*captions, *paired_captions]
    tokens = components.tokenizer.batch(all_captions, cfg.text.max_length, device=device)
    text_tokens = components.text_encoder(tokens.input_ids, tokens.attention_mask)
    predicted_noise = components.dit(
        torch.cat((noisy, noisy), dim=0),
        timesteps.repeat(2),
        text_tokens,
        tokens.attention_mask,
    )
    first_noise, second_noise = predicted_noise.chunk(2, dim=0)
    predicted_first = components.schedule.predict_clean(noisy, first_noise, timestep_value)
    predicted_second = components.schedule.predict_clean(noisy, second_noise, timestep_value)
    causal_loss, delta_cosine, magnitude_ratio = causal_latent_delta_loss(
        predicted_first,
        predicted_second,
        clean_latents,
        paired_latents,
    )
    rank_loss, _, rank_margin = latent_target_ranking_loss(
        predicted_first,
        predicted_second,
        clean_latents,
        paired_latents,
        margin=research.latent_rank_margin,
    )
    sinkhorn_gate = sinkhorn_causal_gate(
        delta_cosine,
        start=research.sinkhorn_gate_start,
        full=research.sinkhorn_gate_full,
    )

    sinkhorn_loss = clean_latents.new_zeros(())
    sinkhorn_margin = clean_latents.new_zeros(())
    if float(sinkhorn_gate.item()) > 0.0 and research.sinkhorn_shape_weight > 0.0:
        decoded = components.vae.decode(torch.cat((predicted_first, predicted_second), dim=0))
        first_video, second_video = decoded.chunk(2, dim=0)
        first_sinkhorn, _, first_margin = sinkhorn_shape_geometry_loss(
            first_video,
            captions,
            size=cfg.data.height,
            blur=research.sinkhorn_blur,
            margin=research.sinkhorn_margin,
        )
        second_sinkhorn, _, second_margin = sinkhorn_shape_geometry_loss(
            second_video,
            paired_captions,
            size=cfg.data.height,
            blur=research.sinkhorn_blur,
            margin=research.sinkhorn_margin,
        )
        sinkhorn_loss = 0.5 * (first_sinkhorn + second_sinkhorn)
        sinkhorn_margin = 0.5 * (first_margin + second_margin)

    total = (
        research.causal_shape_weight * causal_loss
        + research.latent_rank_weight * rank_loss
        + research.sinkhorn_shape_weight * sinkhorn_gate * sinkhorn_loss
    )
    return (
        total,
        causal_loss,
        rank_loss,
        sinkhorn_loss,
        delta_cosine,
        magnitude_ratio,
        rank_margin,
        sinkhorn_margin,
        sinkhorn_gate,
    )


def _shape_latent_svd_metrics(
    *,
    components: StageBComponents,
    dataset: StageBSyntheticDataset,
    research: M1ResearchSettings,
    device: torch.device,
) -> dict[str, float]:
    sample_count = min(research.svd_samples, len(dataset))
    indices = list(range(sample_count))
    videos, captions = _sample_batch(dataset, indices, device=device)
    paired_videos, _ = _sample_shape_counterfactual_batch(dataset, indices, device=device)
    components.vae.eval()
    with torch.no_grad():
        first_latents = components.vae.encode(videos)
        second_latents = components.vae.encode(paired_videos)
    signs = torch.tensor(
        [1.0 if " square " in caption else -1.0 for caption in captions],
        device=device,
        dtype=first_latents.dtype,
    )
    while signs.ndim < first_latents.ndim:
        signs = signs.unsqueeze(-1)
    square_minus_circle = signs * (first_latents - second_latents)
    return latent_svd_diagnostics(square_minus_circle)


def vae_warmup_step(
    *,
    components: StageBComponents,
    optimizer: torch.optim.Optimizer,
    videos: Tensor,
    cfg: ProjectConfig,
) -> StageBStepMetrics:
    components.vae.train()
    optimizer.zero_grad(set_to_none=True)
    reconstruction, _ = components.vae(videos)
    recon_loss = reconstruction_loss(
        reconstruction,
        videos,
        silhouette_weight=cfg.m1.vae_silhouette_weight,
    )
    optimization_loss = cfg.m1.reconstruction_weight * recon_loss
    torch.autograd.backward(optimization_loss)
    grad_norm = torch.nn.utils.clip_grad_norm_(components.vae.parameters(), cfg.m1.grad_clip)
    optimizer.step()
    recon_value = float(recon_loss.detach().item())
    return StageBStepMetrics(
        reconstruction_loss=recon_value,
        diffusion_loss=0.0,
        total_loss=float(optimization_loss.detach().item()),
        gradient_norm=float(grad_norm.detach().item()),
    )


def diffusion_train_step(
    *,
    components: StageBComponents,
    optimizer: torch.optim.Optimizer,
    videos: Tensor,
    captions: list[str],
    paired_videos: Tensor | None = None,
    paired_captions: list[str] | None = None,
    cfg: ProjectConfig,
    device: torch.device,
    research: M1ResearchSettings | None = None,
    run_full_rollout: bool = False,
    diagnose_gradients: bool = False,
) -> StageBStepMetrics:
    components.text_encoder.train()
    components.dit.train()
    components.vae.eval()
    optimizer.zero_grad(set_to_none=True)
    research_settings = research or M1ResearchSettings(
        causal_shape_weight=0.0,
        sinkhorn_shape_weight=0.0,
        gradient_diagnostics=False,
    )
    research_enabled = (
        research is not None and paired_videos is not None and paired_captions is not None
    )

    with torch.no_grad():
        reconstruction, latents = components.vae(videos)
        recon_loss = reconstruction_loss(
            reconstruction,
            videos,
            silhouette_weight=cfg.m1.vae_silhouette_weight,
        )
        paired_latents: Tensor | None = None
        if research_enabled:
            assert paired_videos is not None
            paired_latents = components.vae.encode(paired_videos)
    timesteps = _sample_training_timesteps(
        batch_size=videos.shape[0],
        timesteps=components.schedule.timesteps,
        high_noise_fraction=cfg.m1.high_noise_conditioning_fraction,
        device=device,
    )
    noise = torch.randn_like(latents)
    noisy_latents = components.schedule.add_noise(latents, noise, timesteps)
    (
        conditioned,
        color_counterfactual,
        direction_counterfactual,
        shape_counterfactual,
        unconditional,
    ) = _conditioning_predictions(
        components=components,
        noisy_latents=noisy_latents,
        timesteps=timesteps,
        captions=captions,
        cfg=cfg,
        device=device,
    )
    conditioned_error = _per_sample_mse(conditioned, noise)
    color_counterfactual_error = _per_sample_mse(color_counterfactual, noise)
    direction_counterfactual_error = _per_sample_mse(direction_counterfactual, noise)
    shape_counterfactual_error = _per_sample_mse(shape_counterfactual, noise)
    unconditional_error = _per_sample_mse(unconditional, noise)
    color_gap = color_counterfactual_error - conditioned_error
    direction_gap = direction_counterfactual_error - conditioned_error
    shape_gap = shape_counterfactual_error - conditioned_error
    color_contrast_loss = F.relu(cfg.m1.prompt_contrast_margin - color_gap).mean()
    direction_contrast_loss = F.relu(cfg.m1.prompt_contrast_margin - direction_gap).mean()
    shape_contrast_loss = F.relu(cfg.m1.prompt_contrast_margin - shape_gap).mean()
    contrast_loss = (color_contrast_loss + direction_contrast_loss + shape_contrast_loss) / 3.0
    snr_weights = min_snr_epsilon_weights(
        components.schedule,
        timesteps,
        gamma=research_settings.min_snr_gamma,
        dtype=conditioned_error.dtype,
    )
    diffusion_loss = (conditioned_error * snr_weights).mean()
    unconditional_loss = (unconditional_error * snr_weights).mean()
    counterfactual_loss = (
        color_counterfactual_error.mean()
        + direction_counterfactual_error.mean()
        + shape_counterfactual_error.mean()
    ) / 3.0
    semantic_direction_loss, semantic_color_loss, semantic_shape_loss = (
        _semantic_conditioning_losses(
            components=components,
            clean_latents=latents,
            captions=captions,
            cfg=cfg,
            device=device,
        )
    )
    research_shape_loss = latents.new_zeros(())
    causal_shape_loss = latents.new_zeros(())
    latent_shape_rank_loss = latents.new_zeros(())
    latent_shape_rank_margin = latents.new_zeros(())
    sinkhorn_shape_loss = latents.new_zeros(())
    sinkhorn_shape_gate = latents.new_zeros(())
    latent_shape_delta_cosine = latents.new_zeros(())
    latent_shape_delta_magnitude_ratio = latents.new_zeros(())
    sinkhorn_shape_margin = latents.new_zeros(())
    if research_enabled:
        assert paired_latents is not None
        assert paired_captions is not None
        (
            research_shape_loss,
            causal_shape_loss,
            latent_shape_rank_loss,
            sinkhorn_shape_loss,
            latent_shape_delta_cosine,
            latent_shape_delta_magnitude_ratio,
            latent_shape_rank_margin,
            sinkhorn_shape_margin,
            sinkhorn_shape_gate,
        ) = _causal_shape_research_losses(
            components=components,
            clean_latents=latents,
            paired_latents=paired_latents,
            captions=captions,
            paired_captions=paired_captions,
            cfg=cfg,
            device=device,
            research=research_settings,
        )
    full_rollout_direction_loss = latents.new_zeros(())
    full_rollout_color_loss = latents.new_zeros(())
    full_rollout_shape_loss = latents.new_zeros(())
    if run_full_rollout:
        (
            full_rollout_direction_loss,
            full_rollout_color_loss,
            full_rollout_shape_loss,
        ) = _sampler_aligned_losses(
            components=components,
            clean_latents=latents,
            captions=captions,
            cfg=cfg,
            device=device,
            initial_noise=torch.randn_like(latents),
            sampling_steps=cfg.m1.full_rollout_steps,
            minimum_motion=cfg.m1.full_rollout_min_motion,
        )
    primary_optimization_loss = (
        cfg.m1.diffusion_weight * diffusion_loss
        + cfg.m1.unconditional_loss_weight * unconditional_loss
        + cfg.m1.prompt_contrast_weight * (color_contrast_loss + direction_contrast_loss) / 3.0
        + cfg.m1.semantic_direction_weight * semantic_direction_loss
        + cfg.m1.semantic_color_weight * semantic_color_loss
        + cfg.m1.full_rollout_direction_weight * full_rollout_direction_loss
        + cfg.m1.full_rollout_color_weight * full_rollout_color_loss
    )
    protected_shape_loss = (
        cfg.m1.prompt_contrast_weight * shape_contrast_loss / 3.0
        + cfg.m1.semantic_shape_weight * semantic_shape_loss
        + cfg.m1.full_rollout_shape_weight * full_rollout_shape_loss
        + research_shape_loss
    )
    optimization_loss = primary_optimization_loss + protected_shape_loss
    trainable = [*components.text_encoder.parameters(), *components.dit.parameters()]
    gradient_metrics: dict[str, float] = {}
    if diagnose_gradients and research_settings.gradient_diagnostics:
        diagnostic_parameters = trainable[-2:]
        gradient_metrics = gradient_cosine_matrix(
            {
                "diffusion": cfg.m1.diffusion_weight * diffusion_loss,
                "direction": cfg.m1.semantic_direction_weight * semantic_direction_loss,
                "color": cfg.m1.semantic_color_weight * semantic_color_loss,
                "shape": protected_shape_loss,
            },
            diagnostic_parameters,
        )

    shape_gradient_cosine = 0.0
    shape_gradient_projection_active = False
    if research_enabled and research_settings.shape_gradient_surgery:
        primary_gradients = torch.autograd.grad(
            primary_optimization_loss,
            trainable,
            retain_graph=True,
            allow_unused=True,
        )
        protected_gradients = torch.autograd.grad(
            protected_shape_loss,
            trainable,
            allow_unused=True,
        )
        merged_gradients, shape_gradient_cosine, shape_gradient_projection_active = (
            merge_primary_and_protected_gradients(primary_gradients, protected_gradients)
        )
        for parameter, gradient in zip(trainable, merged_gradients, strict=True):
            parameter.grad = gradient
    else:
        torch.autograd.backward(optimization_loss)
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, cfg.m1.grad_clip)
    optimizer.step()

    recon_value = float(recon_loss.item())
    diffusion_value = float(diffusion_loss.detach().item())
    unconditional_value = float(unconditional_loss.detach().item())
    counterfactual_value = float(counterfactual_loss.detach().item())
    contrast_value = float(contrast_loss.detach().item())
    total_value = cfg.m1.reconstruction_weight * recon_value + float(
        optimization_loss.detach().item()
    )
    return StageBStepMetrics(
        reconstruction_loss=recon_value,
        diffusion_loss=diffusion_value,
        total_loss=total_value,
        gradient_norm=float(grad_norm.detach().item()),
        unconditional_loss=unconditional_value,
        counterfactual_loss=counterfactual_value,
        prompt_contrast_loss=contrast_value,
        color_prompt_gap=float(color_gap.detach().mean().item()),
        direction_prompt_gap=float(direction_gap.detach().mean().item()),
        shape_prompt_gap=float(shape_gap.detach().mean().item()),
        semantic_direction_loss=float(semantic_direction_loss.detach().item()),
        semantic_color_loss=float(semantic_color_loss.detach().item()),
        semantic_shape_loss=float(semantic_shape_loss.detach().item()),
        full_rollout_direction_loss=float(full_rollout_direction_loss.detach().item()),
        full_rollout_color_loss=float(full_rollout_color_loss.detach().item()),
        full_rollout_shape_loss=float(full_rollout_shape_loss.detach().item()),
        causal_shape_loss=float(causal_shape_loss.detach().item()),
        latent_shape_rank_loss=float(latent_shape_rank_loss.detach().item()),
        latent_shape_rank_margin=float(latent_shape_rank_margin.detach().item()),
        sinkhorn_shape_loss=float(sinkhorn_shape_loss.detach().item()),
        sinkhorn_shape_gate=float(sinkhorn_shape_gate.detach().item()),
        latent_shape_delta_cosine=float(latent_shape_delta_cosine.detach().item()),
        latent_shape_delta_magnitude_ratio=float(
            latent_shape_delta_magnitude_ratio.detach().item()
        ),
        sinkhorn_shape_margin=float(sinkhorn_shape_margin.detach().item()),
        shape_gradient_cosine=shape_gradient_cosine,
        shape_gradient_projection_active=float(shape_gradient_projection_active),
        grad_cos_diffusion_direction=gradient_metrics.get("grad_cos_diffusion_direction", 0.0),
        grad_cos_diffusion_color=gradient_metrics.get("grad_cos_diffusion_color", 0.0),
        grad_cos_diffusion_shape=gradient_metrics.get("grad_cos_diffusion_shape", 0.0),
        grad_cos_direction_color=gradient_metrics.get("grad_cos_direction_color", 0.0),
        grad_cos_direction_shape=gradient_metrics.get("grad_cos_direction_shape", 0.0),
        grad_cos_color_shape=gradient_metrics.get("grad_cos_color_shape", 0.0),
    )


def _validate_reconstruction(
    *,
    components: StageBComponents,
    dataset: StageBSyntheticDataset,
    batch_size: int,
    cfg: ProjectConfig,
    device: torch.device,
) -> float:
    components.vae.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            indices = list(range(start, min(start + batch_size, len(dataset))))
            videos, _ = _sample_batch(dataset, indices, device=device)
            reconstruction, _ = components.vae(videos)
            loss = reconstruction_loss(
                reconstruction,
                videos,
                silhouette_weight=cfg.m1.vae_silhouette_weight,
            )
            total += float(loss.item()) * len(indices)
            count += len(indices)
    return total / count


def _validate_vae_visual_fidelity(
    *,
    components: StageBComponents,
    cfg: ProjectConfig,
    device: torch.device,
) -> tuple[float, float, float, float, float, float]:
    """Evaluate whether VAE reconstruction preserves controlled renderer semantics."""
    from vexa_video.training.evaluation import evaluate_generated_videos

    dataset = StageBSyntheticDataset(
        length=cfg.m1.vae_visual_validation_samples,
        frames=cfg.data.frames,
        size=cfg.data.height,
        base_seed=cfg.seed,
        split="validation",
    )
    reconstructed_batches: list[Tensor] = []
    controls: list[SyntheticControl] = []
    components.vae.eval()
    with torch.no_grad():
        for start in range(0, len(dataset), cfg.m1.eval_batch_size):
            stop = min(start + cfg.m1.eval_batch_size, len(dataset))
            samples = [dataset.sample(index) for index in range(start, stop)]
            videos = torch.stack([sample.video for sample in samples]).to(device)
            reconstruction, _ = components.vae(videos)
            reconstructed_batches.append(reconstruction.cpu())
            controls.extend(sample.control for sample in samples)
    metrics = evaluate_generated_videos(
        torch.cat(reconstructed_batches, dim=0),
        controls,
        static_motion_threshold=cfg.m1.static_motion_threshold,
        sampling_steps=0,
    )
    return (
        metrics.direction_accuracy,
        metrics.color_accuracy,
        metrics.shape_accuracy,
        metrics.object_like_frame_rate,
        metrics.persistent_video_rate,
        metrics.mean_foreground_area_ratio,
    )


def _passes_vae_visual_gate(
    metrics: tuple[float, float, float, float, float, float], cfg: ProjectConfig
) -> bool:
    direction, color, shape, object_like, persistent, area_ratio = metrics
    semantic_gate = cfg.m1.vae_semantic_accuracy_gate
    return (
        direction >= semantic_gate
        and color >= semantic_gate
        and shape >= semantic_gate
        and object_like >= cfg.m1.vae_object_like_frame_gate
        and persistent >= cfg.m1.vae_persistent_video_gate
        and cfg.m1.vae_foreground_area_ratio_min
        <= area_ratio
        <= cfg.m1.vae_foreground_area_ratio_max
    )


def validate_stage_b(
    *,
    components: StageBComponents,
    dataset: StageBSyntheticDataset,
    cfg: ProjectConfig,
    device: torch.device,
) -> tuple[float, float, float, float, float, float, float, float]:
    components.vae.eval()
    components.text_encoder.eval()
    components.dit.eval()
    generator = torch.Generator(device=device).manual_seed(cfg.seed + 7_001)
    reconstruction_total = 0.0
    diffusion_total = 0.0
    color_gap_total = 0.0
    direction_gap_total = 0.0
    shape_gap_total = 0.0
    semantic_direction_total = 0.0
    semantic_color_total = 0.0
    semantic_shape_total = 0.0
    count = 0

    with torch.no_grad():
        for start in range(0, len(dataset), cfg.m1.batch_size):
            indices = list(range(start, min(start + cfg.m1.batch_size, len(dataset))))
            videos, captions = _sample_batch(dataset, indices, device=device)
            reconstruction, latents = components.vae(videos)
            recon_loss = reconstruction_loss(
                reconstruction,
                videos,
                silhouette_weight=cfg.m1.vae_silhouette_weight,
            )
            timesteps = torch.randint(
                0,
                components.schedule.timesteps,
                (len(indices),),
                generator=generator,
                device=device,
            )
            noise = torch.randn(
                latents.shape,
                generator=generator,
                device=device,
                dtype=latents.dtype,
            )
            noisy = components.schedule.add_noise(latents, noise, timesteps)
            (
                conditioned,
                color_counterfactual,
                direction_counterfactual,
                shape_counterfactual,
                _,
            ) = _conditioning_predictions(
                components=components,
                noisy_latents=noisy,
                timesteps=timesteps,
                captions=captions,
                cfg=cfg,
                device=device,
            )
            conditioned_error = _per_sample_mse(conditioned, noise)
            color_error = _per_sample_mse(color_counterfactual, noise)
            direction_error = _per_sample_mse(direction_counterfactual, noise)
            shape_error = _per_sample_mse(shape_counterfactual, noise)
            diff_loss = conditioned_error.mean()
            color_gap = (color_error - conditioned_error).mean()
            direction_gap = (direction_error - conditioned_error).mean()
            shape_gap = (shape_error - conditioned_error).mean()
            semantic_noise = torch.randn(
                latents.shape,
                generator=generator,
                device=device,
                dtype=latents.dtype,
            )
            semantic_direction, semantic_color, semantic_shape = _semantic_conditioning_losses(
                components=components,
                clean_latents=latents,
                captions=captions,
                cfg=cfg,
                device=device,
                noise=semantic_noise,
            )
            reconstruction_total += float(recon_loss.item()) * len(indices)
            diffusion_total += float(diff_loss.item()) * len(indices)
            color_gap_total += float(color_gap.item()) * len(indices)
            direction_gap_total += float(direction_gap.item()) * len(indices)
            shape_gap_total += float(shape_gap.item()) * len(indices)
            semantic_direction_total += float(semantic_direction.item()) * len(indices)
            semantic_color_total += float(semantic_color.item()) * len(indices)
            semantic_shape_total += float(semantic_shape.item()) * len(indices)
            count += len(indices)
    return (
        reconstruction_total / count,
        diffusion_total / count,
        color_gap_total / count,
        direction_gap_total / count,
        shape_gap_total / count,
        semantic_direction_total / count,
        semantic_color_total / count,
        semantic_shape_total / count,
    )


def _validation_generation_metrics(
    *,
    components: StageBComponents,
    cfg: ProjectConfig,
    device: torch.device,
) -> tuple[float, float, float, float, float, float, float, float]:
    """Measure fixed validation-split generation at the full frozen sampler horizon."""
    from vexa_video.training.evaluation import evaluate_generated_videos

    dataset = StageBSyntheticDataset(
        length=cfg.m1.validation_generation_samples,
        frames=cfg.data.frames,
        size=cfg.data.height,
        base_seed=cfg.seed,
        split="validation",
    )
    generated_batches: list[Tensor] = []
    controls: list[SyntheticControl] = []
    for start in range(0, len(dataset), cfg.m1.eval_batch_size):
        stop = min(start + cfg.m1.eval_batch_size, len(dataset))
        batch_samples = [dataset.sample(index) for index in range(start, stop)]
        generated = sample_video(
            cfg=cfg,
            tokenizer=components.tokenizer,
            text_encoder=components.text_encoder,
            vae=components.vae,
            dit=components.dit,
            schedule=components.schedule,
            prompts=[sample.caption for sample in batch_samples],
            seed=cfg.seed + 80_000 + start,
            sampling_steps=cfg.m1.full_rollout_steps,
            device=device,
        )
        generated_batches.append(generated.cpu())
        controls.extend(sample.control for sample in batch_samples)
    metrics = evaluate_generated_videos(
        torch.cat(generated_batches, dim=0),
        controls,
        static_motion_threshold=cfg.m1.static_motion_threshold,
        sampling_steps=cfg.m1.full_rollout_steps,
    )
    return (
        metrics.direction_accuracy,
        metrics.color_accuracy,
        metrics.shape_accuracy,
        metrics.mean_motion,
        metrics.static_rate,
        metrics.object_like_frame_rate,
        metrics.persistent_video_rate,
        metrics.mean_foreground_area_ratio,
    )


def _freeze_vae(vae: TinyVideoVAE) -> None:
    vae.eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)


def _normalize_cuda_rng_states(raw_states: object) -> list[Tensor]:
    if not isinstance(raw_states, (list, tuple)):
        raise ValueError("cuda_rng_state_all must contain a list or tuple of tensors")
    states: list[Tensor] = []
    for state in raw_states:
        if not isinstance(state, Tensor):
            raise ValueError("cuda_rng_state_all entries must be tensors")
        if state.dtype != torch.uint8:
            raise ValueError("CUDA RNG states must use torch.uint8 dtype")
        states.append(state.detach().cpu())
    return states


def _validate_checkpoint_vae_contract(
    payload: dict[str, Any], components: StageBComponents
) -> None:
    raw_config = payload.get("config")
    if not isinstance(raw_config, dict):
        return
    raw_vae = raw_config.get("vae")
    if not isinstance(raw_vae, dict):
        return
    stored_spatial = raw_vae.get("spatial_downsample")
    stored_temporal = raw_vae.get("temporal_downsample")
    if stored_spatial is None or stored_temporal is None:
        return
    stored_contract = (int(stored_spatial), int(stored_temporal))
    current_contract = (
        components.vae.spatial_downsample,
        components.vae.temporal_downsample,
    )
    if stored_contract != current_contract:
        raise ValueError(
            "checkpoint VAE compression is incompatible with the current latent contract: "
            f"checkpoint={stored_contract[0]}x/{stored_contract[1]}x, "
            f"current={current_contract[0]}x/{current_contract[1]}x. "
            "Start a clean M1 shape-recovery run instead of resuming this checkpoint."
        )


def _restore_checkpoint(
    *,
    checkpoint_path: Path,
    components: StageBComponents,
    vae_optimizer: torch.optim.Optimizer,
    diffusion_optimizer: torch.optim.Optimizer,
    data_generator: torch.Generator,
    device: torch.device,
) -> tuple[int, int, float, str]:
    raw = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(raw, dict):
        raise ValueError("Stage-B checkpoint must contain a dictionary payload")
    payload = cast(dict[str, Any], raw)
    if payload.get("stage") != "stage-b-generative-synthetic-motion":
        raise ValueError("checkpoint is not an M1 Stage-B checkpoint")
    _validate_checkpoint_vae_contract(payload, components)
    models = cast(dict[str, Any], payload["models"])
    optimizers = cast(dict[str, Any], payload["optimizers"])
    components.vae.load_state_dict(models["vae"])
    components.text_encoder.load_state_dict(models["text_encoder"])
    components.dit.load_state_dict(models["dit"])
    vae_optimizer.load_state_dict(optimizers["vae"])
    diffusion_optimizer.load_state_dict(optimizers["diffusion"])
    torch.set_rng_state(cast(Tensor, payload["torch_rng_state"]).cpu())
    data_generator.set_state(cast(Tensor, payload["data_generator_state"]).cpu())
    if device.type == "cuda" and "cuda_rng_state_all" in payload:
        torch.cuda.set_rng_state_all(_normalize_cuda_rng_states(payload["cuda_rng_state_all"]))
    vae_step = int(payload.get("vae_step", 0))
    diffusion_step = int(payload.get("diffusion_step", 0))
    best_validation_score = float(payload.get("best_validation_score", math.inf))
    phase = str(payload.get("phase", "vae_warmup"))
    return vae_step, diffusion_step, best_validation_score, phase


def _save_checkpoint(
    *,
    path: Path,
    components: StageBComponents,
    vae_optimizer: torch.optim.Optimizer,
    diffusion_optimizer: torch.optim.Optimizer,
    cfg: ProjectConfig,
    vae_step: int,
    diffusion_step: int,
    best_validation_score: float,
    phase: str,
    metrics: StageBStepMetrics | dict[str, float],
    data_generator: torch.Generator,
) -> None:
    metric_payload = asdict(metrics) if isinstance(metrics, StageBStepMetrics) else metrics
    checkpoint = build_training_checkpoint(
        vae=components.vae,
        text_encoder=components.text_encoder,
        dit=components.dit,
        vae_optimizer=vae_optimizer,
        diffusion_optimizer=diffusion_optimizer,
        global_step=vae_step + diffusion_step,
        vae_step=vae_step,
        diffusion_step=diffusion_step,
        config=asdict(cfg),
        metrics={key: float(value) for key, value in metric_payload.items()},
        best_validation_score=best_validation_score,
        phase=phase,
        data_generator_state=data_generator.get_state(),
        metadata={
            "selection": (
                "best.pt minimizes held-out reconstruction + diffusion + "
                "prompt-gap + semantic + full-horizon generated-validation penalties"
            ),
            "curriculum": "one object; four directions; four colors; square/circle; medium speed",
            "conditioning": (
                "byte-text only; token attention + text counterfactuals + "
                "decoded semantic losses + balanced full-horizon motion/color/shape preservation"
            ),
        },
    )
    torch.save(checkpoint, path)


def train_stage_b(
    cfg: ProjectConfig,
    *,
    device: torch.device,
    run_dir: str | Path,
    steps: int | None = None,
    resume: str | Path | None = None,
    vae_only: bool = False,
    research: M1ResearchSettings | None = None,
) -> StageBTrainResult:
    if cfg.data.height != cfg.data.width:
        raise ValueError("M1 Stage-B currently requires square video dimensions")
    target_steps = steps if steps is not None else cfg.m1.max_steps
    if target_steps <= 0:
        raise ValueError("steps must be positive")
    research_settings = research or M1ResearchSettings.from_env()
    research_settings.validate()
    output_dir = Path(run_dir)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    latest_path = checkpoints_dir / "latest.pt"
    best_path = checkpoints_dir / "best.pt"

    train_dataset = StageBSyntheticDataset(
        length=cfg.m1.train_samples,
        frames=cfg.data.frames,
        size=cfg.data.height,
        base_seed=cfg.seed,
        split="train",
    )
    validation_dataset = StageBSyntheticDataset(
        length=cfg.m1.validation_samples,
        frames=cfg.data.frames,
        size=cfg.data.height,
        base_seed=cfg.seed,
        split="validation",
    )
    components = build_stage_b_components(cfg, device=device)
    vae_optimizer = torch.optim.AdamW(
        components.vae.parameters(),
        lr=cfg.m1.vae_learning_rate,
    )
    diffusion_optimizer = torch.optim.AdamW(
        [*components.text_encoder.parameters(), *components.dit.parameters()],
        lr=cfg.m1.learning_rate,
    )
    data_generator = torch.Generator().manual_seed(cfg.seed + 5_001)
    vae_step = 0
    diffusion_step = 0
    best_validation_score = math.inf
    phase = "vae_warmup"

    if resume is not None:
        resume_path = Path(resume)
        vae_step, diffusion_step, best_validation_score, phase = _restore_checkpoint(
            checkpoint_path=resume_path,
            components=components,
            vae_optimizer=vae_optimizer,
            diffusion_optimizer=diffusion_optimizer,
            data_generator=data_generator,
            device=device,
        )
        raw_resume_config = torch.load(resume_path, map_location="cpu", weights_only=False).get(
            "config", {}
        )
        resume_m1 = raw_resume_config.get("m1", {}) if isinstance(raw_resume_config, dict) else {}
        current_selector_keys = {
            "full_rollout_color_weight",
            "full_rollout_shape_weight",
            "full_rollout_min_motion",
            "validation_generation_samples",
            "generation_shape_gate",
            "generation_persistent_video_gate",
        }
        if not isinstance(resume_m1, dict) or not current_selector_keys.issubset(resume_m1):
            best_validation_score = math.inf
            print("training resume_objective_changed reset_best_validation_score=true")
        print(
            "m1_stage_b resumed "
            f"checkpoint={resume_path} phase={phase} vae_step={vae_step} "
            f"diffusion_step={diffusion_step}"
        )

    while vae_step < cfg.m1.vae_warmup_steps:
        vae_step += 1
        indices = _random_indices(
            dataset_size=len(train_dataset),
            batch_size=cfg.m1.batch_size,
            generator=data_generator,
        )
        videos, _ = _sample_batch(train_dataset, indices, device=device)
        metrics = vae_warmup_step(
            components=components,
            optimizer=vae_optimizer,
            videos=videos,
            cfg=cfg,
        )
        if vae_step == 1 or vae_step % cfg.m1.log_every == 0:
            print(
                f"m1_stage_b phase=vae step={vae_step} "
                f"reconstruction={metrics.reconstruction_loss:.6f} "
                f"grad_norm={metrics.gradient_norm:.6f}"
            )
        if vae_step % cfg.m1.checkpoint_every == 0:
            _save_checkpoint(
                path=latest_path,
                components=components,
                vae_optimizer=vae_optimizer,
                diffusion_optimizer=diffusion_optimizer,
                cfg=cfg,
                vae_step=vae_step,
                diffusion_step=diffusion_step,
                best_validation_score=best_validation_score,
                phase="vae_warmup",
                metrics=metrics,
                data_generator=data_generator,
            )

    vae_validation_loss = _validate_reconstruction(
        components=components,
        dataset=validation_dataset,
        batch_size=cfg.m1.batch_size,
        cfg=cfg,
        device=device,
    )
    print(f"m1_stage_b vae_validation_reconstruction={vae_validation_loss:.6f}")
    if vae_validation_loss > cfg.m1.vae_reconstruction_gate:
        _save_checkpoint(
            path=latest_path,
            components=components,
            vae_optimizer=vae_optimizer,
            diffusion_optimizer=diffusion_optimizer,
            cfg=cfg,
            vae_step=vae_step,
            diffusion_step=diffusion_step,
            best_validation_score=best_validation_score,
            phase="vae_gate_failed",
            metrics={"vae_validation_reconstruction_loss": vae_validation_loss},
            data_generator=data_generator,
        )
        raise RuntimeError(
            "Stage-B VAE reconstruction safety gate failed: "
            f"{vae_validation_loss:.6f} > {cfg.m1.vae_reconstruction_gate:.6f}. "
            "Diffusion training was not started."
        )

    vae_visual_metrics = _validate_vae_visual_fidelity(
        components=components,
        cfg=cfg,
        device=device,
    )
    (
        vae_direction,
        vae_color,
        vae_shape,
        vae_object_like,
        vae_persistent,
        vae_area_ratio,
    ) = vae_visual_metrics
    print(
        "m1_stage_b vae_visual "
        f"direction={vae_direction:.6f} "
        f"color={vae_color:.6f} "
        f"shape={vae_shape:.6f} "
        f"object_like={vae_object_like:.6f} "
        f"persistent={vae_persistent:.6f} "
        f"foreground_area_ratio={vae_area_ratio:.6f}"
    )
    vae_visual_metric_payload = {
        "vae_validation_reconstruction_loss": vae_validation_loss,
        "vae_visual_direction_accuracy": vae_direction,
        "vae_visual_color_accuracy": vae_color,
        "vae_visual_shape_accuracy": vae_shape,
        "vae_visual_object_like_frame_rate": vae_object_like,
        "vae_visual_persistent_video_rate": vae_persistent,
        "vae_visual_foreground_area_ratio": vae_area_ratio,
    }
    if not _passes_vae_visual_gate(vae_visual_metrics, cfg):
        _save_checkpoint(
            path=latest_path,
            components=components,
            vae_optimizer=vae_optimizer,
            diffusion_optimizer=diffusion_optimizer,
            cfg=cfg,
            vae_step=vae_step,
            diffusion_step=diffusion_step,
            best_validation_score=best_validation_score,
            phase="vae_visual_gate_failed",
            metrics=vae_visual_metric_payload,
            data_generator=data_generator,
        )
        raise RuntimeError(
            "M1 VAE visual-fidelity gate failed; diffusion training was not started. "
            "Inspect the reported direction/color/shape/persistence/foreground-area metrics."
        )

    shape_svd_metrics = _shape_latent_svd_metrics(
        components=components,
        dataset=validation_dataset,
        research=research_settings,
        device=device,
    )
    vae_visual_metric_payload.update(shape_svd_metrics)
    print(
        "m1_stage_b shape_latent_svd "
        f"top1={shape_svd_metrics['shape_svd_top1_energy']:.6f} "
        f"top4={shape_svd_metrics['shape_svd_top4_energy']:.6f} "
        f"top8={shape_svd_metrics['shape_svd_top8_energy']:.6f} "
        f"effective_rank={shape_svd_metrics['shape_svd_effective_rank']:.6f}"
    )

    if vae_only:
        _save_checkpoint(
            path=latest_path,
            components=components,
            vae_optimizer=vae_optimizer,
            diffusion_optimizer=diffusion_optimizer,
            cfg=cfg,
            vae_step=vae_step,
            diffusion_step=diffusion_step,
            best_validation_score=best_validation_score,
            phase="vae_ready",
            metrics=vae_visual_metric_payload,
            data_generator=data_generator,
        )
        return StageBTrainResult(
            latest_checkpoint=latest_path,
            best_checkpoint=None,
            diffusion_step=diffusion_step,
            vae_validation_reconstruction_loss=vae_validation_loss,
            best_validation_score=best_validation_score,
        )

    _freeze_vae(components.vae)
    phase = "diffusion"
    last_metrics = StageBStepMetrics(
        reconstruction_loss=vae_validation_loss,
        diffusion_loss=math.nan,
        total_loss=math.nan,
        gradient_norm=math.nan,
    )

    while diffusion_step < target_steps:
        diffusion_step += 1
        indices = _random_indices(
            dataset_size=len(train_dataset),
            batch_size=cfg.m1.batch_size,
            generator=data_generator,
        )
        videos, captions = _sample_batch(train_dataset, indices, device=device)
        paired_videos, paired_captions = _sample_shape_counterfactual_batch(
            train_dataset,
            indices,
            device=device,
        )
        last_metrics = diffusion_train_step(
            components=components,
            optimizer=diffusion_optimizer,
            videos=videos,
            captions=captions,
            paired_videos=paired_videos,
            paired_captions=paired_captions,
            cfg=cfg,
            device=device,
            research=research_settings,
            run_full_rollout=diffusion_step % cfg.m1.full_rollout_every == 0,
            diagnose_gradients=diffusion_step % cfg.m1.checkpoint_every == 0,
        )
        if diffusion_step == 1 or diffusion_step % cfg.m1.log_every == 0:
            print(
                f"m1_stage_b phase=diffusion step={diffusion_step} "
                f"reconstruction={last_metrics.reconstruction_loss:.6f} "
                f"diffusion={last_metrics.diffusion_loss:.6f} "
                f"contrast={last_metrics.prompt_contrast_loss:.6f} "
                f"color_gap={last_metrics.color_prompt_gap:.6f} "
                f"direction_gap={last_metrics.direction_prompt_gap:.6f} "
                f"shape_gap={last_metrics.shape_prompt_gap:.6f} "
                f"semantic_direction={last_metrics.semantic_direction_loss:.6f} "
                f"semantic_color={last_metrics.semantic_color_loss:.6f} "
                f"semantic_shape={last_metrics.semantic_shape_loss:.6f} "
                f"full_rollout_direction={last_metrics.full_rollout_direction_loss:.6f} "
                f"full_rollout_color={last_metrics.full_rollout_color_loss:.6f} "
                f"full_rollout_shape={last_metrics.full_rollout_shape_loss:.6f} "
                f"causal_shape={last_metrics.causal_shape_loss:.6f} "
                f"latent_rank={last_metrics.latent_shape_rank_loss:.6f} "
                f"latent_rank_margin={last_metrics.latent_shape_rank_margin:.6f} "
                f"sinkhorn_shape={last_metrics.sinkhorn_shape_loss:.6f} "
                f"sinkhorn_gate={last_metrics.sinkhorn_shape_gate:.6f} "
                f"latent_shape_cos={last_metrics.latent_shape_delta_cosine:.6f} "
                f"latent_shape_ratio={last_metrics.latent_shape_delta_magnitude_ratio:.6f} "
                f"sinkhorn_margin={last_metrics.sinkhorn_shape_margin:.6f} "
                f"shape_grad_cos={last_metrics.shape_gradient_cosine:.6f} "
                f"shape_grad_projected={last_metrics.shape_gradient_projection_active:.0f} "
                f"total={last_metrics.total_loss:.6f} "
                f"grad_norm={last_metrics.gradient_norm:.6f}"
            )

        if diffusion_step % cfg.m1.checkpoint_every == 0 and research_settings.gradient_diagnostics:
            print(
                "m1_stage_b gradient_cosines "
                f"step={diffusion_step} "
                f"diff_direction={last_metrics.grad_cos_diffusion_direction:.6f} "
                f"diff_color={last_metrics.grad_cos_diffusion_color:.6f} "
                f"diff_shape={last_metrics.grad_cos_diffusion_shape:.6f} "
                f"direction_color={last_metrics.grad_cos_direction_color:.6f} "
                f"direction_shape={last_metrics.grad_cos_direction_shape:.6f} "
                f"color_shape={last_metrics.grad_cos_color_shape:.6f}"
            )

        should_validate = (
            diffusion_step % cfg.m1.checkpoint_every == 0 or diffusion_step == target_steps
        )
        if should_validate:
            (
                validation_reconstruction,
                validation_diffusion,
                validation_color_gap,
                validation_direction_gap,
                validation_shape_gap,
                validation_semantic_direction,
                validation_semantic_color,
                validation_semantic_shape,
            ) = validate_stage_b(
                components=components,
                dataset=validation_dataset,
                cfg=cfg,
                device=device,
            )
            conditioning_penalty = (
                max(0.0, cfg.m1.prompt_contrast_margin - validation_color_gap)
                + max(0.0, cfg.m1.prompt_contrast_margin - validation_direction_gap)
                + max(0.0, cfg.m1.prompt_contrast_margin - validation_shape_gap)
            ) / 3.0
            (
                validation_generated_direction,
                validation_generated_color,
                validation_generated_shape,
                validation_generated_motion,
                validation_generated_static,
                validation_generated_object_like,
                validation_generated_persistent,
                validation_generated_area_ratio,
            ) = _validation_generation_metrics(
                components=components,
                cfg=cfg,
                device=device,
            )
            generated_direction_penalty = (
                max(
                    0.0,
                    cfg.m1.generation_direction_gate - validation_generated_direction,
                )
                / cfg.m1.generation_direction_gate
            )
            generated_color_penalty = (
                max(
                    0.0,
                    cfg.m1.generation_color_gate - validation_generated_color,
                )
                / cfg.m1.generation_color_gate
            )
            generated_shape_penalty = (
                max(
                    0.0,
                    cfg.m1.generation_shape_gate - validation_generated_shape,
                )
                / cfg.m1.generation_shape_gate
            )
            generated_static_penalty = max(
                0.0,
                validation_generated_static - cfg.m1.generation_static_rate_gate,
            ) / max(cfg.m1.generation_static_rate_gate, 1e-8)
            generated_motion_penalty = (
                max(
                    0.0,
                    cfg.m1.generation_mean_motion_gate - validation_generated_motion,
                )
                / cfg.m1.generation_mean_motion_gate
            )
            generated_object_like_penalty = (
                max(
                    0.0,
                    cfg.m1.generation_object_like_frame_gate - validation_generated_object_like,
                )
                / cfg.m1.generation_object_like_frame_gate
            )
            generated_persistent_penalty = (
                max(
                    0.0,
                    cfg.m1.generation_persistent_video_gate - validation_generated_persistent,
                )
                / cfg.m1.generation_persistent_video_gate
            )
            if validation_generated_area_ratio < cfg.m1.generation_foreground_area_ratio_min:
                generated_area_penalty = (
                    cfg.m1.generation_foreground_area_ratio_min - validation_generated_area_ratio
                ) / cfg.m1.generation_foreground_area_ratio_min
            elif validation_generated_area_ratio > cfg.m1.generation_foreground_area_ratio_max:
                generated_area_penalty = (
                    validation_generated_area_ratio - cfg.m1.generation_foreground_area_ratio_max
                ) / cfg.m1.generation_foreground_area_ratio_max
            else:
                generated_area_penalty = 0.0
            generated_other_penalty = (
                generated_color_penalty
                + generated_shape_penalty
                + generated_static_penalty
                + generated_motion_penalty
                + generated_object_like_penalty
                + generated_persistent_penalty
                + generated_area_penalty
            )
            constraint_violation = generated_direction_penalty + generated_other_penalty
            validation_tie_break_score = (
                cfg.m1.reconstruction_weight * validation_reconstruction
                + cfg.m1.diffusion_weight * validation_diffusion
                + cfg.m1.prompt_contrast_weight * conditioning_penalty
                + cfg.m1.semantic_direction_weight * validation_semantic_direction
                + cfg.m1.semantic_color_weight * validation_semantic_color
                + cfg.m1.semantic_shape_weight * validation_semantic_shape
            )
            validation_score = constraint_rank_score(
                constraint_violation,
                validation_tie_break_score,
            )
            checkpoint_metrics = {
                **asdict(last_metrics),
                "validation_reconstruction_loss": validation_reconstruction,
                "validation_diffusion_loss": validation_diffusion,
                "validation_color_prompt_gap": validation_color_gap,
                "validation_direction_prompt_gap": validation_direction_gap,
                "validation_shape_prompt_gap": validation_shape_gap,
                "validation_conditioning_penalty": conditioning_penalty,
                "validation_semantic_direction_loss": validation_semantic_direction,
                "validation_semantic_color_loss": validation_semantic_color,
                "validation_semantic_shape_loss": validation_semantic_shape,
                "validation_generated_direction_accuracy": validation_generated_direction,
                "validation_generated_color_accuracy": validation_generated_color,
                "validation_generated_shape_accuracy": validation_generated_shape,
                "validation_generated_mean_motion": validation_generated_motion,
                "validation_generated_static_rate": validation_generated_static,
                "validation_generated_object_like_frame_rate": validation_generated_object_like,
                "validation_generated_persistent_video_rate": validation_generated_persistent,
                "validation_generated_foreground_area_ratio": validation_generated_area_ratio,
                "validation_generated_direction_penalty": generated_direction_penalty,
                "validation_generated_color_penalty": generated_color_penalty,
                "validation_generated_shape_penalty": generated_shape_penalty,
                "validation_generated_static_penalty": generated_static_penalty,
                "validation_generated_motion_penalty": generated_motion_penalty,
                "validation_generated_object_like_penalty": generated_object_like_penalty,
                "validation_generated_persistent_penalty": generated_persistent_penalty,
                "validation_generated_area_penalty": generated_area_penalty,
                "validation_generated_other_penalty": generated_other_penalty,
                "validation_constraint_violation": constraint_violation,
                "validation_tie_break_score": validation_tie_break_score,
                "validation_score": validation_score,
                **shape_svd_metrics,
                "vae_visual_direction_accuracy": vae_direction,
                "vae_visual_color_accuracy": vae_color,
                "vae_visual_shape_accuracy": vae_shape,
                "vae_visual_object_like_frame_rate": vae_object_like,
                "vae_visual_persistent_video_rate": vae_persistent,
                "vae_visual_foreground_area_ratio": vae_area_ratio,
                "vae_validation_reconstruction_loss": vae_validation_loss,
            }
            if validation_score < best_validation_score:
                best_validation_score = validation_score
                _save_checkpoint(
                    path=best_path,
                    components=components,
                    vae_optimizer=vae_optimizer,
                    diffusion_optimizer=diffusion_optimizer,
                    cfg=cfg,
                    vae_step=vae_step,
                    diffusion_step=diffusion_step,
                    best_validation_score=best_validation_score,
                    phase=phase,
                    metrics=checkpoint_metrics,
                    data_generator=data_generator,
                )
            _save_checkpoint(
                path=latest_path,
                components=components,
                vae_optimizer=vae_optimizer,
                diffusion_optimizer=diffusion_optimizer,
                cfg=cfg,
                vae_step=vae_step,
                diffusion_step=diffusion_step,
                best_validation_score=best_validation_score,
                phase=phase,
                metrics=checkpoint_metrics,
                data_generator=data_generator,
            )
            (output_dir / "train-metrics.json").write_text(
                json.dumps(checkpoint_metrics, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(
                f"m1_stage_b validation step={diffusion_step} "
                f"reconstruction={validation_reconstruction:.6f} "
                f"diffusion={validation_diffusion:.6f} "
                f"color_gap={validation_color_gap:.6f} "
                f"direction_gap={validation_direction_gap:.6f} "
                f"shape_gap={validation_shape_gap:.6f} "
                f"semantic_direction={validation_semantic_direction:.6f} "
                f"semantic_color={validation_semantic_color:.6f} "
                f"semantic_shape={validation_semantic_shape:.6f} "
                f"generated_direction={validation_generated_direction:.6f} "
                f"generated_color={validation_generated_color:.6f} "
                f"generated_shape={validation_generated_shape:.6f} "
                f"generated_motion={validation_generated_motion:.6f} "
                f"generated_static={validation_generated_static:.6f} "
                f"generated_object_like={validation_generated_object_like:.6f} "
                f"generated_persistent={validation_generated_persistent:.6f} "
                f"generated_area_ratio={validation_generated_area_ratio:.6f} "
                f"constraint_violation={constraint_violation:.6f} "
                f"score={validation_score:.6f}"
            )

    return StageBTrainResult(
        latest_checkpoint=latest_path,
        best_checkpoint=best_path if best_path.exists() else None,
        diffusion_step=diffusion_step,
        vae_validation_reconstruction_loss=vae_validation_loss,
        best_validation_score=best_validation_score,
    )


def load_stage_b_weights(
    checkpoint: str | Path,
    *,
    components: StageBComponents,
    device: torch.device,
) -> dict[str, Any]:
    """Load synthetic-motion model states for deterministic evaluation/generation."""
    raw = torch.load(Path(checkpoint), map_location=device, weights_only=False)
    if not isinstance(raw, dict):
        raise ValueError("Stage-B checkpoint must contain a dictionary payload")
    payload = cast(dict[str, Any], raw)
    if payload.get("stage") != "stage-b-generative-synthetic-motion":
        raise ValueError("checkpoint is not an M1 Stage-B checkpoint")
    _validate_checkpoint_vae_contract(payload, components)
    models = cast(dict[str, Any], payload["models"])
    components.vae.load_state_dict(models["vae"])
    components.text_encoder.load_state_dict(models["text_encoder"])
    components.dit.load_state_dict(models["dit"])
    return payload
