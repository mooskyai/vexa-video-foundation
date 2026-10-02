from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor

from vexa_video.data.controlled_motion import controlled_motion_shape_size
from vexa_video.diffusion import LinearNoiseSchedule


@dataclass(frozen=True, slots=True)
class M1ResearchSettings:
    """Research-only controls for the M1 conditioning rescue path."""

    causal_shape_weight: float = 1.0
    latent_rank_weight: float = 1.0
    latent_rank_margin: float = 0.25
    sinkhorn_shape_weight: float = 0.5
    sinkhorn_blur: float = 0.12
    sinkhorn_margin: float = 0.02
    sinkhorn_gate_start: float = 0.10
    sinkhorn_gate_full: float = 0.30
    min_snr_gamma: float = 5.0
    shape_gradient_surgery: bool = True
    gradient_diagnostics: bool = True
    svd_samples: int = 32

    def validate(self) -> None:
        if self.causal_shape_weight < 0.0:
            raise ValueError("causal_shape_weight must be non-negative")
        if self.latent_rank_weight < 0.0:
            raise ValueError("latent_rank_weight must be non-negative")
        if self.latent_rank_margin < 0.0:
            raise ValueError("latent_rank_margin must be non-negative")
        if self.sinkhorn_shape_weight < 0.0:
            raise ValueError("sinkhorn_shape_weight must be non-negative")
        if self.sinkhorn_blur <= 0.0:
            raise ValueError("sinkhorn_blur must be positive")
        if self.sinkhorn_margin < 0.0:
            raise ValueError("sinkhorn_margin must be non-negative")
        if not 0.0 <= self.sinkhorn_gate_start < self.sinkhorn_gate_full <= 1.0:
            raise ValueError("require 0 <= sinkhorn_gate_start < sinkhorn_gate_full <= 1")
        if self.min_snr_gamma <= 0.0:
            raise ValueError("min_snr_gamma must be positive")
        if self.svd_samples <= 1:
            raise ValueError("svd_samples must be greater than one")

    @classmethod
    def from_env(cls) -> M1ResearchSettings:
        settings = cls(
            causal_shape_weight=_env_float("VEXA_M1_CAUSAL_SHAPE_WEIGHT", 1.0),
            latent_rank_weight=_env_float("VEXA_M1_LATENT_RANK_WEIGHT", 1.0),
            latent_rank_margin=_env_float("VEXA_M1_LATENT_RANK_MARGIN", 0.25),
            sinkhorn_shape_weight=_env_float("VEXA_M1_SINKHORN_SHAPE_WEIGHT", 0.5),
            sinkhorn_blur=_env_float("VEXA_M1_SINKHORN_BLUR", 0.12),
            sinkhorn_margin=_env_float("VEXA_M1_SINKHORN_MARGIN", 0.02),
            sinkhorn_gate_start=_env_float("VEXA_M1_SINKHORN_GATE_START", 0.10),
            sinkhorn_gate_full=_env_float("VEXA_M1_SINKHORN_GATE_FULL", 0.30),
            min_snr_gamma=_env_float("VEXA_M1_MIN_SNR_GAMMA", 5.0),
            shape_gradient_surgery=os.getenv("VEXA_M1_SHAPE_GRADIENT_SURGERY", "1") != "0",
            gradient_diagnostics=os.getenv("VEXA_M1_GRADIENT_DIAGNOSTICS", "1") != "0",
            svd_samples=int(os.getenv("VEXA_M1_SVD_SAMPLES", "32")),
        )
        settings.validate()
        return settings


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None else float(raw)


def min_snr_epsilon_weights(
    schedule: LinearNoiseSchedule,
    timesteps: Tensor,
    *,
    gamma: float,
    dtype: torch.dtype,
) -> Tensor:
    """Return Min-SNR-gamma weights for epsilon prediction."""
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    alpha = schedule.alpha_cumprod.to(device=timesteps.device, dtype=dtype)[timesteps]
    snr = alpha / (1.0 - alpha).clamp_min(1e-8)
    gamma_tensor = torch.full_like(snr, gamma)
    return torch.minimum(snr, gamma_tensor) / snr.clamp_min(1e-8)


def causal_latent_delta_loss(
    predicted_first: Tensor,
    predicted_second: Tensor,
    target_first: Tensor,
    target_second: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Match the latent change caused only by switching square/circle text."""
    if predicted_first.shape != predicted_second.shape:
        raise ValueError("predicted latent pairs must have identical shapes")
    if target_first.shape != target_second.shape or target_first.shape != predicted_first.shape:
        raise ValueError("predicted and target latent pairs must have identical shapes")

    predicted_delta = predicted_first - predicted_second
    target_delta = target_first - target_second
    predicted_flat = predicted_delta.flatten(start_dim=1)
    target_flat = target_delta.flatten(start_dim=1)
    target_rms = target_flat.square().mean(dim=1).sqrt().clamp_min(1e-4)
    normalized_predicted = predicted_flat / target_rms[:, None]
    normalized_target = target_flat / target_rms[:, None]
    regression = F.smooth_l1_loss(
        normalized_predicted,
        normalized_target,
        beta=0.5,
    )
    cosine = F.cosine_similarity(predicted_flat, target_flat, dim=1, eps=1e-8)
    cosine_loss = (1.0 - cosine).mean()
    target_norm = target_flat.norm(dim=1).clamp_min(1e-8)
    magnitude_ratio = predicted_flat.norm(dim=1) / target_norm
    return regression + cosine_loss, cosine.mean(), magnitude_ratio.mean()


def latent_target_ranking_loss(
    predicted_first: Tensor,
    predicted_second: Tensor,
    target_first: Tensor,
    target_second: Tensor,
    *,
    margin: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Require each caption-conditioned latent to prefer its paired VAE target."""
    if margin < 0.0:
        raise ValueError("margin must be non-negative")
    if predicted_first.shape != predicted_second.shape:
        raise ValueError("predicted latent pairs must have identical shapes")
    if target_first.shape != target_second.shape or target_first.shape != predicted_first.shape:
        raise ValueError("predicted and target latent pairs must have identical shapes")

    target_delta = (target_first - target_second).flatten(start_dim=1)
    scale = target_delta.square().mean(dim=1).clamp_min(1e-6)

    def normalized_distance(left: Tensor, right: Tensor) -> Tensor:
        squared = (left - right).flatten(start_dim=1).square().mean(dim=1)
        return squared / scale

    first_correct = normalized_distance(predicted_first, target_first)
    first_wrong = normalized_distance(predicted_first, target_second)
    second_correct = normalized_distance(predicted_second, target_second)
    second_wrong = normalized_distance(predicted_second, target_first)
    correct = 0.5 * (first_correct + second_correct)
    wrong = 0.5 * (first_wrong + second_wrong)
    ranking = F.relu(margin + correct - wrong)
    loss = ranking.mean() + 0.25 * correct.mean()
    return loss, correct.mean(), (wrong - correct).mean()


def sinkhorn_causal_gate(
    delta_cosine: Tensor,
    *,
    start: float,
    full: float,
) -> Tensor:
    """Delay geometry pressure until text causes a measurable latent shape change."""
    if not 0.0 <= start < full <= 1.0:
        raise ValueError("require 0 <= start < full <= 1")
    return ((delta_cosine.detach() - start) / (full - start)).clamp(0.0, 1.0)


def merge_primary_and_protected_gradients(
    primary_gradients: Sequence[Tensor | None],
    protected_gradients: Sequence[Tensor | None],
    *,
    eps: float = 1e-12,
) -> tuple[tuple[Tensor | None, ...], float, bool]:
    """Project only the primary gradient component that opposes the protected task."""
    if len(primary_gradients) != len(protected_gradients):
        raise ValueError("gradient sequences must have identical lengths")
    reference = next(
        (
            gradient
            for gradient in [*primary_gradients, *protected_gradients]
            if gradient is not None
        ),
        None,
    )
    if reference is None:
        return tuple([None] * len(primary_gradients)), 0.0, False

    dot = reference.new_zeros(())
    primary_sq = reference.new_zeros(())
    protected_sq = reference.new_zeros(())
    for primary, protected in zip(primary_gradients, protected_gradients, strict=True):
        if primary is not None:
            primary_sq = primary_sq + primary.square().sum()
        if protected is not None:
            protected_sq = protected_sq + protected.square().sum()
        if primary is not None and protected is not None:
            dot = dot + (primary * protected).sum()

    denominator = (primary_sq.sqrt() * protected_sq.sqrt()).clamp_min(eps)
    cosine = float((dot / denominator).item())
    active = bool((dot < 0.0).item()) and float(protected_sq.item()) > eps
    coefficient = dot / protected_sq.clamp_min(eps) if active else dot.new_zeros(())

    merged: list[Tensor | None] = []
    for primary, protected in zip(primary_gradients, protected_gradients, strict=True):
        projected_primary = primary
        if active and primary is not None and protected is not None:
            projected_primary = primary - coefficient * protected
        if projected_primary is None:
            merged.append(protected)
        elif protected is None:
            merged.append(projected_primary)
        else:
            merged.append(projected_primary + protected)
    return tuple(merged), cosine, active


def latent_svd_diagnostics(shape_deltas: Tensor) -> dict[str, float]:
    """Summarize how concentrated square-vs-circle information is in latent space."""
    if shape_deltas.ndim < 2 or shape_deltas.shape[0] <= 1:
        raise ValueError("shape_deltas must contain at least two samples")
    matrix = shape_deltas.flatten(start_dim=1)
    matrix = matrix - matrix.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(matrix)
    energy = singular_values.square()
    total = energy.sum().clamp_min(1e-12)
    fractions = energy / total
    top1 = fractions[:1].sum()
    top4 = fractions[: min(4, fractions.numel())].sum()
    top8 = fractions[: min(8, fractions.numel())].sum()
    entropy = -(fractions * fractions.clamp_min(1e-12).log()).sum()
    return {
        "shape_svd_top1_energy": float(top1.item()),
        "shape_svd_top4_energy": float(top4.item()),
        "shape_svd_top8_energy": float(top8.item()),
        "shape_svd_effective_rank": float(torch.exp(entropy).item()),
    }


def gradient_cosine_matrix(
    losses: Mapping[str, Tensor],
    parameters: Sequence[Tensor],
) -> dict[str, float]:
    """Measure pairwise cosine similarity between task gradients."""
    params = [parameter for parameter in parameters if parameter.requires_grad]
    if not params:
        raise ValueError("at least one trainable parameter is required")
    names = list(losses)
    gradient_sets: dict[str, tuple[Tensor | None, ...]] = {}
    for name in names:
        gradient_sets[name] = torch.autograd.grad(
            losses[name],
            params,
            retain_graph=True,
            allow_unused=True,
        )

    result: dict[str, float] = {}
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            dot = losses[left_name].new_zeros(())
            left_sq = losses[left_name].new_zeros(())
            right_sq = losses[left_name].new_zeros(())
            for left_grad, right_grad in zip(
                gradient_sets[left_name], gradient_sets[right_name], strict=True
            ):
                if left_grad is not None:
                    left_sq = left_sq + left_grad.square().sum()
                if right_grad is not None:
                    right_sq = right_sq + right_grad.square().sum()
                if left_grad is not None and right_grad is not None:
                    dot = dot + (left_grad * right_grad).sum()
            denominator = (left_sq.sqrt() * right_sq.sqrt()).clamp_min(1e-12)
            result[f"grad_cos_{left_name}_{right_name}"] = float((dot / denominator).item())
    return result


def _centered_shape_measure(video: Tensor, size: int) -> tuple[Tensor, Tensor]:
    if video.ndim != 5 or video.shape[1] != 3:
        raise ValueError("video must have shape [B, 3, T, H, W]")
    signal = ((video + 1.0) * 0.5).clamp(0.0, 1.0).amax(dim=1)
    batch, frames, height, width = signal.shape
    y_grid, x_grid = torch.meshgrid(
        torch.arange(height, device=video.device, dtype=video.dtype),
        torch.arange(width, device=video.device, dtype=video.dtype),
        indexing="ij",
    )
    center_weights = signal.square()
    mass = center_weights.sum(dim=(2, 3)).clamp_min(1e-6)
    center_x = (center_weights * x_grid).sum(dim=(2, 3)) / mass
    center_y = (center_weights * y_grid).sum(dim=(2, 3)) / mass

    shape_size = controlled_motion_shape_size(size)
    offsets = torch.arange(shape_size, device=video.device, dtype=video.dtype)
    offsets = offsets - (shape_size - 1) / 2.0
    patch_y, patch_x = torch.meshgrid(offsets, offsets, indexing="ij")
    sample_x = center_x.detach().reshape(-1, 1, 1) + patch_x
    sample_y = center_y.detach().reshape(-1, 1, 1) + patch_y
    normalized_x = 2.0 * sample_x / max(width - 1, 1) - 1.0
    normalized_y = 2.0 * sample_y / max(height - 1, 1) - 1.0
    sampling_grid = torch.stack((normalized_x, normalized_y), dim=-1)
    flat_signal = signal.reshape(batch * frames, 1, height, width)
    patch = F.grid_sample(
        flat_signal,
        sampling_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[:, 0]
    weights = patch.flatten(start_dim=1).clamp_min(1e-6)
    weights = weights / weights.sum(dim=1, keepdim=True)

    coords = torch.linspace(-1.0, 1.0, shape_size, device=video.device, dtype=video.dtype)
    coord_y, coord_x = torch.meshgrid(coords, coords, indexing="ij")
    points = torch.stack((coord_x.flatten(), coord_y.flatten()), dim=-1)
    points = points.unsqueeze(0).expand(batch * frames, -1, -1).contiguous()
    return weights, points


def _canonical_shape_weights(
    captions: list[str],
    *,
    frames: int,
    size: int,
    device: torch.device,
    dtype: torch.dtype,
    opposite: bool,
) -> Tensor:
    shape_size = controlled_motion_shape_size(size)
    coords = torch.arange(shape_size, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    center = (shape_size - 1) / 2.0
    radius = max(1.0, shape_size / 2.0)
    circle = ((xx - center).square() + (yy - center).square() <= radius**2).to(dtype)
    square = torch.ones_like(circle)
    weights: list[Tensor] = []
    for caption in captions:
        is_square = " square " in caption
        is_circle = " circle " in caption
        if not is_square and not is_circle:
            raise ValueError(f"caption does not contain a shape term: {caption}")
        choose_square = is_square if not opposite else is_circle
        target = square if choose_square else circle
        normalized = target.flatten().clamp_min(1e-6)
        normalized = normalized / normalized.sum()
        weights.extend([normalized] * frames)
    return torch.stack(weights)


@lru_cache(maxsize=16)
def _sinkhorn_criterion(blur: float) -> Any:
    try:
        from geomloss import SamplesLoss  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "M1 Sinkhorn research loss requires the research extra: "
            "uv add --optional research geomloss==0.3.1"
        ) from exc
    return SamplesLoss(
        loss="sinkhorn",
        p=2,
        blur=blur,
        debias=True,
        backend="tensorized",
    )


def sinkhorn_shape_geometry_loss(
    video: Tensor,
    captions: list[str],
    *,
    size: int,
    blur: float,
    margin: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Match centered geometry and rank the correct shape ahead of its counterfactual."""
    if video.shape[0] != len(captions):
        raise ValueError("video/caption counts must match")
    if blur <= 0.0 or margin < 0.0:
        raise ValueError("invalid Sinkhorn shape settings")
    predicted_weights, points = _centered_shape_measure(video, size)
    frames = int(video.shape[2])
    correct_weights = _canonical_shape_weights(
        captions,
        frames=frames,
        size=size,
        device=video.device,
        dtype=video.dtype,
        opposite=False,
    )
    opposite_weights = _canonical_shape_weights(
        captions,
        frames=frames,
        size=size,
        device=video.device,
        dtype=video.dtype,
        opposite=True,
    )
    criterion = _sinkhorn_criterion(float(blur))
    correct = cast(Tensor, criterion(predicted_weights, points, correct_weights, points))
    opposite = cast(Tensor, criterion(predicted_weights, points, opposite_weights, points))
    ranking = F.relu(margin + correct - opposite)
    return correct.mean() + ranking.mean(), correct.mean(), (opposite - correct).mean()


def constraint_rank_score(constraint_violation: float, tie_break_score: float) -> float:
    """Feasibility-first scalar: gate violation dominates the bounded tie-break term."""
    if constraint_violation < 0.0 or tie_break_score < 0.0:
        raise ValueError("constraint and tie-break scores must be non-negative")
    bounded_tie_break = tie_break_score / (1.0 + tie_break_score)
    return constraint_violation + bounded_tie_break * 1e-6
