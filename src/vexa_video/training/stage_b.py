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
from vexa_video.data import COLORS, DIRECTIONS, StageBSyntheticDataset
from vexa_video.diffusion import LinearNoiseSchedule
from vexa_video.inference.sampler import guided_ddim_rollout
from vexa_video.models import ByteTokenizer, TinyVideoVAE, TransformerTextEncoder, VideoDiT
from vexa_video.training.checkpoint import build_stage_b_checkpoint


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
    semantic_direction_loss: float = 0.0
    semantic_color_loss: float = 0.0


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


def reconstruction_loss(reconstruction: Tensor, target: Tensor) -> Tensor:
    """Pixel MSE plus foreground-weighted MSE so background collapse cannot look healthy."""
    if reconstruction.shape != target.shape:
        raise ValueError("reconstruction and target must have identical shapes")
    pixel_loss = F.mse_loss(reconstruction, target)
    foreground = target.amax(dim=1, keepdim=True).gt(-0.9)
    expanded = foreground.expand_as(target).to(dtype=target.dtype)
    foreground_count = expanded.sum()
    if float(foreground_count.item()) == 0.0:
        return pixel_loss
    foreground_loss = ((reconstruction - target).square() * expanded).sum() / foreground_count
    return pixel_loss + foreground_loss


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


def _conditioning_predictions(
    *,
    components: StageBComponents,
    noisy_latents: Tensor,
    timesteps: Tensor,
    captions: list[str],
    cfg: ProjectConfig,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    color_counterfactuals = [_counterfactual_caption(caption, COLORS) for caption in captions]
    direction_counterfactuals = [
        _counterfactual_caption(caption, DIRECTIONS) for caption in captions
    ]
    unconditional = [""] * len(captions)
    all_captions = [*captions, *color_counterfactuals, *direction_counterfactuals, *unconditional]
    tokens = components.tokenizer.batch(all_captions, cfg.text.max_length, device=device)
    text_tokens = components.text_encoder(tokens.input_ids, tokens.attention_mask)
    predicted = components.dit(
        torch.cat((noisy_latents, noisy_latents, noisy_latents, noisy_latents), dim=0),
        timesteps.repeat(4),
        text_tokens,
        tokens.attention_mask,
    )
    return cast(tuple[Tensor, Tensor, Tensor, Tensor], predicted.chunk(4, dim=0))


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


def _motion_target_loss(motion: Tensor, target: Tensor, minimum_motion: float) -> Tensor:
    alignment = 1.0 - F.cosine_similarity(motion, target, dim=-1, eps=1e-6)
    magnitude = motion.norm(dim=-1)
    return (alignment + F.relu(minimum_motion - magnitude)).mean()


def _sampler_aligned_direction_loss(
    *,
    components: StageBComponents,
    clean_latents: Tensor,
    captions: list[str],
    cfg: ProjectConfig,
    device: torch.device,
    initial_noise: Tensor,
) -> Tensor:
    batch = min(cfg.m1.rollout_batch_size, clean_latents.shape[0], len(captions))
    if batch <= 0:
        return clean_latents.new_zeros(())

    base_captions = captions[:batch]
    rollout_captions = [
        _caption_with_direction(caption, direction)
        for caption in base_captions
        for direction in DIRECTIONS
    ]
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
    rollout_noise = initial_noise[:batch].repeat_interleave(len(DIRECTIONS), dim=0)
    rollout_latents = guided_ddim_rollout(
        latents=rollout_noise,
        dit=components.dit,
        schedule=components.schedule,
        conditional_text=conditional_text,
        conditional_mask=conditional_tokens.attention_mask,
        unconditional_text=unconditional_text,
        unconditional_mask=unconditional_tokens.attention_mask,
        sampling_timesteps=components.schedule.sampling_timesteps(
            cfg.m1.rollout_steps,
            device=device,
        ),
        guidance_scale=cfg.m1.guidance_scale,
    )
    rollout_video = components.vae.decode(rollout_latents)
    rollout_motion, _ = _soft_video_features(rollout_video)
    direction_targets = torch.tensor(
        [_DIRECTION_TARGETS[direction] for _ in base_captions for direction in DIRECTIONS],
        device=device,
        dtype=clean_latents.dtype,
    )
    return _motion_target_loss(
        rollout_motion,
        direction_targets,
        cfg.m1.semantic_min_motion,
    )


def _semantic_conditioning_losses(
    *,
    components: StageBComponents,
    clean_latents: Tensor,
    captions: list[str],
    cfg: ProjectConfig,
    device: torch.device,
    noise: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    batch = clean_latents.shape[0]
    semantic_timesteps = torch.full(
        (batch,), cfg.m1.semantic_timestep, device=device, dtype=torch.long
    )
    semantic_noise = torch.randn_like(clean_latents) if noise is None else noise
    noisy = components.schedule.add_noise(clean_latents, semantic_noise, semantic_timesteps)
    conditioned, color_counterfactual, direction_counterfactual, _ = _conditioning_predictions(
        components=components,
        noisy_latents=noisy,
        timesteps=semantic_timesteps,
        captions=captions,
        cfg=cfg,
        device=device,
    )
    color_captions = [_counterfactual_caption(caption, COLORS) for caption in captions]
    direction_captions = [_counterfactual_caption(caption, DIRECTIONS) for caption in captions]
    predicted_clean = torch.cat(
        [
            components.schedule.predict_clean(noisy, conditioned, cfg.m1.semantic_timestep),
            components.schedule.predict_clean(
                noisy, color_counterfactual, cfg.m1.semantic_timestep
            ),
            components.schedule.predict_clean(
                noisy, direction_counterfactual, cfg.m1.semantic_timestep
            ),
        ],
        dim=0,
    )
    decoded = components.vae.decode(predicted_clean)
    correct_video, color_video, direction_video = decoded.chunk(3, dim=0)
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
    rollout_direction_loss = _sampler_aligned_direction_loss(
        components=components,
        clean_latents=clean_latents,
        captions=captions,
        cfg=cfg,
        device=device,
        initial_noise=semantic_noise,
    )
    direction_loss = direction_loss + cfg.m1.rollout_direction_weight * rollout_direction_loss
    return direction_loss, color_loss


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
    recon_loss = reconstruction_loss(reconstruction, videos)
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
    cfg: ProjectConfig,
    device: torch.device,
) -> StageBStepMetrics:
    components.text_encoder.train()
    components.dit.train()
    components.vae.eval()
    optimizer.zero_grad(set_to_none=True)

    with torch.no_grad():
        reconstruction, latents = components.vae(videos)
        recon_loss = reconstruction_loss(reconstruction, videos)
    timesteps = _sample_training_timesteps(
        batch_size=videos.shape[0],
        timesteps=components.schedule.timesteps,
        high_noise_fraction=cfg.m1.high_noise_conditioning_fraction,
        device=device,
    )
    noise = torch.randn_like(latents)
    noisy_latents = components.schedule.add_noise(latents, noise, timesteps)
    conditioned, color_counterfactual, direction_counterfactual, unconditional = (
        _conditioning_predictions(
            components=components,
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            captions=captions,
            cfg=cfg,
            device=device,
        )
    )
    conditioned_error = _per_sample_mse(conditioned, noise)
    color_counterfactual_error = _per_sample_mse(color_counterfactual, noise)
    direction_counterfactual_error = _per_sample_mse(direction_counterfactual, noise)
    unconditional_error = _per_sample_mse(unconditional, noise)
    color_gap = color_counterfactual_error - conditioned_error
    direction_gap = direction_counterfactual_error - conditioned_error
    contrast_loss = 0.5 * (
        F.relu(cfg.m1.prompt_contrast_margin - color_gap).mean()
        + F.relu(cfg.m1.prompt_contrast_margin - direction_gap).mean()
    )
    diffusion_loss = conditioned_error.mean()
    unconditional_loss = unconditional_error.mean()
    counterfactual_loss = 0.5 * (
        color_counterfactual_error.mean() + direction_counterfactual_error.mean()
    )
    semantic_direction_loss, semantic_color_loss = _semantic_conditioning_losses(
        components=components,
        clean_latents=latents,
        captions=captions,
        cfg=cfg,
        device=device,
    )
    optimization_loss = (
        cfg.m1.diffusion_weight * diffusion_loss
        + cfg.m1.unconditional_loss_weight * unconditional_loss
        + cfg.m1.prompt_contrast_weight * contrast_loss
        + cfg.m1.semantic_direction_weight * semantic_direction_loss
        + cfg.m1.semantic_color_weight * semantic_color_loss
    )
    torch.autograd.backward(optimization_loss)
    trainable = [*components.text_encoder.parameters(), *components.dit.parameters()]
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
        semantic_direction_loss=float(semantic_direction_loss.detach().item()),
        semantic_color_loss=float(semantic_color_loss.detach().item()),
    )


def _validate_reconstruction(
    *,
    components: StageBComponents,
    dataset: StageBSyntheticDataset,
    batch_size: int,
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
            loss = reconstruction_loss(reconstruction, videos)
            total += float(loss.item()) * len(indices)
            count += len(indices)
    return total / count


def validate_stage_b(
    *,
    components: StageBComponents,
    dataset: StageBSyntheticDataset,
    cfg: ProjectConfig,
    device: torch.device,
) -> tuple[float, float, float, float, float, float]:
    components.vae.eval()
    components.text_encoder.eval()
    components.dit.eval()
    generator = torch.Generator(device=device).manual_seed(cfg.seed + 7_001)
    reconstruction_total = 0.0
    diffusion_total = 0.0
    color_gap_total = 0.0
    direction_gap_total = 0.0
    semantic_direction_total = 0.0
    semantic_color_total = 0.0
    count = 0

    with torch.no_grad():
        for start in range(0, len(dataset), cfg.m1.batch_size):
            indices = list(range(start, min(start + cfg.m1.batch_size, len(dataset))))
            videos, captions = _sample_batch(dataset, indices, device=device)
            reconstruction, latents = components.vae(videos)
            recon_loss = reconstruction_loss(reconstruction, videos)
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
            conditioned, color_counterfactual, direction_counterfactual, _ = (
                _conditioning_predictions(
                    components=components,
                    noisy_latents=noisy,
                    timesteps=timesteps,
                    captions=captions,
                    cfg=cfg,
                    device=device,
                )
            )
            conditioned_error = _per_sample_mse(conditioned, noise)
            color_error = _per_sample_mse(color_counterfactual, noise)
            direction_error = _per_sample_mse(direction_counterfactual, noise)
            diff_loss = conditioned_error.mean()
            color_gap = (color_error - conditioned_error).mean()
            direction_gap = (direction_error - conditioned_error).mean()
            semantic_noise = torch.randn(
                latents.shape,
                generator=generator,
                device=device,
                dtype=latents.dtype,
            )
            semantic_direction, semantic_color = _semantic_conditioning_losses(
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
            semantic_direction_total += float(semantic_direction.item()) * len(indices)
            semantic_color_total += float(semantic_color.item()) * len(indices)
            count += len(indices)
    return (
        reconstruction_total / count,
        diffusion_total / count,
        color_gap_total / count,
        direction_gap_total / count,
        semantic_direction_total / count,
        semantic_color_total / count,
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
    stored_config = payload.get("config", {})
    stored_m1 = stored_config.get("m1", {}) if isinstance(stored_config, dict) else {}
    if not isinstance(stored_m1, dict) or "prompt_contrast_weight" not in stored_m1:
        best_validation_score = math.inf
        print("m1_stage_b resume_revision=conditioning-v2 reset_best_validation_score=true")
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
    checkpoint = build_stage_b_checkpoint(
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
                "prompt-gap + semantic penalties"
            ),
            "curriculum": "one object; four directions; four colors; square/circle; medium speed",
            "conditioning": (
                "byte-text only; token attention + text counterfactuals + "
                "decoded semantic motion/color losses + CFG"
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
) -> StageBTrainResult:
    if cfg.data.height != cfg.data.width:
        raise ValueError("M1 Stage-B currently requires square video dimensions")
    target_steps = steps if steps is not None else cfg.m1.max_steps
    if target_steps <= 0:
        raise ValueError("steps must be positive")
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
        if not isinstance(resume_m1, dict) or "semantic_direction_weight" not in resume_m1:
            best_validation_score = math.inf
            print("m1_stage_b resume_revision=semantic-v3 reset_best_validation_score=true")
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
        last_metrics = diffusion_train_step(
            components=components,
            optimizer=diffusion_optimizer,
            videos=videos,
            captions=captions,
            cfg=cfg,
            device=device,
        )
        if diffusion_step == 1 or diffusion_step % cfg.m1.log_every == 0:
            print(
                f"m1_stage_b phase=diffusion step={diffusion_step} "
                f"reconstruction={last_metrics.reconstruction_loss:.6f} "
                f"diffusion={last_metrics.diffusion_loss:.6f} "
                f"contrast={last_metrics.prompt_contrast_loss:.6f} "
                f"color_gap={last_metrics.color_prompt_gap:.6f} "
                f"direction_gap={last_metrics.direction_prompt_gap:.6f} "
                f"semantic_direction={last_metrics.semantic_direction_loss:.6f} "
                f"semantic_color={last_metrics.semantic_color_loss:.6f} "
                f"total={last_metrics.total_loss:.6f} "
                f"grad_norm={last_metrics.gradient_norm:.6f}"
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
                validation_semantic_direction,
                validation_semantic_color,
            ) = validate_stage_b(
                components=components,
                dataset=validation_dataset,
                cfg=cfg,
                device=device,
            )
            conditioning_penalty = 0.5 * (
                max(0.0, cfg.m1.prompt_contrast_margin - validation_color_gap)
                + max(0.0, cfg.m1.prompt_contrast_margin - validation_direction_gap)
            )
            validation_score = (
                cfg.m1.reconstruction_weight * validation_reconstruction
                + cfg.m1.diffusion_weight * validation_diffusion
                + cfg.m1.prompt_contrast_weight * conditioning_penalty
                + cfg.m1.semantic_direction_weight * validation_semantic_direction
                + cfg.m1.semantic_color_weight * validation_semantic_color
            )
            checkpoint_metrics = {
                **asdict(last_metrics),
                "validation_reconstruction_loss": validation_reconstruction,
                "validation_diffusion_loss": validation_diffusion,
                "validation_color_prompt_gap": validation_color_gap,
                "validation_direction_prompt_gap": validation_direction_gap,
                "validation_conditioning_penalty": conditioning_penalty,
                "validation_semantic_direction_loss": validation_semantic_direction,
                "validation_semantic_color_loss": validation_semantic_color,
                "validation_score": validation_score,
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
                f"semantic_direction={validation_semantic_direction:.6f} "
                f"semantic_color={validation_semantic_color:.6f} "
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
    """Load Stage-B model states for deterministic evaluation/generation."""
    raw = torch.load(Path(checkpoint), map_location=device, weights_only=False)
    if not isinstance(raw, dict):
        raise ValueError("Stage-B checkpoint must contain a dictionary payload")
    payload = cast(dict[str, Any], raw)
    if payload.get("stage") != "stage-b-generative-synthetic-motion":
        raise ValueError("checkpoint is not an M1 Stage-B checkpoint")
    models = cast(dict[str, Any], payload["models"])
    components.vae.load_state_dict(models["vae"])
    components.text_encoder.load_state_dict(models["text_encoder"])
    components.dit.load_state_dict(models["dit"])
    return payload
