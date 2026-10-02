from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class DataConfig:
    frames: int
    height: int
    width: int


@dataclass(frozen=True, slots=True)
class TextConfig:
    vocab_size: int
    max_length: int
    d_model: int
    layers: int
    heads: int
    ff_mult: int


@dataclass(frozen=True, slots=True)
class VAEConfig:
    in_channels: int
    latent_channels: int
    base_channels: int
    spatial_downsample: int
    temporal_downsample: int


@dataclass(frozen=True, slots=True)
class DiTConfig:
    hidden_size: int
    layers: int
    heads: int
    patch_t: int
    patch_h: int
    patch_w: int


@dataclass(frozen=True, slots=True)
class DiffusionConfig:
    timesteps: int
    beta_start: float
    beta_end: float


@dataclass(frozen=True, slots=True)
class M1Config:
    probe_train_samples: int = 2_048
    probe_eval_samples: int = 256
    probe_steps: int = 250
    probe_batch_size: int = 32
    probe_learning_rate: float = 1e-3
    probe_direction_gate: float = 0.95
    probe_color_gate: float = 0.95
    train_samples: int = 2_048
    validation_samples: int = 256
    batch_size: int = 8
    learning_rate: float = 3e-4
    vae_learning_rate: float = 1e-3
    vae_warmup_steps: int = 400
    vae_reconstruction_gate: float = 0.20
    vae_silhouette_weight: float = 1.0
    vae_visual_validation_samples: int = 64
    vae_semantic_accuracy_gate: float = 0.95
    vae_object_like_frame_gate: float = 0.95
    vae_persistent_video_gate: float = 0.95
    vae_foreground_area_ratio_min: float = 0.75
    vae_foreground_area_ratio_max: float = 1.50
    max_steps: int = 2_000
    checkpoint_every: int = 100
    log_every: int = 20
    reconstruction_weight: float = 1.0
    diffusion_weight: float = 1.0
    grad_clip: float = 1.0
    sampling_steps: int = 50
    eval_samples: int = 64
    eval_batch_size: int = 4
    static_motion_threshold: float = 0.02
    prompt_contrast_weight: float = 2.0
    prompt_contrast_margin: float = 0.02
    unconditional_loss_weight: float = 0.1
    high_noise_conditioning_fraction: float = 0.5
    semantic_timestep: int = 500
    semantic_direction_weight: float = 2.0
    semantic_color_weight: float = 0.5
    semantic_shape_weight: float = 1.0
    semantic_min_motion: float = 0.10
    rollout_steps: int = 4
    rollout_batch_size: int = 1
    rollout_direction_weight: float = 1.0
    rollout_shape_weight: float = 1.0
    full_rollout_steps: int = 50
    full_rollout_every: int = 4
    full_rollout_direction_weight: float = 0.25
    full_rollout_color_weight: float = 1.0
    full_rollout_shape_weight: float = 1.0
    full_rollout_min_motion: float = 0.35
    validation_generation_samples: int = 32
    validation_generation_direction_weight: float = 1.0
    validation_generation_other_weight: float = 1.0
    guidance_scale: float = 3.0
    generation_direction_gate: float = 0.75
    generation_color_gate: float = 0.75
    generation_shape_gate: float = 0.75
    generation_static_rate_gate: float = 0.10
    generation_mean_motion_gate: float = 0.02
    generation_object_like_frame_gate: float = 0.90
    generation_persistent_video_gate: float = 0.90
    generation_foreground_area_ratio_min: float = 0.75
    generation_foreground_area_ratio_max: float = 1.50


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    seed: int
    data: DataConfig
    text: TextConfig
    vae: VAEConfig
    dit: DiTConfig
    diffusion: DiffusionConfig
    m1: M1Config


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"missing [{name}] section")
    return value


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    project = _section(raw, "project")
    data_raw = _section(raw, "data")
    text_raw = _section(raw, "text")
    vae_raw = _section(raw, "vae")
    dit_raw = _section(raw, "dit")
    diffusion_raw = _section(raw, "diffusion")
    m1_raw = raw.get("m1", {})
    if not isinstance(m1_raw, dict):
        raise ValueError("[m1] must be a TOML table")

    cfg = ProjectConfig(
        seed=int(project["seed"]),
        data=DataConfig(**data_raw),
        text=TextConfig(**text_raw),
        vae=VAEConfig(**vae_raw),
        dit=DiTConfig(**dit_raw),
        diffusion=DiffusionConfig(**diffusion_raw),
        m1=M1Config(**m1_raw),
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: ProjectConfig) -> None:
    if cfg.data.frames <= 0 or cfg.data.height <= 0 or cfg.data.width <= 0:
        raise ValueError("video dimensions must be positive")
    if cfg.data.height % cfg.vae.spatial_downsample != 0:
        raise ValueError("height must be divisible by vae.spatial_downsample")
    if cfg.data.width % cfg.vae.spatial_downsample != 0:
        raise ValueError("width must be divisible by vae.spatial_downsample")
    if cfg.data.frames % cfg.vae.temporal_downsample != 0:
        raise ValueError("frames must be divisible by vae.temporal_downsample")
    if cfg.text.d_model % cfg.text.heads != 0:
        raise ValueError("text.d_model must be divisible by text.heads")
    if cfg.dit.hidden_size % cfg.dit.heads != 0:
        raise ValueError("dit.hidden_size must be divisible by dit.heads")
    if cfg.m1.probe_train_samples < cfg.m1.probe_batch_size:
        raise ValueError("m1.probe_train_samples must be >= m1.probe_batch_size")
    if cfg.m1.probe_eval_samples <= 0 or cfg.m1.probe_steps <= 0:
        raise ValueError("m1 probe samples/steps must be positive")
    if cfg.m1.probe_batch_size <= 0 or cfg.m1.probe_learning_rate <= 0:
        raise ValueError("m1 probe batch size/learning rate must be positive")
    if not 0.0 < cfg.m1.probe_direction_gate <= 1.0:
        raise ValueError("m1.probe_direction_gate must be in (0, 1]")
    if not 0.0 < cfg.m1.probe_color_gate <= 1.0:
        raise ValueError("m1.probe_color_gate must be in (0, 1]")
    if cfg.m1.train_samples < cfg.m1.batch_size:
        raise ValueError("m1.train_samples must be >= m1.batch_size")
    if cfg.m1.validation_samples <= 0 or cfg.m1.batch_size <= 0:
        raise ValueError("m1 validation samples/batch size must be positive")
    if cfg.m1.learning_rate <= 0 or cfg.m1.vae_learning_rate <= 0:
        raise ValueError("m1 learning rates must be positive")
    if cfg.m1.vae_warmup_steps <= 0 or cfg.m1.max_steps <= 0:
        raise ValueError("m1 VAE warmup/max steps must be positive")
    if cfg.m1.vae_reconstruction_gate <= 0:
        raise ValueError("m1.vae_reconstruction_gate must be positive")
    if cfg.m1.vae_silhouette_weight < 0:
        raise ValueError("m1.vae_silhouette_weight must be non-negative")
    if cfg.m1.vae_visual_validation_samples <= 0 or cfg.m1.vae_visual_validation_samples % 32 != 0:
        raise ValueError("m1.vae_visual_validation_samples must be a positive multiple of 32")
    if not 0.0 < cfg.m1.vae_semantic_accuracy_gate <= 1.0:
        raise ValueError("m1.vae_semantic_accuracy_gate must be in (0, 1]")
    if not 0.0 < cfg.m1.vae_object_like_frame_gate <= 1.0:
        raise ValueError("m1.vae_object_like_frame_gate must be in (0, 1]")
    if not 0.0 < cfg.m1.vae_persistent_video_gate <= 1.0:
        raise ValueError("m1.vae_persistent_video_gate must be in (0, 1]")
    if cfg.m1.vae_foreground_area_ratio_min <= 0:
        raise ValueError("m1.vae_foreground_area_ratio_min must be positive")
    if cfg.m1.vae_foreground_area_ratio_max <= cfg.m1.vae_foreground_area_ratio_min:
        raise ValueError(
            "m1.vae_foreground_area_ratio_max must exceed vae_foreground_area_ratio_min"
        )
    if cfg.m1.checkpoint_every <= 0 or cfg.m1.log_every <= 0:
        raise ValueError("m1 checkpoint/log intervals must be positive")
    if cfg.m1.reconstruction_weight < 0 or cfg.m1.diffusion_weight <= 0:
        raise ValueError("m1 loss weights must be non-negative with positive diffusion weight")
    if cfg.m1.grad_clip <= 0:
        raise ValueError("m1.grad_clip must be positive")
    if not 1 <= cfg.m1.sampling_steps <= cfg.diffusion.timesteps:
        raise ValueError("m1.sampling_steps must be in [1, diffusion.timesteps]")
    if cfg.m1.eval_samples <= 0 or cfg.m1.eval_samples % 16 != 0:
        raise ValueError("m1.eval_samples must be a positive multiple of 16")
    if cfg.m1.eval_batch_size <= 0:
        raise ValueError("m1.eval_batch_size must be positive")
    if cfg.m1.static_motion_threshold <= 0:
        raise ValueError("m1.static_motion_threshold must be positive")
    if cfg.m1.prompt_contrast_weight < 0:
        raise ValueError("m1.prompt_contrast_weight must be non-negative")
    if cfg.m1.prompt_contrast_margin <= 0:
        raise ValueError("m1.prompt_contrast_margin must be positive")
    if cfg.m1.unconditional_loss_weight < 0:
        raise ValueError("m1.unconditional_loss_weight must be non-negative")
    if not 0.0 <= cfg.m1.high_noise_conditioning_fraction <= 1.0:
        raise ValueError("m1.high_noise_conditioning_fraction must be in [0, 1]")
    if not 0 <= cfg.m1.semantic_timestep < cfg.diffusion.timesteps:
        raise ValueError("m1.semantic_timestep must be in [0, diffusion.timesteps)")
    if (
        cfg.m1.semantic_direction_weight < 0
        or cfg.m1.semantic_color_weight < 0
        or cfg.m1.semantic_shape_weight < 0
    ):
        raise ValueError("m1 semantic loss weights must be non-negative")
    if cfg.m1.semantic_min_motion <= 0:
        raise ValueError("m1.semantic_min_motion must be positive")
    if not 1 <= cfg.m1.rollout_steps <= cfg.diffusion.timesteps:
        raise ValueError("m1.rollout_steps must be in [1, diffusion.timesteps]")
    if cfg.m1.rollout_batch_size <= 0:
        raise ValueError("m1.rollout_batch_size must be positive")
    if cfg.m1.rollout_direction_weight < 0 or cfg.m1.rollout_shape_weight < 0:
        raise ValueError("m1 rollout weights must be non-negative")
    if not cfg.m1.rollout_steps <= cfg.m1.full_rollout_steps <= cfg.m1.sampling_steps:
        raise ValueError("m1.full_rollout_steps must be in [m1.rollout_steps, m1.sampling_steps]")
    if cfg.m1.full_rollout_every <= 0:
        raise ValueError("m1.full_rollout_every must be positive")
    if cfg.m1.full_rollout_direction_weight < 0 or cfg.m1.full_rollout_shape_weight < 0:
        raise ValueError("m1 full-rollout direction/shape weights must be non-negative")
    if (
        cfg.m1.validation_generation_samples <= 0
        or cfg.m1.validation_generation_samples % 16 != 0
        or cfg.m1.validation_generation_samples > cfg.m1.validation_samples
    ):
        raise ValueError(
            "m1.validation_generation_samples must be a positive multiple of 16 "
            "not exceeding m1.validation_samples"
        )
    if (
        cfg.m1.validation_generation_direction_weight < 0
        or cfg.m1.validation_generation_other_weight < 0
    ):
        raise ValueError("m1 validation-generation weights must be non-negative")
    if cfg.m1.full_rollout_color_weight < 0:
        raise ValueError("m1.full_rollout_color_weight must be non-negative")
    if cfg.m1.full_rollout_min_motion <= 0:
        raise ValueError("m1.full_rollout_min_motion must be positive")
    if cfg.m1.guidance_scale < 1.0:
        raise ValueError("m1.guidance_scale must be >= 1")
    if not 0.0 < cfg.m1.generation_direction_gate <= 1.0:
        raise ValueError("m1.generation_direction_gate must be in (0, 1]")
    if not 0.0 < cfg.m1.generation_color_gate <= 1.0:
        raise ValueError("m1.generation_color_gate must be in (0, 1]")
    if not 0.0 < cfg.m1.generation_shape_gate <= 1.0:
        raise ValueError("m1.generation_shape_gate must be in (0, 1]")
    if not 0.0 <= cfg.m1.generation_static_rate_gate <= 1.0:
        raise ValueError("m1.generation_static_rate_gate must be in [0, 1]")
    if cfg.m1.generation_mean_motion_gate <= 0:
        raise ValueError("m1.generation_mean_motion_gate must be positive")
    if not 0.0 < cfg.m1.generation_object_like_frame_gate <= 1.0:
        raise ValueError("m1.generation_object_like_frame_gate must be in (0, 1]")
    if not 0.0 < cfg.m1.generation_persistent_video_gate <= 1.0:
        raise ValueError("m1.generation_persistent_video_gate must be in (0, 1]")
    if cfg.m1.generation_foreground_area_ratio_min <= 0:
        raise ValueError("m1.generation_foreground_area_ratio_min must be positive")
    if cfg.m1.generation_foreground_area_ratio_max <= cfg.m1.generation_foreground_area_ratio_min:
        raise ValueError("m1.generation_foreground_area_ratio_max must exceed the minimum")
