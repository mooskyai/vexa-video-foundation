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
