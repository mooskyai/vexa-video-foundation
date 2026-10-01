from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import torch
from torch import Tensor, nn


def build_checkpoint(
    *,
    model: nn.Module,
    step: int,
    config: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "step": step,
        "model": model.state_dict(),
        "config": config,
        "metadata": metadata or {},
        "torch_rng_state": torch.get_rng_state(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return payload


def build_stage_b_checkpoint(
    *,
    vae: nn.Module,
    text_encoder: nn.Module,
    dit: nn.Module,
    vae_optimizer: torch.optim.Optimizer,
    diffusion_optimizer: torch.optim.Optimizer,
    global_step: int,
    vae_step: int,
    diffusion_step: int,
    config: dict[str, Any],
    metrics: dict[str, float],
    best_validation_score: float,
    phase: str,
    data_generator_state: Tensor,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the resumable composite checkpoint used by M1 Stage-B."""
    payload: dict[str, Any] = {
        "schema_version": 2,
        "created_at": datetime.now(UTC).isoformat(),
        "milestone": "M1",
        "stage": "stage-b-generative-synthetic-motion",
        "phase": phase,
        "global_step": global_step,
        "vae_step": vae_step,
        "diffusion_step": diffusion_step,
        "models": {
            "vae": vae.state_dict(),
            "text_encoder": text_encoder.state_dict(),
            "dit": dit.state_dict(),
        },
        "optimizers": {
            "vae": vae_optimizer.state_dict(),
            "diffusion": diffusion_optimizer.state_dict(),
        },
        "config": config,
        "seed": int(config["seed"]),
        "dataset": {
            "name": "synthetic-stage-b-cardinal-v1",
            "train_split": "train",
            "validation_split": "validation",
            "test_split": "test",
            "speed_bucket": "medium",
        },
        "metrics": metrics,
        "best_validation_score": best_validation_score,
        "torch_rng_state": torch.get_rng_state(),
        "data_generator_state": data_generator_state,
        "metadata": metadata or {},
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return payload
