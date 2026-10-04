from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from vexa_video.config import load_config
from vexa_video.training.checkpoint import build_training_checkpoint
from vexa_video.training.trainer import build_stage_b_components
from vexa_video.utils.seed import seed_everything


def prepare_checkpoint(*, config: str, source: str, output: str) -> Path:
    cfg = load_config(config)
    seed_everything(cfg.seed)
    device = torch.device("cpu")
    components = build_stage_b_components(cfg, device=device)

    raw = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise ValueError("source checkpoint must contain a dictionary payload")
    payload = cast(dict[str, Any], raw)
    if payload.get("stage") != "stage-b-generative-synthetic-motion":
        raise ValueError("source checkpoint is not an M1 Stage-B checkpoint")
    if payload.get("phase") != "vae_ready" or int(payload.get("diffusion_step", -1)) != 0:
        raise ValueError("source checkpoint must be VAE-ready with diffusion_step=0")
    if payload.get("config") != asdict(cfg):
        raise ValueError("source checkpoint config does not match the requested config")

    models = cast(dict[str, Any], payload["models"])
    optimizers = cast(dict[str, Any], payload["optimizers"])
    components.vae.load_state_dict(models["vae"])
    components.text_encoder.load_state_dict(models["text_encoder"])
    incompat = components.dit.load_state_dict(models["dit"], strict=False)
    if incompat.unexpected_keys:
        raise ValueError(f"unexpected legacy DiT keys: {incompat.unexpected_keys}")
    if not incompat.missing_keys or not all(
        key.startswith("text_cross_attention.") for key in incompat.missing_keys
    ):
        raise ValueError(f"unexpected new DiT keys: {incompat.missing_keys}")

    vae_optimizer = torch.optim.AdamW(
        components.vae.parameters(),
        lr=cfg.m1.vae_learning_rate,
    )
    vae_optimizer.load_state_dict(optimizers["vae"])
    diffusion_optimizer = torch.optim.AdamW(
        [*components.text_encoder.parameters(), *components.dit.parameters()],
        lr=cfg.m1.learning_rate,
    )

    checkpoint = build_training_checkpoint(
        vae=components.vae,
        text_encoder=components.text_encoder,
        dit=components.dit,
        vae_optimizer=vae_optimizer,
        diffusion_optimizer=diffusion_optimizer,
        global_step=int(payload.get("global_step", payload.get("vae_step", 0))),
        vae_step=int(payload["vae_step"]),
        diffusion_step=0,
        config=cast(dict[str, Any], payload["config"]),
        metrics=cast(dict[str, float], payload.get("metrics", {})),
        best_validation_score=float("inf"),
        phase="vae_ready",
        data_generator_state=cast(Tensor, payload["data_generator_state"]).cpu(),
        metadata={
            "initialization": "legacy-vae-ready-plus-learned-cross-attention",
            "source_checkpoint": str(source),
            "new_dit_keys": sorted(incompat.missing_keys),
            "cross_attention_gate_init": 0.05,
            "cross_attention_qk_normalization": "per-head-rms",
            "cross_attention_kernel": "torch-scaled-dot-product-attention",
        },
    )
    checkpoint["torch_rng_state"] = cast(Tensor, payload["torch_rng_state"]).cpu()
    if "cuda_rng_state_all" in payload:
        checkpoint["cuda_rng_state_all"] = payload["cuda_rng_state_all"]

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the frozen M1 VAE checkpoint for the learned cross-attention DiT."
    )
    parser.add_argument("--config", default="configs/tiny.toml")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    destination = prepare_checkpoint(config=args.config, source=args.source, output=args.output)
    digest = torch.load(destination, map_location="cpu", weights_only=False)
    print(f"prepared={destination}")
    print(f"phase={digest['phase']}")
    print(f"vae_step={digest['vae_step']}")
    print(f"diffusion_step={digest['diffusion_step']}")
    print(f"initialization={digest['metadata']['initialization']}")


if __name__ == "__main__":
    main()
