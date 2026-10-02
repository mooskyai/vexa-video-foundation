from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from vexa_video.config import ProjectConfig, load_config
from vexa_video.data import SHAPES, StageBSyntheticDataset, SyntheticControl, render_motion_sample
from vexa_video.data.controlled_motion import controlled_motion_shape_size
from vexa_video.training.m1_diagnostics import counterfactual_response_metrics
from vexa_video.training.trainer import StageBComponents, build_stage_b_components


def _paired_batch(
    dataset: StageBSyntheticDataset,
    indices: list[int],
    *,
    device: torch.device,
) -> tuple[Tensor, list[str], Tensor, list[str], Tensor]:
    videos: list[Tensor] = []
    paired_videos: list[Tensor] = []
    captions: list[str] = []
    paired_captions: list[str] = []
    signs: list[float] = []
    for index in indices:
        sample = dataset.sample(index)
        opposite = SHAPES[(SHAPES.index(sample.control.shape) + 1) % len(SHAPES)]
        paired_control = SyntheticControl(
            direction=sample.control.direction,
            color=sample.control.color,
            shape=opposite,
            speed_bucket=sample.control.speed_bucket,
        )
        paired = render_motion_sample(
            frames=dataset.frames,
            size=dataset.size,
            seed=dataset.seed_for_index(index),
            control=paired_control,
            shape_size=controlled_motion_shape_size(dataset.size),
        )
        videos.append(sample.video)
        paired_videos.append(paired.video)
        captions.append(sample.caption)
        paired_captions.append(paired.caption)
        signs.append(1.0 if sample.control.shape == "square" else -1.0)
    return (
        torch.stack(videos).to(device),
        captions,
        torch.stack(paired_videos).to(device),
        paired_captions,
        torch.tensor(signs, device=device),
    )


def _expand_sign(sign: Tensor, reference: Tensor) -> Tensor:
    result = sign.to(dtype=reference.dtype)
    while result.ndim < reference.ndim:
        result = result.unsqueeze(-1)
    return result


def _load_models(cfg: ProjectConfig, checkpoint: Path, device: torch.device) -> StageBComponents:
    components = build_stage_b_components(cfg, device=device)
    raw = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(raw, dict):
        raise ValueError("checkpoint must contain a dictionary payload")
    payload = cast(dict[str, Any], raw)
    models = cast(dict[str, Any], payload["models"])
    components.vae.load_state_dict(models["vae"])
    components.text_encoder.load_state_dict(models["text_encoder"])
    components.dit.load_state_dict(models["dit"])
    components.vae.eval()
    components.text_encoder.eval()
    components.dit.eval()
    return components


def _local_response_spectrum(
    *,
    cfg: ProjectConfig,
    components: StageBComponents,
    clean: Tensor,
    paired: Tensor,
    captions: list[str],
    paired_captions: list[str],
    sign: Tensor,
    device: torch.device,
) -> list[dict[str, float | int]]:
    with torch.no_grad():
        clean_latents = components.vae.encode(clean)
        paired_latents = components.vae.encode(paired)
        target_delta = _expand_sign(sign, clean_latents) * (clean_latents - paired_latents)
        common_clean = 0.5 * (clean_latents + paired_latents)
        generator = torch.Generator(device=device).manual_seed(cfg.seed + 81_001)
        common_noise = torch.randn(
            common_clean.shape,
            generator=generator,
            device=device,
            dtype=common_clean.dtype,
        )
        all_captions = [*captions, *paired_captions]
        tokens = components.tokenizer.batch(all_captions, cfg.text.max_length, device=device)
        text = components.text_encoder(tokens.input_ids, tokens.attention_mask)
        first_text, second_text = text.chunk(2, dim=0)
        first_mask, second_mask = tokens.attention_mask.chunk(2, dim=0)

        rows: list[dict[str, float | int]] = []
        for timestep_tensor in components.schedule.sampling_timesteps(
            cfg.m1.sampling_steps,
            device=device,
        ):
            timestep = int(timestep_tensor.item())
            batch = clean_latents.shape[0]
            timesteps = torch.full((batch,), timestep, device=device, dtype=torch.long)
            noisy = components.schedule.add_noise(common_clean, common_noise, timesteps)
            first_noise = components.dit(noisy, timesteps, first_text, first_mask)
            second_noise = components.dit(noisy, timesteps, second_text, second_mask)
            first_clean = components.schedule.predict_clean(noisy, first_noise, timestep)
            second_clean = components.schedule.predict_clean(noisy, second_noise, timestep)
            predicted_delta = _expand_sign(sign, first_clean) * (first_clean - second_clean)
            metrics = counterfactual_response_metrics(predicted_delta, target_delta)
            alpha = float(components.schedule.alpha_cumprod[timestep].item())
            snr = alpha / max(1.0 - alpha, 1e-12)
            rows.append({"timestep": timestep, "snr": snr, **asdict(metrics)})
    return rows


def _trajectory_divergence(
    *,
    cfg: ProjectConfig,
    components: StageBComponents,
    captions: list[str],
    paired_captions: list[str],
    device: torch.device,
) -> list[dict[str, float | int]]:
    batch = len(captions)
    shape = (
        batch,
        cfg.vae.latent_channels,
        cfg.data.frames // cfg.vae.temporal_downsample,
        cfg.data.height // cfg.vae.spatial_downsample,
        cfg.data.width // cfg.vae.spatial_downsample,
    )
    generator = torch.Generator(device=device).manual_seed(cfg.seed + 82_001)
    base = torch.randn(
        shape, generator=generator, device=device, dtype=next(components.dit.parameters()).dtype
    )
    square_or_first = base.clone()
    circle_or_second = base.clone()
    first_tokens = components.tokenizer.batch(captions, cfg.text.max_length, device=device)
    second_tokens = components.tokenizer.batch(paired_captions, cfg.text.max_length, device=device)
    null_tokens = components.tokenizer.batch([""] * batch, cfg.text.max_length, device=device)
    with torch.no_grad():
        first_text = components.text_encoder(first_tokens.input_ids, first_tokens.attention_mask)
        second_text = components.text_encoder(second_tokens.input_ids, second_tokens.attention_mask)
        null_text = components.text_encoder(null_tokens.input_ids, null_tokens.attention_mask)
        timesteps = components.schedule.sampling_timesteps(cfg.m1.sampling_steps, device=device)
        rows: list[dict[str, float | int]] = []
        previous_rms = 0.0
        for index, timestep_tensor in enumerate(timesteps):
            timestep = int(timestep_tensor.item())
            previous_timestep = (
                int(timesteps[index + 1].item()) if index + 1 < len(timesteps) else -1
            )
            t = torch.full((batch,), timestep, device=device, dtype=torch.long)

            def guided(
                state: Tensor,
                text: Tensor,
                mask: Tensor,
                timestep_batch: Tensor = t,
            ) -> Tensor:
                model_state = torch.cat((state, state), dim=0)
                model_t = torch.cat((timestep_batch, timestep_batch), dim=0)
                model_text = torch.cat((null_text, text), dim=0)
                model_mask = torch.cat((null_tokens.attention_mask, mask), dim=0)
                pred = components.dit(model_state, model_t, model_text, model_mask)
                uncond, cond = pred.chunk(2, dim=0)
                return uncond + cfg.m1.guidance_scale * (cond - uncond)

            first_eps = guided(square_or_first, first_text, first_tokens.attention_mask)
            second_eps = guided(circle_or_second, second_text, second_tokens.attention_mask)
            square_or_first = components.schedule.ddim_step(
                square_or_first,
                first_eps,
                timestep=timestep,
                previous_timestep=previous_timestep,
            )
            circle_or_second = components.schedule.ddim_step(
                circle_or_second,
                second_eps,
                timestep=timestep,
                previous_timestep=previous_timestep,
            )
            delta = square_or_first - circle_or_second
            rms = float(delta.square().mean().sqrt().item())
            growth = rms / max(previous_rms, 1e-12) if previous_rms > 0.0 else 0.0
            rows.append(
                {"step": index + 1, "timestep": timestep, "latent_delta_rms": rms, "growth": growth}
            )
            previous_rms = rms
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only M1 square/circle causal response scan")
    parser.add_argument("--config", default="configs/tiny.toml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    if args.samples < 4:
        raise ValueError("--samples must be at least 4 for subspace diagnostics")

    device = torch.device("cuda" if args.cuda else "cpu")
    if args.cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    cfg = load_config(args.config)
    components = _load_models(cfg, Path(args.checkpoint), device)
    dataset = StageBSyntheticDataset(
        length=max(args.samples, 32),
        frames=cfg.data.frames,
        size=cfg.data.height,
        base_seed=cfg.seed,
        split="validation",
    )
    indices = list(range(args.samples))
    videos, captions, paired_videos, paired_captions, signs = _paired_batch(
        dataset,
        indices,
        device=device,
    )
    payload = {
        "checkpoint": str(args.checkpoint),
        "samples": args.samples,
        "sampling_steps": cfg.m1.sampling_steps,
        "semantic_timestep": cfg.m1.semantic_timestep,
        "local_response_spectrum": _local_response_spectrum(
            cfg=cfg,
            components=components,
            clean=videos,
            paired=paired_videos,
            captions=captions,
            paired_captions=paired_captions,
            sign=signs,
            device=device,
        ),
        "trajectory_divergence": _trajectory_divergence(
            cfg=cfg,
            components=components,
            captions=captions,
            paired_captions=paired_captions,
            device=device,
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"m1_shape_causal_scan output={output}")


if __name__ == "__main__":
    main()
