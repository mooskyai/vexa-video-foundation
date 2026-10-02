from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor

from vexa_video.config import ProjectConfig, load_config
from vexa_video.data import SHAPES, StageBSyntheticDataset, SyntheticControl, render_motion_sample
from vexa_video.data.controlled_motion import controlled_motion_shape_size
from vexa_video.training.m1_causal_math import (
    parallel_response_decomposition,
    permutation_linear_cka,
    symmetric_two_factor_decomposition,
)
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


def _load_models(
    cfg: ProjectConfig,
    checkpoint: Path,
    device: torch.device,
) -> StageBComponents:
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


def _mean_fill_ratio(video: Tensor) -> float:
    if video.ndim != 4 or video.shape[0] != 3:
        raise ValueError("video must have shape [3, T, H, W]")
    fills: list[float] = []
    for frame_index in range(video.shape[1]):
        frame = video[:, frame_index]
        strength = (frame + 1.0).abs().mean(dim=0)
        maximum = float(strength.max().item())
        threshold = max(0.10, maximum * 0.25)
        mask = strength >= threshold
        points = mask.nonzero(as_tuple=False)
        if points.numel() == 0:
            fills.append(0.0)
            continue
        y_min = int(points[:, 0].min().item())
        y_max = int(points[:, 0].max().item())
        x_min = int(points[:, 1].min().item())
        x_max = int(points[:, 1].max().item())
        box_area = (y_max - y_min + 1) * (x_max - x_min + 1)
        fills.append(float(mask.sum().item()) / box_area)
    return sum(fills) / len(fills)


def _decoded_shape_metrics(
    first_video: Tensor,
    second_video: Tensor,
    captions: list[str],
    paired_captions: list[str],
    signs: Tensor,
    *,
    size: int,
) -> dict[str, float]:
    shape_size = controlled_motion_shape_size(size)
    coords = torch.arange(shape_size, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    center = (shape_size - 1) / 2.0
    radius = max(1.0, shape_size / 2.0)
    circle_fill = float(
        (((xx - center).square() + (yy - center).square()) <= radius**2).sum().item()
    ) / float(shape_size * shape_size)
    boundary = 0.5 * (1.0 + circle_fill)

    first_correct = 0
    second_correct = 0
    opposite = 0
    signed_gaps: list[float] = []
    for index in range(first_video.shape[0]):
        first_fill = _mean_fill_ratio(first_video[index])
        second_fill = _mean_fill_ratio(second_video[index])
        first_prediction = "square" if first_fill >= boundary else "circle"
        second_prediction = "square" if second_fill >= boundary else "circle"
        first_target = "square" if " square " in captions[index] else "circle"
        second_target = "square" if " square " in paired_captions[index] else "circle"
        first_correct += int(first_prediction == first_target)
        second_correct += int(second_prediction == second_target)
        opposite += int(first_prediction != second_prediction)
        signed_gaps.append(float(signs[index].item()) * (first_fill - second_fill))
    count = first_video.shape[0]
    return {
        "first_shape_accuracy": first_correct / count,
        "second_shape_accuracy": second_correct / count,
        "paired_opposite_rate": opposite / count,
        "signed_fill_gap": sum(signed_gaps) / count,
        "target_fill_gap": 1.0 - circle_fill,
    }


def _model_three_conditions(
    *,
    components: StageBComponents,
    state: Tensor,
    timestep_batch: Tensor,
    null_text: Tensor,
    first_text: Tensor,
    second_text: Tensor,
    null_mask: Tensor,
    first_mask: Tensor,
    second_mask: Tensor,
    guidance_scale: float,
) -> tuple[Tensor, Tensor]:
    model_state = torch.cat((state, state, state), dim=0)
    model_t = timestep_batch.repeat(3)
    model_text = torch.cat((null_text, first_text, second_text), dim=0)
    model_mask = torch.cat((null_mask, first_mask, second_mask), dim=0)
    predicted = components.dit(model_state, model_t, model_text, model_mask)
    uncond, first_cond, second_cond = predicted.chunk(3, dim=0)
    first_guided = uncond + guidance_scale * (first_cond - uncond)
    second_guided = uncond + guidance_scale * (second_cond - uncond)
    return first_guided, second_guided


def _rms(value: Tensor) -> float:
    return float(value.square().mean().sqrt().item())


def _mean_cosine(first: Tensor, second: Tensor) -> float:
    first_flat = first.flatten(start_dim=1)
    second_flat = second.flatten(start_dim=1)
    return float(F.cosine_similarity(first_flat, second_flat, dim=1, eps=1e-8).mean().item())


def _run_decomposition(
    *,
    cfg: ProjectConfig,
    components: StageBComponents,
    clean: Tensor,
    paired: Tensor,
    captions: list[str],
    paired_captions: list[str],
    signs: Tensor,
    permutations: int,
    device: torch.device,
) -> dict[str, object]:
    batch = len(captions)
    latent_shape = (
        batch,
        cfg.vae.latent_channels,
        cfg.data.frames // cfg.vae.temporal_downsample,
        cfg.data.height // cfg.vae.spatial_downsample,
        cfg.data.width // cfg.vae.spatial_downsample,
    )
    with torch.no_grad():
        clean_latents = components.vae.encode(clean)
        paired_latents = components.vae.encode(paired)
        sign = _expand_sign(signs, clean_latents)
        target_delta = sign * (clean_latents - paired_latents)

        generator = torch.Generator(device=device).manual_seed(cfg.seed + 83_001)
        initial = torch.randn(
            latent_shape,
            generator=generator,
            device=device,
            dtype=next(components.dit.parameters()).dtype,
        )
        first_state = initial.clone()
        second_state = initial.clone()

        first_tokens = components.tokenizer.batch(captions, cfg.text.max_length, device=device)
        second_tokens = components.tokenizer.batch(
            paired_captions,
            cfg.text.max_length,
            device=device,
        )
        null_tokens = components.tokenizer.batch([""] * batch, cfg.text.max_length, device=device)
        first_text = components.text_encoder(first_tokens.input_ids, first_tokens.attention_mask)
        second_text = components.text_encoder(second_tokens.input_ids, second_tokens.attention_mask)
        null_text = components.text_encoder(null_tokens.input_ids, null_tokens.attention_mask)

        timesteps = components.schedule.sampling_timesteps(cfg.m1.sampling_steps, device=device)
        rows: list[dict[str, object]] = []
        decoded_rows: list[dict[str, float | int]] = []
        decode_steps = {1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50}

        for index, timestep_tensor in enumerate(timesteps):
            timestep = int(timestep_tensor.item())
            previous_timestep = (
                int(timesteps[index + 1].item()) if index + 1 < len(timesteps) else -1
            )
            timestep_batch = torch.full((batch,), timestep, device=device, dtype=torch.long)
            current_delta = sign * (first_state - second_state)
            current_rms = _rms(current_delta)

            ff_eps, fs_eps = _model_three_conditions(
                components=components,
                state=first_state,
                timestep_batch=timestep_batch,
                null_text=null_text,
                first_text=first_text,
                second_text=second_text,
                null_mask=null_tokens.attention_mask,
                first_mask=first_tokens.attention_mask,
                second_mask=second_tokens.attention_mask,
                guidance_scale=cfg.m1.guidance_scale,
            )
            sf_eps, ss_eps = _model_three_conditions(
                components=components,
                state=second_state,
                timestep_batch=timestep_batch,
                null_text=null_text,
                first_text=first_text,
                second_text=second_text,
                null_mask=null_tokens.attention_mask,
                first_mask=first_tokens.attention_mask,
                second_mask=second_tokens.attention_mask,
                guidance_scale=cfg.m1.guidance_scale,
            )

            y_ff = components.schedule.ddim_step(
                first_state,
                ff_eps,
                timestep=timestep,
                previous_timestep=previous_timestep,
            )
            y_fs = components.schedule.ddim_step(
                first_state,
                fs_eps,
                timestep=timestep,
                previous_timestep=previous_timestep,
            )
            y_sf = components.schedule.ddim_step(
                second_state,
                sf_eps,
                timestep=timestep,
                previous_timestep=previous_timestep,
            )
            y_ss = components.schedule.ddim_step(
                second_state,
                ss_eps,
                timestep=timestep,
                previous_timestep=previous_timestep,
            )

            condition, state_feedback, total = symmetric_two_factor_decomposition(
                y_ff,
                y_fs,
                y_sf,
                y_ss,
            )
            condition = sign * condition
            state_feedback = sign * state_feedback
            total = sign * total
            condition_rms = _rms(condition)
            state_rms = _rms(state_feedback)
            total_rms = _rms(total)
            response = parallel_response_decomposition(condition, target_delta)
            calibrated_cka = permutation_linear_cka(
                condition,
                target_delta,
                permutations=permutations,
                seed=cfg.seed + 84_001 + index,
            )
            alpha = float(components.schedule.alpha_cumprod[timestep].item())
            snr = alpha / max(1.0 - alpha, 1e-12)
            rows.append(
                {
                    "step": index + 1,
                    "timestep": timestep,
                    "snr": snr,
                    "current_delta_rms": current_rms,
                    "condition_next_rms": condition_rms,
                    "state_feedback_next_rms": state_rms,
                    "total_next_rms": total_rms,
                    "state_feedback_gain": state_rms / max(current_rms, 1e-12)
                    if current_rms > 0.0
                    else 0.0,
                    "condition_state_cosine": _mean_cosine(condition, state_feedback),
                    **{f"condition_{key}": value for key, value in asdict(response).items()},
                    "condition_cka_observed": calibrated_cka.observed,
                    "condition_cka_null_mean": calibrated_cka.null_mean,
                    "condition_cka_null_std": calibrated_cka.null_std,
                    "condition_cka_z": calibrated_cka.z_score,
                    "condition_cka_p": calibrated_cka.p_value,
                }
            )

            first_state = y_ff
            second_state = y_ss
            if index + 1 in decode_steps:
                first_video = components.vae.decode(first_state).clamp(-1.0, 1.0)
                second_video = components.vae.decode(second_state).clamp(-1.0, 1.0)
                decoded_rows.append(
                    {
                        "step": index + 1,
                        "timestep": timestep,
                        **_decoded_shape_metrics(
                            first_video,
                            second_video,
                            captions,
                            paired_captions,
                            signs,
                            size=cfg.data.height,
                        ),
                    }
                )

    return {
        "dynamics": rows,
        "decoded_shape_trajectory": decoded_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only M1 state-vs-condition causal dynamics decomposition"
    )
    parser.add_argument("--config", default="configs/tiny.toml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--permutations", type=int, default=256)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    if args.samples < 4:
        raise ValueError("--samples must be at least 4")
    if args.permutations <= 0:
        raise ValueError("--permutations must be positive")

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
    videos, captions, paired, paired_captions, signs = _paired_batch(
        dataset,
        list(range(args.samples)),
        device=device,
    )
    result = _run_decomposition(
        cfg=cfg,
        components=components,
        clean=videos,
        paired=paired,
        captions=captions,
        paired_captions=paired_captions,
        signs=signs,
        permutations=args.permutations,
        device=device,
    )
    payload = {
        "checkpoint": str(args.checkpoint),
        "samples": args.samples,
        "permutations": args.permutations,
        "sampling_steps": cfg.m1.sampling_steps,
        "guidance_scale": cfg.m1.guidance_scale,
        **result,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"m1_shape_dynamics_decomposition output={output}")


if __name__ == "__main__":
    main()
