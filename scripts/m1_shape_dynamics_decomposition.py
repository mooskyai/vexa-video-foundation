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
from vexa_video.training.evaluation import evaluate_generated_videos
from vexa_video.training.m1_causal_math import (
    ddim_linear_coefficients,
    parallel_response_decomposition,
    permutation_linear_cka,
    symmetric_two_factor_decomposition,
)
from vexa_video.training.m1_conditioning_diagnostics import diagnostic_dit_forward
from vexa_video.training.trainer import StageBComponents, build_stage_b_components


def _paired_batch(
    dataset: StageBSyntheticDataset,
    indices: list[int],
    *,
    position_variants: int,
    device: torch.device,
) -> tuple[
    Tensor,
    list[str],
    list[SyntheticControl],
    Tensor,
    list[str],
    list[SyntheticControl],
    Tensor,
    list[dict[str, int]],
]:
    if position_variants <= 0:
        raise ValueError("position_variants must be positive")
    videos: list[Tensor] = []
    paired_videos: list[Tensor] = []
    captions: list[str] = []
    paired_captions: list[str] = []
    controls: list[SyntheticControl] = []
    paired_controls: list[SyntheticControl] = []
    signs: list[float] = []
    position_records: list[dict[str, int]] = []
    for index in indices:
        sample = dataset.sample(index)
        opposite = SHAPES[(SHAPES.index(sample.control.shape) + 1) % len(SHAPES)]
        paired_control = SyntheticControl(
            direction=sample.control.direction,
            color=sample.control.color,
            shape=opposite,
            speed_bucket=sample.control.speed_bucket,
        )
        for variant in range(position_variants):
            seed = dataset.seed_for_index(index) + variant * 1_000_003
            if variant == 0:
                first = sample
            else:
                first = render_motion_sample(
                    frames=dataset.frames,
                    size=dataset.size,
                    seed=seed,
                    control=sample.control,
                    shape_size=controlled_motion_shape_size(dataset.size),
                )
            paired = render_motion_sample(
                frames=dataset.frames,
                size=dataset.size,
                seed=seed,
                control=paired_control,
                shape_size=controlled_motion_shape_size(dataset.size),
            )
            videos.append(first.video)
            paired_videos.append(paired.video)
            captions.append(first.caption)
            paired_captions.append(paired.caption)
            controls.append(sample.control)
            paired_controls.append(paired_control)
            signs.append(1.0 if sample.control.shape == "square" else -1.0)
            position_records.append(
                {
                    "source_index": index,
                    "variant": variant,
                    "seed": seed,
                    "start_x": int(first.trajectory_xy[0, 0].item()),
                    "start_y": int(first.trajectory_xy[0, 1].item()),
                }
            )
    return (
        torch.stack(videos).to(device),
        captions,
        controls,
        torch.stack(paired_videos).to(device),
        paired_captions,
        paired_controls,
        torch.tensor(signs, device=device),
        position_records,
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


def _evaluation_summary(
    video: Tensor,
    controls: list[SyntheticControl],
    *,
    cfg: ProjectConfig,
    step: int,
) -> dict[str, float]:
    metrics = evaluate_generated_videos(
        video,
        controls,
        static_motion_threshold=cfg.m1.static_motion_threshold,
        sampling_steps=step,
    )
    return {
        "direction_accuracy": metrics.direction_accuracy,
        "color_accuracy": metrics.color_accuracy,
        "shape_accuracy": metrics.shape_accuracy,
        "mean_motion": metrics.mean_motion,
        "static_rate": metrics.static_rate,
        "object_like_frame_rate": metrics.object_like_frame_rate,
        "persistent_video_rate": metrics.persistent_video_rate,
        "mean_foreground_area_ratio": metrics.mean_foreground_area_ratio,
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


def _relative_delta(first: Tensor, second: Tensor) -> dict[str, float]:
    delta = first - second
    scale = 0.5 * (_rms(first) + _rms(second))
    return {
        "delta_rms": _rms(delta),
        "relative_delta_rms": _rms(delta) / max(scale, 1e-12),
    }


def _response_metrics(response: Tensor, target: Tensor) -> dict[str, float]:
    decomposition = parallel_response_decomposition(response, target)
    return {
        "rms": _rms(response),
        "cosine": _mean_cosine(response, target),
        **asdict(decomposition),
    }


def _shape_word_mask(
    captions: list[str],
    *,
    sequence_length: int,
    device: torch.device,
) -> Tensor:
    mask = torch.zeros((len(captions), sequence_length), dtype=torch.bool, device=device)
    for row, caption in enumerate(captions):
        encoded = caption.encode("utf-8")
        matches = [word for word in (b"square", b"circle") if word in encoded]
        if len(matches) != 1:
            raise ValueError(f"caption must contain exactly one shape word: {caption}")
        word = matches[0]
        start = encoded.index(word) + 1
        stop = start + len(word)
        if stop > sequence_length:
            raise ValueError("shape word is truncated by the configured text sequence length")
        mask[row, start:stop] = True
    return mask


def _canonical_pair(
    first: Tensor,
    second: Tensor,
    signs: Tensor,
) -> tuple[Tensor, Tensor]:
    selector = signs.gt(0.0)
    while selector.ndim < first.ndim:
        selector = selector.unsqueeze(-1)
    return torch.where(selector, first, second), torch.where(selector, second, first)


def _canonical_captions(
    captions: list[str],
    paired_captions: list[str],
    signs: Tensor,
) -> tuple[list[str], list[str]]:
    square: list[str] = []
    circle: list[str] = []
    for index, sign in enumerate(signs.tolist()):
        if float(sign) > 0.0:
            square.append(captions[index])
            circle.append(paired_captions[index])
        else:
            square.append(paired_captions[index])
            circle.append(captions[index])
    return square, circle


def _previous_timestep_for_local_probe(timestep: int, sampling_timesteps: Tensor) -> int:
    lower = [int(value.item()) for value in sampling_timesteps if int(value.item()) < timestep]
    return max(lower) if lower else -1


def _latent_statistics(square_latents: Tensor, circle_latents: Tensor) -> dict[str, object]:
    combined = torch.cat((square_latents, circle_latents), dim=0)
    channel_view = combined.permute(1, 0, 2, 3, 4).reshape(combined.shape[1], -1)
    channel_mean = channel_view.mean(dim=1)
    channel_variance = channel_view.var(dim=1, unbiased=False)
    target_delta = square_latents - circle_latents
    return {
        "channel_mean": [float(value.item()) for value in channel_mean],
        "channel_centered_variance": [float(value.item()) for value in channel_variance],
        "mean_centered_variance": float(channel_variance.mean().item()),
        "target_delta_rms": _rms(target_delta),
    }


def _local_anchor_response_spectrum(
    *,
    cfg: ProjectConfig,
    components: StageBComponents,
    square_latents: Tensor,
    circle_latents: Tensor,
    target_delta: Tensor,
    square_text: Tensor,
    circle_text: Tensor,
    square_mask: Tensor,
    circle_mask: Tensor,
    noise_seeds: int,
    device: torch.device,
) -> list[dict[str, object]]:
    if noise_seeds <= 0:
        raise ValueError("noise_seeds must be positive")
    sampling_timesteps = components.schedule.sampling_timesteps(
        cfg.m1.sampling_steps, device=device
    )
    timestep_values = {int(value.item()) for value in sampling_timesteps}
    timestep_values.add(cfg.m1.semantic_timestep)
    combined = torch.cat((square_latents, circle_latents), dim=0)
    channel_view = combined.permute(1, 0, 2, 3, 4).reshape(combined.shape[1], -1)
    latent_power = float(channel_view.var(dim=1, unbiased=False).mean().item())
    anchors = {
        "square": square_latents,
        "circle": circle_latents,
        "midpoint": 0.5 * (square_latents + circle_latents),
    }
    rows: list[dict[str, object]] = []
    for timestep in sorted(timestep_values, reverse=True):
        previous_timestep = _previous_timestep_for_local_probe(timestep, sampling_timesteps)
        alpha = float(components.schedule.alpha_cumprod[timestep].item())
        snr = alpha / max(1.0 - alpha, 1e-12)
        for anchor_name, anchor in anchors.items():
            epsilon_deltas: list[Tensor] = []
            clean_deltas: list[Tensor] = []
            forcing_deltas: list[Tensor] = []
            for seed_index in range(noise_seeds):
                generator = torch.Generator(device=device).manual_seed(
                    cfg.seed + 85_001 + seed_index * 997 + timestep
                )
                noise = torch.randn(
                    anchor.shape,
                    generator=generator,
                    device=device,
                    dtype=anchor.dtype,
                )
                timestep_batch = torch.full(
                    (anchor.shape[0],), timestep, device=device, dtype=torch.long
                )
                noisy = components.schedule.add_noise(anchor, noise, timestep_batch)
                square_eps = components.dit(noisy, timestep_batch, square_text, square_mask)
                circle_eps = components.dit(noisy, timestep_batch, circle_text, circle_mask)
                epsilon_delta = square_eps - circle_eps
                square_clean = components.schedule.predict_clean(noisy, square_eps, timestep)
                circle_clean = components.schedule.predict_clean(noisy, circle_eps, timestep)
                square_next = components.schedule.ddim_step(
                    noisy,
                    square_eps,
                    timestep=timestep,
                    previous_timestep=previous_timestep,
                )
                circle_next = components.schedule.ddim_step(
                    noisy,
                    circle_eps,
                    timestep=timestep,
                    previous_timestep=previous_timestep,
                )
                epsilon_deltas.append(epsilon_delta)
                clean_deltas.append(square_clean - circle_clean)
                forcing_deltas.append(square_next - circle_next)
            epsilon_delta = torch.cat(epsilon_deltas, dim=0)
            clean_delta = torch.cat(clean_deltas, dim=0)
            forcing_delta = torch.cat(forcing_deltas, dim=0)
            repeated_target = target_delta.repeat((noise_seeds, 1, 1, 1, 1))
            rows.append(
                {
                    "timestep": timestep,
                    "previous_timestep": previous_timestep,
                    "anchor": anchor_name,
                    "nominal_snr": snr,
                    "empirical_centered_signal_snr": snr * latent_power,
                    "epsilon_vs_negative_target": _response_metrics(
                        epsilon_delta, -repeated_target
                    ),
                    "predicted_clean": _response_metrics(clean_delta, repeated_target),
                    "ddim_forcing": _response_metrics(forcing_delta, repeated_target),
                }
            )
    return rows


def _conditioning_trace_at_semantic_timestep(
    *,
    cfg: ProjectConfig,
    components: StageBComponents,
    midpoint: Tensor,
    square_captions: list[str],
    circle_captions: list[str],
    square_text: Tensor,
    circle_text: Tensor,
    square_mask: Tensor,
    circle_mask: Tensor,
    device: torch.device,
) -> dict[str, object]:
    timestep = cfg.m1.semantic_timestep
    generator = torch.Generator(device=device).manual_seed(cfg.seed + 86_001)
    noise = torch.randn(midpoint.shape, generator=generator, device=device, dtype=midpoint.dtype)
    timestep_batch = torch.full((midpoint.shape[0],), timestep, device=device, dtype=torch.long)
    noisy = components.schedule.add_noise(midpoint, noise, timestep_batch)
    square_shape_mask = _shape_word_mask(
        square_captions, sequence_length=square_mask.shape[1], device=device
    )
    circle_shape_mask = _shape_word_mask(
        circle_captions, sequence_length=circle_mask.shape[1], device=device
    )
    square_prediction, square_trace = diagnostic_dit_forward(
        components.dit,
        noisy,
        timestep_batch,
        square_text,
        square_mask,
        shape_token_mask=square_shape_mask,
    )
    circle_prediction, circle_trace = diagnostic_dit_forward(
        components.dit,
        noisy,
        timestep_batch,
        circle_text,
        circle_mask,
        shape_token_mask=circle_shape_mask,
    )
    return {
        "timestep": timestep,
        "epsilon": _relative_delta(square_prediction, circle_prediction),
        "pool": _relative_delta(square_trace.pooled, circle_trace.pooled),
        "first_attention": {
            **_relative_delta(square_trace.first_attention, circle_trace.first_attention),
            "square_entropy": float(square_trace.first_entropy.mean().item()),
            "circle_entropy": float(circle_trace.first_entropy.mean().item()),
            "square_shape_token_mass": float(square_trace.first_shape_mass.mean().item()),
            "circle_shape_token_mass": float(circle_trace.first_shape_mass.mean().item()),
            "square_valid_logit_std": float(square_trace.first_logit_std.mean().item()),
            "circle_valid_logit_std": float(circle_trace.first_logit_std.mean().item()),
        },
        "final_attention": {
            **_relative_delta(square_trace.final_attention, circle_trace.final_attention),
            "square_entropy": float(square_trace.final_entropy.mean().item()),
            "circle_entropy": float(circle_trace.final_entropy.mean().item()),
            "square_shape_token_mass": float(square_trace.final_shape_mass.mean().item()),
            "circle_shape_token_mass": float(circle_trace.final_shape_mass.mean().item()),
            "square_valid_logit_std": float(square_trace.final_logit_std.mean().item()),
            "circle_valid_logit_std": float(circle_trace.final_logit_std.mean().item()),
        },
    }


def _guided_diagnostic_prediction(
    *,
    cfg: ProjectConfig,
    components: StageBComponents,
    state: Tensor,
    timestep_batch: Tensor,
    null_text: Tensor,
    square_text: Tensor,
    circle_text: Tensor,
    null_mask: Tensor,
    square_mask: Tensor,
    circle_mask: Tensor,
    disable_pool: bool,
    disable_first_attention: bool,
    disable_final_attention: bool,
) -> tuple[Tensor, Tensor, float]:
    model_state = torch.cat((state, state, state), dim=0)
    model_t = timestep_batch.repeat(3)
    model_text = torch.cat((null_text, square_text, circle_text), dim=0)
    model_mask = torch.cat((null_mask, square_mask, circle_mask), dim=0)
    diagnostic, _ = diagnostic_dit_forward(
        components.dit,
        model_state,
        model_t,
        model_text,
        model_mask,
        disable_pool=disable_pool,
        disable_first_attention=disable_first_attention,
        disable_final_attention=disable_final_attention,
    )
    equivalence_error = 0.0
    if not disable_pool and not disable_first_attention and not disable_final_attention:
        production = components.dit(model_state, model_t, model_text, model_mask)
        equivalence_error = float((diagnostic - production).abs().max().item())
        if equivalence_error > 1e-5:
            raise RuntimeError(
                "diagnostic VideoDiT mirror no longer matches production forward: "
                f"max_abs={equivalence_error:.8f}"
            )
    uncond, square_cond, circle_cond = diagnostic.chunk(3, dim=0)
    square_guided = uncond + cfg.m1.guidance_scale * (square_cond - uncond)
    circle_guided = uncond + cfg.m1.guidance_scale * (circle_cond - uncond)
    return square_guided, circle_guided, equivalence_error


def _conditioning_branch_ablation(
    *,
    cfg: ProjectConfig,
    components: StageBComponents,
    midpoint: Tensor,
    target_delta: Tensor,
    null_text: Tensor,
    square_text: Tensor,
    circle_text: Tensor,
    null_mask: Tensor,
    square_mask: Tensor,
    circle_mask: Tensor,
    device: torch.device,
) -> dict[str, object]:
    timestep = cfg.m1.semantic_timestep
    sampling_timesteps = components.schedule.sampling_timesteps(
        cfg.m1.sampling_steps, device=device
    )
    previous_timestep = _previous_timestep_for_local_probe(timestep, sampling_timesteps)
    generator = torch.Generator(device=device).manual_seed(cfg.seed + 87_001)
    noise = torch.randn(midpoint.shape, generator=generator, device=device, dtype=midpoint.dtype)
    timestep_batch = torch.full((midpoint.shape[0],), timestep, device=device, dtype=torch.long)
    noisy = components.schedule.add_noise(midpoint, noise, timestep_batch)
    variants = {
        "baseline": (False, False, False),
        "no_pool": (True, False, False),
        "no_first_attention": (False, True, False),
        "no_final_attention": (False, False, True),
        "no_text_conditioning": (True, True, True),
    }
    rows: dict[str, object] = {}
    for name, (disable_pool, disable_first, disable_final) in variants.items():
        square_eps, circle_eps, equivalence_error = _guided_diagnostic_prediction(
            cfg=cfg,
            components=components,
            state=noisy,
            timestep_batch=timestep_batch,
            null_text=null_text,
            square_text=square_text,
            circle_text=circle_text,
            null_mask=null_mask,
            square_mask=square_mask,
            circle_mask=circle_mask,
            disable_pool=disable_pool,
            disable_first_attention=disable_first,
            disable_final_attention=disable_final,
        )
        epsilon_delta = square_eps - circle_eps
        square_clean = components.schedule.predict_clean(noisy, square_eps, timestep)
        circle_clean = components.schedule.predict_clean(noisy, circle_eps, timestep)
        square_next = components.schedule.ddim_step(
            noisy,
            square_eps,
            timestep=timestep,
            previous_timestep=previous_timestep,
        )
        circle_next = components.schedule.ddim_step(
            noisy,
            circle_eps,
            timestep=timestep,
            previous_timestep=previous_timestep,
        )
        rows[name] = {
            "diagnostic_forward_max_abs_error": equivalence_error,
            "epsilon_vs_negative_target": _response_metrics(epsilon_delta, -target_delta),
            "predicted_clean": _response_metrics(square_clean - circle_clean, target_delta),
            "ddim_forcing": _response_metrics(square_next - circle_next, target_delta),
        }
    return {
        "timestep": timestep,
        "previous_timestep": previous_timestep,
        "variants": rows,
    }


def _run_decomposition(
    *,
    cfg: ProjectConfig,
    components: StageBComponents,
    clean: Tensor,
    paired: Tensor,
    captions: list[str],
    controls: list[SyntheticControl],
    paired_captions: list[str],
    paired_controls: list[SyntheticControl],
    signs: Tensor,
    position_records: list[dict[str, int]],
    permutations: int,
    noise_seeds: int,
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
        square_latents, circle_latents = _canonical_pair(clean_latents, paired_latents, signs)
        square_captions, circle_captions = _canonical_captions(captions, paired_captions, signs)

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
        square_text, circle_text = _canonical_pair(first_text, second_text, signs)
        square_mask, circle_mask = _canonical_pair(
            first_tokens.attention_mask, second_tokens.attention_mask, signs
        )

        timesteps = components.schedule.sampling_timesteps(cfg.m1.sampling_steps, device=device)
        rows: list[dict[str, object]] = []
        decoded_rows: list[dict[str, object]] = []
        decode_steps = {1, 5, 10, 15, 20, 25, 30, 35, 40, 45, cfg.m1.sampling_steps}

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
            eps_condition, eps_state, _ = symmetric_two_factor_decomposition(
                ff_eps, fs_eps, sf_eps, ss_eps
            )
            eps_condition = sign * eps_condition
            eps_state = sign * eps_state

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

            alpha_t = components.schedule.alpha_cumprod.to(
                device=first_state.device, dtype=first_state.dtype
            )[timestep]
            alpha_previous = (
                None
                if previous_timestep < 0
                else components.schedule.alpha_cumprod.to(
                    device=first_state.device, dtype=first_state.dtype
                )[previous_timestep]
            )
            state_coefficient, noise_coefficient = ddim_linear_coefficients(alpha_t, alpha_previous)
            _, clean_noise_coefficient = ddim_linear_coefficients(alpha_t, None)
            analytic_transport = state_coefficient * current_delta
            learned_state_feedback = noise_coefficient * eps_state
            condition_from_epsilon = noise_coefficient * eps_condition
            predicted_clean_condition = clean_noise_coefficient * eps_condition
            decomposition_residual = total - (
                analytic_transport + learned_state_feedback + condition_from_epsilon
            )

            condition_response = parallel_response_decomposition(condition, target_delta)
            calibrated_cka = permutation_linear_cka(
                condition,
                target_delta,
                permutations=permutations,
                seed=cfg.seed + 84_001 + index,
            )
            alpha = float(alpha_t.item())
            snr = alpha / max(1.0 - alpha, 1e-12)
            rows.append(
                {
                    "step": index + 1,
                    "timestep": timestep,
                    "previous_timestep": previous_timestep,
                    "snr": snr,
                    "ddim_state_coefficient": float(state_coefficient.item()),
                    "ddim_noise_coefficient": float(noise_coefficient.item()),
                    "current_delta_rms": current_rms,
                    "epsilon_condition": _response_metrics(eps_condition, -target_delta),
                    "predicted_clean_condition": _response_metrics(
                        predicted_clean_condition, target_delta
                    ),
                    "condition_next_rms": _rms(condition),
                    "analytic_transport_rms": _rms(analytic_transport),
                    "learned_state_feedback_rms": _rms(learned_state_feedback),
                    "state_feedback_next_rms": _rms(state_feedback),
                    "total_next_rms": _rms(total),
                    "analytic_transport_gain": _rms(analytic_transport) / max(current_rms, 1e-12)
                    if current_rms > 0.0
                    else 0.0,
                    "learned_state_feedback_gain": _rms(learned_state_feedback)
                    / max(current_rms, 1e-12)
                    if current_rms > 0.0
                    else 0.0,
                    "state_feedback_gain": _rms(state_feedback) / max(current_rms, 1e-12)
                    if current_rms > 0.0
                    else 0.0,
                    "condition_formula_residual_rms": _rms(condition - condition_from_epsilon),
                    "state_formula_residual_rms": _rms(
                        state_feedback - analytic_transport - learned_state_feedback
                    ),
                    "total_decomposition_residual_rms": _rms(decomposition_residual),
                    "condition_state_cosine": _mean_cosine(condition, state_feedback),
                    **{
                        f"condition_{key}": value
                        for key, value in asdict(condition_response).items()
                    },
                    "condition_cka_observed": calibrated_cka.observed,
                    "condition_cka_null_mean": calibrated_cka.null_mean,
                    "condition_cka_null_std": calibrated_cka.null_std,
                    "condition_cka_z": calibrated_cka.z_score,
                    "condition_cka_p": calibrated_cka.p_value,
                }
            )

            first_predicted_clean = components.schedule.predict_clean(first_state, ff_eps, timestep)
            second_predicted_clean = components.schedule.predict_clean(
                second_state, ss_eps, timestep
            )
            first_state = y_ff
            second_state = y_ss
            if index + 1 in decode_steps:
                first_video = components.vae.decode(first_predicted_clean).clamp(-1.0, 1.0)
                second_video = components.vae.decode(second_predicted_clean).clamp(-1.0, 1.0)
                decoded_rows.append(
                    {
                        "step": index + 1,
                        "timestep": timestep,
                        "previous_timestep": previous_timestep,
                        "paired_shape": _decoded_shape_metrics(
                            first_video,
                            second_video,
                            captions,
                            paired_captions,
                            signs,
                            size=cfg.data.height,
                        ),
                        "first_frozen_metrics": _evaluation_summary(
                            first_video, controls, cfg=cfg, step=index + 1
                        ),
                        "second_frozen_metrics": _evaluation_summary(
                            second_video, paired_controls, cfg=cfg, step=index + 1
                        ),
                    }
                )

        local_response = _local_anchor_response_spectrum(
            cfg=cfg,
            components=components,
            square_latents=square_latents,
            circle_latents=circle_latents,
            target_delta=target_delta,
            square_text=square_text,
            circle_text=circle_text,
            square_mask=square_mask,
            circle_mask=circle_mask,
            noise_seeds=noise_seeds,
            device=device,
        )
        conditioning_trace = _conditioning_trace_at_semantic_timestep(
            cfg=cfg,
            components=components,
            midpoint=0.5 * (square_latents + circle_latents),
            square_captions=square_captions,
            circle_captions=circle_captions,
            square_text=square_text,
            circle_text=circle_text,
            square_mask=square_mask,
            circle_mask=circle_mask,
            device=device,
        )
        branch_ablation = _conditioning_branch_ablation(
            cfg=cfg,
            components=components,
            midpoint=0.5 * (square_latents + circle_latents),
            target_delta=target_delta,
            null_text=null_text,
            square_text=square_text,
            circle_text=circle_text,
            null_mask=null_tokens.attention_mask,
            square_mask=square_mask,
            circle_mask=circle_mask,
            device=device,
        )

    unique_positions = sorted({(row["start_x"], row["start_y"]) for row in position_records})
    return {
        "latent_statistics": _latent_statistics(square_latents, circle_latents),
        "position_coverage": {
            "records": position_records,
            "unique_start_positions": len(unique_positions),
        },
        "consumed_conditioning": conditioning_trace,
        "conditioning_branch_ablation": branch_ablation,
        "local_anchor_response_spectrum": local_response,
        "dynamics": rows,
        "decoded_predicted_clean_trajectory": decoded_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only M1 conditioning, local-response, and reverse-dynamics decomposition"
    )
    parser.add_argument("--config", default="configs/tiny.toml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--position-variants", type=int, default=2)
    parser.add_argument("--noise-seeds", type=int, default=3)
    parser.add_argument("--permutations", type=int, default=256)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    if args.samples < 4:
        raise ValueError("--samples must be at least 4")
    if args.position_variants <= 0:
        raise ValueError("--position-variants must be positive")
    if args.noise_seeds <= 0:
        raise ValueError("--noise-seeds must be positive")
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
    (
        videos,
        captions,
        controls,
        paired,
        paired_captions,
        paired_controls,
        signs,
        position_records,
    ) = _paired_batch(
        dataset,
        list(range(args.samples)),
        position_variants=args.position_variants,
        device=device,
    )
    result = _run_decomposition(
        cfg=cfg,
        components=components,
        clean=videos,
        paired=paired,
        captions=captions,
        controls=controls,
        paired_captions=paired_captions,
        paired_controls=paired_controls,
        signs=signs,
        position_records=position_records,
        permutations=args.permutations,
        noise_seeds=args.noise_seeds,
        device=device,
    )
    payload = {
        "checkpoint": str(args.checkpoint),
        "base_samples": args.samples,
        "position_variants": args.position_variants,
        "effective_samples": len(captions),
        "noise_seeds": args.noise_seeds,
        "permutations": args.permutations,
        "sampling_steps": cfg.m1.sampling_steps,
        "semantic_timestep": cfg.m1.semantic_timestep,
        "guidance_scale": cfg.m1.guidance_scale,
        **result,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"m1_shape_dynamics_decomposition output={output}")


if __name__ == "__main__":
    main()
