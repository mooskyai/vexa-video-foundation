from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
from torch import Tensor

from vexa_video.config import ProjectConfig
from vexa_video.data import COLORS, DIRECTIONS, SHAPES, StageBSyntheticDataset, SyntheticControl
from vexa_video.data.controlled_motion import controlled_motion_shape_size
from vexa_video.inference import sample_video
from vexa_video.training.trainer import StageBComponents

_COLOR_PROTOTYPES: dict[str, tuple[float, float, float]] = {
    "red": (1.0, -0.65, -0.65),
    "green": (-0.65, 1.0, -0.65),
    "blue": (-0.65, -0.65, 1.0),
    "yellow": (1.0, 1.0, -0.65),
}


@dataclass(frozen=True, slots=True)
class GeneratedVideoObservation:
    direction: str
    color: str
    shape: str
    mean_motion: float
    foreground_pixels: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class M1GenerationMetrics:
    direction_accuracy: float
    color_accuracy: float
    shape_accuracy: float
    mean_motion: float
    static_rate: float
    object_like_frame_rate: float
    persistent_video_rate: float
    mean_foreground_area_ratio: float
    direction_confusion: list[list[int]]
    color_confusion: list[list[int]]
    shape_confusion: list[list[int]]
    samples: int
    sampling_steps: int
    guidance_scale: float = 1.0
    gate_passed: bool = False


def passes_m1_generation_gate(metrics: M1GenerationMetrics, cfg: ProjectConfig) -> bool:
    return (
        metrics.direction_accuracy >= cfg.m1.generation_direction_gate
        and metrics.color_accuracy >= cfg.m1.generation_color_gate
        and metrics.shape_accuracy >= cfg.m1.generation_shape_gate
        and metrics.static_rate <= cfg.m1.generation_static_rate_gate
        and metrics.mean_motion > cfg.m1.generation_mean_motion_gate
        and metrics.object_like_frame_rate >= cfg.m1.generation_object_like_frame_gate
        and metrics.persistent_video_rate >= cfg.m1.generation_persistent_video_gate
        and cfg.m1.generation_foreground_area_ratio_min
        <= metrics.mean_foreground_area_ratio
        <= cfg.m1.generation_foreground_area_ratio_max
    )


def _foreground_weights(frame: Tensor) -> Tensor:
    strength = (frame + 1.0).abs().mean(dim=0)
    maximum = float(strength.max().item())
    threshold = max(0.10, maximum * 0.25)
    return torch.where(strength >= threshold, strength, torch.zeros_like(strength))


def _foreground_fill_ratio(weights: Tensor) -> float:
    mask = weights > 0
    points = mask.nonzero(as_tuple=False)
    if points.numel() == 0:
        return 0.0
    y_min = int(points[:, 0].min().item())
    y_max = int(points[:, 0].max().item())
    x_min = int(points[:, 1].min().item())
    x_max = int(points[:, 1].max().item())
    bbox_area = (y_max - y_min + 1) * (x_max - x_min + 1)
    return float(mask.sum().item()) / bbox_area


def _expected_shape_area(shape: str, size: int) -> int:
    shape_size = controlled_motion_shape_size(size)
    if shape == "square":
        return shape_size * shape_size
    if shape != "circle":
        raise ValueError(f"unknown shape: {shape}")
    coords = torch.arange(shape_size, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    center = (shape_size - 1) / 2.0
    radius = max(1.0, shape_size / 2.0)
    return int((((xx - center).square() + (yy - center).square()) <= radius**2).sum().item())


def analyze_generated_video(video: Tensor) -> GeneratedVideoObservation:
    if video.ndim != 4 or video.shape[0] != 3:
        raise ValueError("video must have shape [3, T, H, W]")
    _, frames, height, width = video.shape
    if frames < 2:
        raise ValueError("video must contain at least two frames")

    y_grid, x_grid = torch.meshgrid(
        torch.arange(height, device=video.device, dtype=video.dtype),
        torch.arange(width, device=video.device, dtype=video.dtype),
        indexing="ij",
    )
    centers: list[Tensor] = []
    all_weights: list[Tensor] = []
    for frame_index in range(frames):
        weights = _foreground_weights(video[:, frame_index])
        total = weights.sum()
        if float(total.item()) <= 1e-8:
            center = torch.tensor(
                [(width - 1) / 2.0, (height - 1) / 2.0],
                device=video.device,
                dtype=video.dtype,
            )
        else:
            center = torch.stack(
                ((weights * x_grid).sum() / total, (weights * y_grid).sum() / total)
            )
        centers.append(center)
        all_weights.append(weights)

    trajectory = torch.stack(centers)
    frame_motion = torch.linalg.vector_norm(trajectory[1:] - trajectory[:-1], dim=-1)
    mean_motion = float((frame_motion.mean() / max(height, width)).item())
    displacement = trajectory[-1] - trajectory[0]
    dx = float(displacement[0].item())
    dy = float(displacement[1].item())
    if abs(dx) >= abs(dy):
        direction = "right" if dx >= 0 else "left"
    else:
        direction = "down" if dy >= 0 else "up"

    fill_ratios = [_foreground_fill_ratio(weights) for weights in all_weights]
    mean_fill_ratio = sum(fill_ratios) / len(fill_ratios)
    square_fill = 1.0
    circle_fill = _expected_shape_area("circle", height) / _expected_shape_area("square", height)
    shape_boundary = (square_fill + circle_fill) / 2.0
    shape = "square" if mean_fill_ratio >= shape_boundary else "circle"

    weight_volume = torch.stack(all_weights)
    maximum_weight = float(weight_volume.max().item())
    color_mask = weight_volume >= max(0.10, maximum_weight * 0.40)
    pixels = video.permute(1, 2, 3, 0)[color_mask]
    mean_color = video.mean(dim=(1, 2, 3)) if pixels.numel() == 0 else pixels.mean(dim=0)
    color = min(
        COLORS,
        key=lambda name: float(
            torch.linalg.vector_norm(
                mean_color
                - torch.tensor(
                    _COLOR_PROTOTYPES[name],
                    device=video.device,
                    dtype=video.dtype,
                )
            ).item()
        ),
    )
    foreground_pixels = tuple(int((weights > 0).sum().item()) for weights in all_weights)
    return GeneratedVideoObservation(
        direction=direction,
        color=color,
        shape=shape,
        mean_motion=mean_motion,
        foreground_pixels=foreground_pixels,
    )


def evaluate_generated_videos(
    videos: Tensor,
    controls: list[SyntheticControl],
    *,
    static_motion_threshold: float,
    sampling_steps: int,
) -> M1GenerationMetrics:
    if videos.ndim != 5 or videos.shape[1] != 3:
        raise ValueError("videos must have shape [B, 3, T, H, W]")
    if videos.shape[0] != len(controls):
        raise ValueError("video/control counts must match")
    if not controls:
        raise ValueError("at least one generated video is required")

    direction_confusion = [[0 for _ in DIRECTIONS] for _ in DIRECTIONS]
    color_confusion = [[0 for _ in COLORS] for _ in COLORS]
    shape_confusion = [[0 for _ in SHAPES] for _ in SHAPES]
    direction_correct = 0
    color_correct = 0
    shape_correct = 0
    static = 0
    motion_total = 0.0
    object_like_frames = 0
    total_frames = 0
    persistent_videos = 0
    foreground_area_ratio_total = 0.0

    for index, control in enumerate(controls):
        observation = analyze_generated_video(videos[index])
        expected_direction = DIRECTIONS.index(control.direction)
        observed_direction = DIRECTIONS.index(observation.direction)
        expected_color = COLORS.index(control.color)
        observed_color = COLORS.index(observation.color)
        expected_shape = SHAPES.index(control.shape)
        observed_shape = SHAPES.index(observation.shape)
        direction_confusion[expected_direction][observed_direction] += 1
        color_confusion[expected_color][observed_color] += 1
        shape_confusion[expected_shape][observed_shape] += 1
        direction_correct += int(observation.direction == control.direction)
        color_correct += int(observation.color == control.color)
        shape_correct += int(observation.shape == control.shape)
        static += int(observation.mean_motion <= static_motion_threshold)
        motion_total += observation.mean_motion

        expected_area = _expected_shape_area(control.shape, int(videos.shape[-1]))
        area_ratios = [pixels / expected_area for pixels in observation.foreground_pixels]
        object_like = [0.5 <= ratio <= 4.0 for ratio in area_ratios]
        object_like_frames += sum(object_like)
        total_frames += len(object_like)
        persistent_videos += int(sum(object_like) / len(object_like) >= 0.875)
        foreground_area_ratio_total += sum(area_ratios) / len(area_ratios)

    count = len(controls)
    return M1GenerationMetrics(
        direction_accuracy=direction_correct / count,
        color_accuracy=color_correct / count,
        shape_accuracy=shape_correct / count,
        mean_motion=motion_total / count,
        static_rate=static / count,
        object_like_frame_rate=object_like_frames / total_frames,
        persistent_video_rate=persistent_videos / count,
        mean_foreground_area_ratio=foreground_area_ratio_total / count,
        direction_confusion=direction_confusion,
        color_confusion=color_confusion,
        shape_confusion=shape_confusion,
        samples=count,
        sampling_steps=sampling_steps,
    )


def evaluate_m1_generation(
    *,
    cfg: ProjectConfig,
    components: StageBComponents,
    device: torch.device,
    samples: int | None = None,
    sampling_steps: int | None = None,
    output: str | Path | None = None,
) -> M1GenerationMetrics:
    if cfg.data.height != cfg.data.width:
        raise ValueError("M1 evaluation currently requires square video dimensions")
    sample_count = samples if samples is not None else cfg.m1.eval_samples
    steps = sampling_steps if sampling_steps is not None else cfg.m1.sampling_steps
    if sample_count <= 0 or sample_count % 16 != 0:
        raise ValueError("M1 evaluation samples must be a positive multiple of 16")
    dataset = StageBSyntheticDataset(
        length=sample_count,
        frames=cfg.data.frames,
        size=cfg.data.height,
        base_seed=cfg.seed,
        split="test",
    )
    generated_batches: list[Tensor] = []
    controls: list[SyntheticControl] = []
    for start in range(0, sample_count, cfg.m1.eval_batch_size):
        stop = min(start + cfg.m1.eval_batch_size, sample_count)
        batch_samples = [dataset.sample(index) for index in range(start, stop)]
        prompts = [sample.caption for sample in batch_samples]
        controls.extend(sample.control for sample in batch_samples)
        generated = sample_video(
            cfg=cfg,
            tokenizer=components.tokenizer,
            text_encoder=components.text_encoder,
            vae=components.vae,
            dit=components.dit,
            schedule=components.schedule,
            prompts=prompts,
            seed=cfg.seed + 90_000 + start,
            sampling_steps=steps,
            device=device,
        )
        generated_batches.append(generated.cpu())
    videos = torch.cat(generated_batches, dim=0)
    metrics = evaluate_generated_videos(
        videos,
        controls,
        static_motion_threshold=cfg.m1.static_motion_threshold,
        sampling_steps=steps,
    )
    metrics = replace(
        metrics,
        guidance_scale=cfg.m1.guidance_scale,
        gate_passed=passes_m1_generation_gate(metrics, cfg),
    )
    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(asdict(metrics), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return metrics
