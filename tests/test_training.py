from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from vexa_video.config import ProjectConfig, load_config
from vexa_video.data import StageBSyntheticDataset
from vexa_video.inference import sample_video
from vexa_video.inference.sampler import guided_ddim_rollout
from vexa_video.models.dit import spatiotemporal_position_embedding, token_text_attention
from vexa_video.training.checkpoint import build_training_checkpoint
from vexa_video.training.evaluation import (
    evaluate_generated_videos,
    passes_m1_generation_gate,
)
from vexa_video.training.trainer import (
    _normalize_cuda_rng_states,
    _soft_shape_score,
    _soft_video_features,
    _validation_generation_metrics,
    build_stage_b_components,
    diffusion_train_step,
    load_stage_b_weights,
    reconstruction_loss,
)


def _tiny_test_config() -> ProjectConfig:
    cfg = load_config(Path(__file__).parents[1] / "configs" / "tiny.toml")
    return replace(
        cfg,
        text=replace(cfg.text, d_model=16, layers=1, heads=2, ff_mult=2, max_length=48),
        vae=replace(cfg.vae, base_channels=4),
        dit=replace(cfg.dit, hidden_size=24, layers=1, heads=2),
        diffusion=replace(cfg.diffusion, timesteps=20),
        m1=replace(
            cfg.m1,
            batch_size=1,
            train_samples=32,
            validation_samples=16,
            sampling_steps=2,
            eval_samples=16,
            eval_batch_size=1,
            semantic_timestep=10,
            rollout_steps=2,
            rollout_batch_size=1,
            full_rollout_steps=2,
            full_rollout_every=1,
            validation_generation_samples=16,
            vae_visual_validation_samples=32,
            vae_semantic_accuracy_gate=0.0,
            vae_object_like_frame_gate=0.0,
            vae_persistent_video_gate=0.0,
            vae_foreground_area_ratio_min=0.0,
            vae_foreground_area_ratio_max=1_000_000.0,
        ),
    )


def test_controlled_motion_curriculum_is_fixed_medium_and_balanced() -> None:
    dataset = StageBSyntheticDataset(
        length=32,
        frames=8,
        size=32,
        base_seed=42,
        split="train",
    )
    controls = [dataset.sample(index).control for index in range(32)]
    assert {control.speed_bucket for control in controls} == {"medium"}
    assert {control.direction for control in controls} == {"right", "left", "down", "up"}
    assert {control.color for control in controls} == {"red", "green", "blue", "yellow"}
    assert {control.shape for control in controls} == {"square", "circle"}


def test_spatiotemporal_positions_are_location_dependent() -> None:
    embedding = spatiotemporal_position_embedding(
        (2, 2, 2),
        24,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert embedding.shape == (1, 8, 24)
    assert not torch.equal(embedding[:, 0], embedding[:, -1])


def test_token_text_attention_uses_unmasked_token_content() -> None:
    video = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    text = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]])
    mask = torch.tensor([[True, True, False]])
    conditioned = token_text_attention(video, text, mask)
    assert conditioned.shape == video.shape
    assert conditioned[0, 0, 0] > conditioned[0, 0, 1]
    assert conditioned[0, 1, 1] > conditioned[0, 1, 0]


def test_soft_video_features_recover_cardinal_direction_and_color() -> None:
    dataset = StageBSyntheticDataset(length=4, frames=8, size=32, base_seed=42, split="train")
    videos = torch.stack([dataset.sample(index).video for index in range(4)])
    motion, color = _soft_video_features(videos)
    assert motion[0, 0] > 0
    assert motion[1, 0] < 0
    assert motion[2, 1] > 0
    assert motion[3, 1] < 0
    assert color.shape == (4, 3)


def test_soft_shape_score_separates_square_and_circle() -> None:
    dataset = StageBSyntheticDataset(
        length=32,
        frames=8,
        size=32,
        base_seed=42,
        split="train",
    )
    square = dataset.sample(0).video
    circle = dataset.sample(16).video
    scores = _soft_shape_score(torch.stack((square, circle)))
    assert scores[0] > scores[1]
    assert float(scores[0].item()) > 0.9
    assert float(scores[1].item()) < 0.8


def test_reconstruction_loss_penalizes_diffuse_foreground() -> None:
    sample = StageBSyntheticDataset(
        length=1,
        frames=8,
        size=32,
        base_seed=42,
        split="train",
    ).sample(0)
    target = sample.video.unsqueeze(0)
    diffuse = torch.full_like(target, -1.0)
    diffuse[:, :, :, 4:28, 4:28] = 0.5
    exact = reconstruction_loss(target, target, silhouette_weight=1.0)
    diffuse_loss = reconstruction_loss(diffuse, target, silhouette_weight=1.0)
    assert exact == 0.0
    assert diffuse_loss > exact


def test_diffusion_train_step_is_finite_and_reaches_text_and_dit() -> None:
    cfg = _tiny_test_config()
    components = build_stage_b_components(cfg, device=torch.device("cpu"))
    for parameter in components.vae.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [*components.text_encoder.parameters(), *components.dit.parameters()],
        lr=cfg.m1.learning_rate,
    )
    dataset = StageBSyntheticDataset(
        length=1,
        frames=cfg.data.frames,
        size=cfg.data.height,
        base_seed=cfg.seed,
        split="train",
    )
    sample = dataset.sample(0)
    metrics = diffusion_train_step(
        components=components,
        optimizer=optimizer,
        videos=sample.video.unsqueeze(0),
        captions=[sample.caption],
        cfg=cfg,
        device=torch.device("cpu"),
    )
    assert torch.isfinite(torch.tensor(metrics.total_loss))
    assert torch.isfinite(torch.tensor(metrics.prompt_contrast_loss))
    assert torch.isfinite(torch.tensor(metrics.color_prompt_gap))
    assert torch.isfinite(torch.tensor(metrics.direction_prompt_gap))
    assert torch.isfinite(torch.tensor(metrics.shape_prompt_gap))
    assert torch.isfinite(torch.tensor(metrics.semantic_direction_loss))
    assert torch.isfinite(torch.tensor(metrics.semantic_color_loss))
    assert torch.isfinite(torch.tensor(metrics.semantic_shape_loss))
    assert any(parameter.grad is not None for parameter in components.text_encoder.parameters())
    assert any(parameter.grad is not None for parameter in components.dit.parameters())


def test_full_horizon_rollout_loss_is_finite_and_reaches_trainable_path() -> None:
    cfg = _tiny_test_config()
    components = build_stage_b_components(cfg, device=torch.device("cpu"))
    for parameter in components.vae.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [*components.text_encoder.parameters(), *components.dit.parameters()],
        lr=cfg.m1.learning_rate,
    )
    sample = StageBSyntheticDataset(
        length=1,
        frames=cfg.data.frames,
        size=cfg.data.height,
        base_seed=cfg.seed,
        split="train",
    ).sample(0)
    metrics = diffusion_train_step(
        components=components,
        optimizer=optimizer,
        videos=sample.video.unsqueeze(0),
        captions=[sample.caption],
        cfg=cfg,
        device=torch.device("cpu"),
        run_full_rollout=True,
    )
    assert torch.isfinite(torch.tensor(metrics.full_rollout_direction_loss))
    assert torch.isfinite(torch.tensor(metrics.full_rollout_color_loss))
    assert torch.isfinite(torch.tensor(metrics.full_rollout_shape_loss))
    assert metrics.full_rollout_direction_loss >= 0.0
    assert metrics.full_rollout_color_loss >= 0.0
    assert metrics.full_rollout_shape_loss >= 0.0
    assert any(parameter.grad is not None for parameter in components.text_encoder.parameters())
    assert any(parameter.grad is not None for parameter in components.dit.parameters())


def test_validation_generation_metrics_use_full_horizon() -> None:
    cfg = _tiny_test_config()
    components = build_stage_b_components(cfg, device=torch.device("cpu"))
    direction, color, shape, motion, static, object_like, persistent, area_ratio = (
        _validation_generation_metrics(
            components=components,
            cfg=cfg,
            device=torch.device("cpu"),
        )
    )
    assert 0.0 <= direction <= 1.0
    assert 0.0 <= color <= 1.0
    assert 0.0 <= shape <= 1.0
    assert motion >= 0.0
    assert 0.0 <= static <= 1.0
    assert 0.0 <= object_like <= 1.0
    assert 0.0 <= persistent <= 1.0
    assert area_ratio >= 0.0


def test_sampler_shape_and_fixed_seed_determinism() -> None:
    cfg = _tiny_test_config()
    torch.manual_seed(3)
    components = build_stage_b_components(cfg, device=torch.device("cpu"))
    prompts = ["a red square moves right at medium speed"]
    device = torch.device("cpu")
    first = sample_video(
        cfg=cfg,
        tokenizer=components.tokenizer,
        text_encoder=components.text_encoder,
        vae=components.vae,
        dit=components.dit,
        schedule=components.schedule,
        prompts=prompts,
        seed=123,
        sampling_steps=2,
        device=device,
    )
    second = sample_video(
        cfg=cfg,
        tokenizer=components.tokenizer,
        text_encoder=components.text_encoder,
        vae=components.vae,
        dit=components.dit,
        schedule=components.schedule,
        prompts=prompts,
        seed=123,
        sampling_steps=2,
        device=device,
    )
    unguided = sample_video(
        cfg=cfg,
        tokenizer=components.tokenizer,
        text_encoder=components.text_encoder,
        vae=components.vae,
        dit=components.dit,
        schedule=components.schedule,
        prompts=prompts,
        seed=123,
        sampling_steps=2,
        guidance_scale=1.0,
        device=device,
    )
    assert first.shape == (1, 3, cfg.data.frames, cfg.data.height, cfg.data.width)
    assert torch.equal(first, second)
    assert not torch.equal(first, unguided)


def test_guided_ddim_rollout_preserves_training_gradients() -> None:
    cfg = _tiny_test_config()
    components = build_stage_b_components(cfg, device=torch.device("cpu"))
    prompt = ["a red square moves right at medium speed"]
    conditional = components.tokenizer.batch(prompt, cfg.text.max_length)
    unconditional = components.tokenizer.batch([""], cfg.text.max_length)
    conditional_text = components.text_encoder(
        conditional.input_ids,
        conditional.attention_mask,
    )
    unconditional_text = components.text_encoder(
        unconditional.input_ids,
        unconditional.attention_mask,
    )
    latents = torch.randn(
        1,
        cfg.vae.latent_channels,
        cfg.data.frames // cfg.vae.temporal_downsample,
        cfg.data.height // cfg.vae.spatial_downsample,
        cfg.data.width // cfg.vae.spatial_downsample,
    )
    rolled = guided_ddim_rollout(
        latents=latents,
        dit=components.dit,
        schedule=components.schedule,
        conditional_text=conditional_text,
        conditional_mask=conditional.attention_mask,
        unconditional_text=unconditional_text,
        unconditional_mask=unconditional.attention_mask,
        sampling_timesteps=components.schedule.sampling_timesteps(
            cfg.m1.rollout_steps,
            device=torch.device("cpu"),
        ),
        guidance_scale=cfg.m1.guidance_scale,
    )
    torch.autograd.backward(rolled.square().mean())
    assert any(parameter.grad is not None for parameter in components.text_encoder.parameters())
    assert any(parameter.grad is not None for parameter in components.dit.parameters())


def test_generated_metrics_identify_cardinal_synthetic_videos() -> None:
    dataset = StageBSyntheticDataset(
        length=32,
        frames=8,
        size=32,
        base_seed=42,
        split="test",
    )
    samples = [dataset.sample(index) for index in range(32)]
    videos = torch.stack([sample.video for sample in samples])
    metrics = evaluate_generated_videos(
        videos,
        [sample.control for sample in samples],
        static_motion_threshold=0.02,
        sampling_steps=2,
    )
    assert metrics.direction_accuracy == 1.0
    assert metrics.color_accuracy == 1.0
    assert metrics.shape_accuracy == 1.0
    assert metrics.static_rate == 0.0
    assert metrics.mean_motion > 0.02
    assert metrics.object_like_frame_rate == 1.0
    assert metrics.persistent_video_rate == 1.0
    assert metrics.mean_foreground_area_ratio == 1.0
    assert metrics.shape_confusion == [[16, 0], [0, 16]]
    cfg = _tiny_test_config()
    assert passes_m1_generation_gate(metrics, cfg)
    assert not passes_m1_generation_gate(replace(metrics, shape_accuracy=0.0), cfg)
    assert not passes_m1_generation_gate(replace(metrics, persistent_video_rate=0.0), cfg)


def test_visual_fidelity_diagnostics_reject_diffuse_foreground() -> None:
    dataset = StageBSyntheticDataset(
        length=16,
        frames=8,
        size=32,
        base_seed=42,
        split="test",
    )
    samples = [dataset.sample(index) for index in range(16)]
    videos = torch.full((16, 3, 8, 32, 32), -1.0)
    videos[:, :, :, 4:28, 4:28] = 0.5
    metrics = evaluate_generated_videos(
        videos,
        [sample.control for sample in samples],
        static_motion_threshold=0.02,
        sampling_steps=2,
    )
    assert metrics.object_like_frame_rate == 0.0
    assert metrics.persistent_video_rate == 0.0
    assert metrics.mean_foreground_area_ratio > 4.0


def test_training_checkpoint_round_trip(tmp_path: Path) -> None:
    cfg = _tiny_test_config()
    device = torch.device("cpu")
    components = build_stage_b_components(cfg, device=device)
    vae_optimizer = torch.optim.AdamW(components.vae.parameters(), lr=cfg.m1.vae_learning_rate)
    diffusion_optimizer = torch.optim.AdamW(
        [*components.text_encoder.parameters(), *components.dit.parameters()],
        lr=cfg.m1.learning_rate,
    )
    generator = torch.Generator().manual_seed(9)
    checkpoint = build_training_checkpoint(
        vae=components.vae,
        text_encoder=components.text_encoder,
        dit=components.dit,
        vae_optimizer=vae_optimizer,
        diffusion_optimizer=diffusion_optimizer,
        global_step=7,
        vae_step=5,
        diffusion_step=2,
        config={"seed": cfg.seed},
        metrics={"validation_score": 1.0},
        best_validation_score=1.0,
        phase="diffusion",
        data_generator_state=generator.get_state(),
    )
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)
    restored = build_stage_b_components(cfg, device=device)
    payload = load_stage_b_weights(path, components=restored, device=device)
    assert payload["diffusion_step"] == 2
    for left, right in zip(components.dit.parameters(), restored.dit.parameters(), strict=True):
        assert torch.equal(left, right)


def test_train_resume_restores_progress(tmp_path: Path) -> None:
    from vexa_video.training.trainer import train_stage_b

    cfg = _tiny_test_config()
    cfg = replace(
        cfg,
        m1=replace(
            cfg.m1,
            vae_warmup_steps=1,
            vae_reconstruction_gate=10.0,
            max_steps=2,
            checkpoint_every=1,
            log_every=1,
            validation_samples=4,
        ),
    )
    first = train_stage_b(cfg, device=torch.device("cpu"), run_dir=tmp_path, steps=1)
    resumed = train_stage_b(
        cfg,
        device=torch.device("cpu"),
        run_dir=tmp_path,
        steps=2,
        resume=first.latest_checkpoint,
    )
    assert first.diffusion_step == 1
    assert resumed.diffusion_step == 2


def test_cuda_rng_states_are_normalized_to_cpu_byte_tensors() -> None:
    state = torch.arange(32, dtype=torch.uint8)
    normalized = _normalize_cuda_rng_states((state.clone(),))
    assert len(normalized) == 1
    assert normalized[0].device.type == "cpu"
    assert normalized[0].dtype == torch.uint8
    assert torch.equal(normalized[0], state)


def test_cuda_rng_state_normalization_rejects_invalid_dtype() -> None:
    with pytest.raises(ValueError, match=r"torch\.uint8"):
        _normalize_cuda_rng_states([torch.zeros(8, dtype=torch.float32)])
