from __future__ import annotations

import torch
from torch import Tensor

from vexa_video.config import ProjectConfig
from vexa_video.diffusion import LinearNoiseSchedule
from vexa_video.models import ByteTokenizer, TinyVideoVAE, TransformerTextEncoder, VideoDiT


def guided_ddim_rollout(
    *,
    latents: Tensor,
    dit: VideoDiT,
    schedule: LinearNoiseSchedule,
    conditional_text: Tensor,
    conditional_mask: Tensor,
    unconditional_text: Tensor,
    unconditional_mask: Tensor,
    sampling_timesteps: Tensor,
    guidance_scale: float,
) -> Tensor:
    """Run the same classifier-free-guided DDIM path used by inference."""
    if guidance_scale < 1.0:
        raise ValueError("guidance_scale must be >= 1")
    batch = latents.shape[0]
    if conditional_text.shape[0] != batch or unconditional_text.shape[0] != batch:
        raise ValueError("text batch size must match latent batch size")
    model_text = torch.cat((unconditional_text, conditional_text), dim=0)
    model_mask = torch.cat((unconditional_mask, conditional_mask), dim=0)

    for index, timestep_tensor in enumerate(sampling_timesteps):
        timestep = int(timestep_tensor.item())
        previous_timestep = (
            int(sampling_timesteps[index + 1].item()) if index + 1 < len(sampling_timesteps) else -1
        )
        timestep_batch = torch.full(
            (batch,),
            timestep,
            dtype=torch.long,
            device=latents.device,
        )
        predicted_noise = dit(
            torch.cat((latents, latents), dim=0),
            torch.cat((timestep_batch, timestep_batch), dim=0),
            model_text,
            model_mask,
        )
        unconditional_noise, conditional_noise = predicted_noise.chunk(2, dim=0)
        guided_noise = unconditional_noise + guidance_scale * (
            conditional_noise - unconditional_noise
        )
        latents = schedule.ddim_step(
            latents,
            guided_noise,
            timestep=timestep,
            previous_timestep=previous_timestep,
        )
    return latents


def sample_video(
    *,
    cfg: ProjectConfig,
    tokenizer: ByteTokenizer,
    text_encoder: TransformerTextEncoder,
    vae: TinyVideoVAE,
    dit: VideoDiT,
    schedule: LinearNoiseSchedule,
    prompts: list[str],
    seed: int,
    sampling_steps: int | None = None,
    guidance_scale: float | None = None,
    device: torch.device,
) -> Tensor:
    """Generate RGB videos from Gaussian latent noise with deterministic DDIM sampling."""
    if not prompts:
        raise ValueError("at least one prompt is required")
    steps = sampling_steps if sampling_steps is not None else cfg.m1.sampling_steps
    guidance = guidance_scale if guidance_scale is not None else cfg.m1.guidance_scale
    if guidance < 1.0:
        raise ValueError("guidance_scale must be >= 1")
    latent_shape = (
        len(prompts),
        cfg.vae.latent_channels,
        cfg.data.frames // cfg.vae.temporal_downsample,
        cfg.data.height // cfg.vae.spatial_downsample,
        cfg.data.width // cfg.vae.spatial_downsample,
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    dtype = next(dit.parameters()).dtype
    latents = torch.randn(latent_shape, generator=generator, device=device, dtype=dtype)
    conditional_tokens = tokenizer.batch(prompts, cfg.text.max_length, device=device)
    unconditional_tokens = tokenizer.batch([""] * len(prompts), cfg.text.max_length, device=device)
    sampling_timesteps = schedule.sampling_timesteps(steps, device=device)

    text_encoder.eval()
    vae.eval()
    dit.eval()
    with torch.no_grad():
        conditional_text = text_encoder(
            conditional_tokens.input_ids,
            conditional_tokens.attention_mask,
        )
        unconditional_text = text_encoder(
            unconditional_tokens.input_ids,
            unconditional_tokens.attention_mask,
        )
        latents = guided_ddim_rollout(
            latents=latents,
            dit=dit,
            schedule=schedule,
            conditional_text=conditional_text,
            conditional_mask=conditional_tokens.attention_mask,
            unconditional_text=unconditional_text,
            unconditional_mask=unconditional_tokens.attention_mask,
            sampling_timesteps=sampling_timesteps,
            guidance_scale=guidance,
        )
        video = vae.decode(latents)
    return video.clamp(-1.0, 1.0)
