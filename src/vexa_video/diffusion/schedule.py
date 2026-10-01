from __future__ import annotations

import torch
from torch import Tensor


class LinearNoiseSchedule:
    def __init__(self, timesteps: int, beta_start: float, beta_end: float) -> None:
        if timesteps <= 1:
            raise ValueError("timesteps must be > 1")
        if not 0 < beta_start < beta_end < 1:
            raise ValueError("require 0 < beta_start < beta_end < 1")
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(alphas, dim=0)

    def add_noise(self, clean: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        if clean.shape != noise.shape:
            raise ValueError("clean and noise tensors must have identical shapes")
        if timesteps.ndim != 1 or timesteps.shape[0] != clean.shape[0]:
            raise ValueError("timesteps must have shape [B]")
        alpha = self.alpha_cumprod.to(device=clean.device, dtype=clean.dtype)[timesteps]
        while alpha.ndim < clean.ndim:
            alpha = alpha.unsqueeze(-1)
        return alpha.sqrt() * clean + (1.0 - alpha).sqrt() * noise

    def sampling_timesteps(self, steps: int, *, device: torch.device) -> Tensor:
        if not 1 <= steps <= self.timesteps:
            raise ValueError("sampling steps must be in [1, timesteps]")
        return (
            torch.linspace(
                self.timesteps - 1,
                0,
                steps=steps,
                device=device,
                dtype=torch.float32,
            )
            .round()
            .to(dtype=torch.long)
        )

    def predict_clean(self, sample: Tensor, predicted_noise: Tensor, timestep: int) -> Tensor:
        if sample.shape != predicted_noise.shape:
            raise ValueError("sample and predicted_noise must have identical shapes")
        alpha = self.alpha_cumprod.to(device=sample.device, dtype=sample.dtype)[timestep]
        return (sample - (1.0 - alpha).sqrt() * predicted_noise) / alpha.sqrt().clamp_min(1e-8)

    def ddim_step(
        self,
        sample: Tensor,
        predicted_noise: Tensor,
        *,
        timestep: int,
        previous_timestep: int,
    ) -> Tensor:
        """Deterministic DDIM-style epsilon step (eta=0)."""
        clean = self.predict_clean(sample, predicted_noise, timestep)
        if previous_timestep < 0:
            return clean
        alpha_previous = self.alpha_cumprod.to(device=sample.device, dtype=sample.dtype)[
            previous_timestep
        ]
        return alpha_previous.sqrt() * clean + (1.0 - alpha_previous).sqrt() * predicted_noise
