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
