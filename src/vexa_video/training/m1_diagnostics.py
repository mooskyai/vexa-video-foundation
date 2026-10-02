from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True, slots=True)
class CounterfactualResponseMetrics:
    cosine: float
    magnitude_ratio: float
    linear_cka: float
    top1_energy: float
    top4_energy: float
    effective_rank: float


def _flatten_centered(samples: Tensor) -> Tensor:
    if samples.ndim < 2 or samples.shape[0] < 2:
        raise ValueError("samples must contain at least two observations")
    matrix = samples.flatten(start_dim=1)
    return matrix - matrix.mean(dim=0, keepdim=True)


def linear_cka(first: Tensor, second: Tensor, *, eps: float = 1e-12) -> Tensor:
    """Linear centered-kernel alignment between two batched representations."""
    if first.shape[0] != second.shape[0]:
        raise ValueError("representations must contain the same number of samples")
    x = _flatten_centered(first)
    y = _flatten_centered(second)
    first_gram = x @ x.transpose(0, 1)
    second_gram = y @ y.transpose(0, 1)
    numerator = (first_gram * second_gram).sum()
    denominator = (first_gram.square().sum().sqrt() * second_gram.square().sum().sqrt()).clamp_min(
        eps
    )
    return numerator / denominator


def response_svd_metrics(responses: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return top-1/top-4 energy and effective rank of a counterfactual response matrix."""
    matrix = _flatten_centered(responses)
    singular_values = torch.linalg.svdvals(matrix)
    energy = singular_values.square()
    fractions = energy / energy.sum().clamp_min(1e-12)
    top1 = fractions[:1].sum()
    top4 = fractions[: min(4, fractions.numel())].sum()
    entropy = -(fractions * fractions.clamp_min(1e-12).log()).sum()
    return top1, top4, torch.exp(entropy)


def counterfactual_response_metrics(
    predicted_delta: Tensor,
    target_delta: Tensor,
) -> CounterfactualResponseMetrics:
    """Measure causal response alignment, gain, and subspace structure."""
    if predicted_delta.shape != target_delta.shape:
        raise ValueError("predicted and target deltas must have identical shapes")
    predicted_flat = predicted_delta.flatten(start_dim=1)
    target_flat = target_delta.flatten(start_dim=1)
    cosine = F.cosine_similarity(predicted_flat, target_flat, dim=1, eps=1e-8).mean()
    target_norm = target_flat.norm(dim=1).clamp_min(1e-8)
    magnitude_ratio = (predicted_flat.norm(dim=1) / target_norm).mean()
    top1, top4, effective_rank = response_svd_metrics(predicted_delta)
    cka = linear_cka(predicted_delta, target_delta)
    return CounterfactualResponseMetrics(
        cosine=float(cosine.item()),
        magnitude_ratio=float(magnitude_ratio.item()),
        linear_cka=float(cka.item()),
        top1_energy=float(top1.item()),
        top4_energy=float(top4.item()),
        effective_rank=float(effective_rank.item()),
    )
