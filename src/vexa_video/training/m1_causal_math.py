from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class PermutationCKAResult:
    observed: float
    null_mean: float
    null_std: float
    z_score: float
    p_value: float


@dataclass(frozen=True, slots=True)
class ParallelResponseResult:
    signed_gain: float
    parallel_ratio: float
    orthogonal_ratio: float
    aligned_energy_fraction: float


def _centered_matrix(samples: Tensor) -> Tensor:
    if samples.ndim < 2 or samples.shape[0] < 4:
        raise ValueError("samples must contain at least four observations")
    matrix = samples.flatten(start_dim=1)
    return matrix - matrix.mean(dim=0, keepdim=True)


def permutation_linear_cka(
    first: Tensor,
    second: Tensor,
    *,
    permutations: int = 256,
    seed: int = 0,
    eps: float = 1e-12,
) -> PermutationCKAResult:
    """Calibrate biased linear CKA against row permutations of the actual representations."""
    if first.shape[0] != second.shape[0]:
        raise ValueError("representations must contain the same number of samples")
    if permutations <= 0:
        raise ValueError("permutations must be positive")
    x = _centered_matrix(first)
    y = _centered_matrix(second)
    first_gram = x @ x.transpose(0, 1)
    second_gram = y @ y.transpose(0, 1)
    denominator = (first_gram.square().sum().sqrt() * second_gram.square().sum().sqrt()).clamp_min(
        eps
    )
    observed_tensor = (first_gram * second_gram).sum() / denominator

    generator = torch.Generator(device="cpu").manual_seed(seed)
    null_values: list[Tensor] = []
    size = first.shape[0]
    for _ in range(permutations):
        permutation = torch.randperm(size, generator=generator).to(device=second_gram.device)
        permuted = second_gram.index_select(0, permutation).index_select(1, permutation)
        null_values.append((first_gram * permuted).sum() / denominator)
    null = torch.stack(null_values)
    null_mean = null.mean()
    null_std = null.std(unbiased=False)
    z_score = (observed_tensor - null_mean) / null_std.clamp_min(eps)
    exceedances = (null >= observed_tensor).sum().to(dtype=torch.float32)
    p_value = (exceedances + 1.0) / float(permutations + 1)
    return PermutationCKAResult(
        observed=float(observed_tensor.item()),
        null_mean=float(null_mean.item()),
        null_std=float(null_std.item()),
        z_score=float(z_score.item()),
        p_value=float(p_value.item()),
    )


def parallel_response_decomposition(
    response: Tensor,
    target: Tensor,
    *,
    eps: float = 1e-12,
) -> ParallelResponseResult:
    """Split a response into target-parallel and target-orthogonal components per sample."""
    if response.shape != target.shape:
        raise ValueError("response and target must have identical shapes")
    response_flat = response.flatten(start_dim=1)
    target_flat = target.flatten(start_dim=1)
    target_sq = target_flat.square().sum(dim=1).clamp_min(eps)
    target_norm = target_sq.sqrt()
    gain = (response_flat * target_flat).sum(dim=1) / target_sq
    parallel = gain[:, None] * target_flat
    orthogonal = response_flat - parallel
    parallel_ratio = parallel.norm(dim=1) / target_norm
    orthogonal_ratio = orthogonal.norm(dim=1) / target_norm
    response_sq = response_flat.square().sum(dim=1).clamp_min(eps)
    aligned_energy = parallel.square().sum(dim=1) / response_sq
    return ParallelResponseResult(
        signed_gain=float(gain.mean().item()),
        parallel_ratio=float(parallel_ratio.mean().item()),
        orthogonal_ratio=float(orthogonal_ratio.mean().item()),
        aligned_energy_fraction=float(aligned_energy.mean().item()),
    )


def symmetric_two_factor_decomposition(
    first_state_first_condition: Tensor,
    first_state_second_condition: Tensor,
    second_state_first_condition: Tensor,
    second_state_second_condition: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Exact Shapley-style split of a two-factor state/condition response difference."""
    shapes = {
        first_state_first_condition.shape,
        first_state_second_condition.shape,
        second_state_first_condition.shape,
        second_state_second_condition.shape,
    }
    if len(shapes) != 1:
        raise ValueError("all two-factor response tensors must have identical shapes")
    condition = 0.5 * (
        (first_state_first_condition - first_state_second_condition)
        + (second_state_first_condition - second_state_second_condition)
    )
    state = 0.5 * (
        (first_state_first_condition - second_state_first_condition)
        + (first_state_second_condition - second_state_second_condition)
    )
    total = first_state_first_condition - second_state_second_condition
    return condition, state, total
