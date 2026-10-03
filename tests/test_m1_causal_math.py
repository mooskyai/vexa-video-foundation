from __future__ import annotations

import pytest
import torch

from vexa_video.training.m1_causal_math import (
    ddim_linear_coefficients,
    parallel_response_decomposition,
    permutation_linear_cka,
    symmetric_two_factor_decomposition,
)


def test_parallel_response_decomposition_identifies_exact_target() -> None:
    target = torch.tensor([[1.0, -2.0], [2.0, 1.0]])
    metrics = parallel_response_decomposition(target, target)

    assert metrics.signed_gain == pytest.approx(1.0, abs=1e-6)
    assert metrics.parallel_ratio == pytest.approx(1.0, abs=1e-6)
    assert metrics.orthogonal_ratio == pytest.approx(0.0, abs=1e-6)
    assert metrics.aligned_energy_fraction == pytest.approx(1.0, abs=1e-6)


def test_parallel_response_decomposition_identifies_orthogonal_response() -> None:
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    response = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]])
    metrics = parallel_response_decomposition(response, target)

    assert metrics.signed_gain == pytest.approx(0.0, abs=1e-6)
    assert metrics.parallel_ratio == pytest.approx(0.0, abs=1e-6)
    assert metrics.orthogonal_ratio == pytest.approx(1.0, abs=1e-6)
    assert metrics.aligned_energy_fraction == pytest.approx(0.0, abs=1e-6)


def test_symmetric_two_factor_decomposition_is_exact() -> None:
    ff = torch.tensor([[7.0, 5.0]])
    fs = torch.tensor([[4.0, 4.0]])
    sf = torch.tensor([[3.0, 2.0]])
    ss = torch.tensor([[1.0, 1.0]])

    condition, state, total = symmetric_two_factor_decomposition(ff, fs, sf, ss)

    assert torch.allclose(condition + state, total)
    assert torch.allclose(total, ff - ss)


def test_permutation_linear_cka_calibrates_observed_alignment() -> None:
    first = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 2.0],
            [-1.0, 0.0, 3.0],
            [0.0, -1.0, 4.0],
            [1.0, 1.0, 5.0],
            [-1.0, -1.0, 6.0],
        ]
    )
    result = permutation_linear_cka(first, first, permutations=64, seed=7)

    assert result.observed == pytest.approx(1.0, abs=1e-6)
    assert result.observed > result.null_mean
    assert result.z_score > 0.0
    assert 0.0 < result.p_value <= 1.0


def test_ddim_linear_coefficients_reconstruct_transition() -> None:
    alpha_t = torch.tensor(0.25)
    alpha_previous = torch.tensor(0.49)
    sample = torch.tensor([[[1.25, -0.5]]])
    predicted_noise = torch.tensor([[[0.2, -0.3]]])

    state_coefficient, noise_coefficient = ddim_linear_coefficients(alpha_t, alpha_previous)
    reconstructed = state_coefficient * sample + noise_coefficient * predicted_noise
    clean = (sample - (1.0 - alpha_t).sqrt() * predicted_noise) / alpha_t.sqrt()
    expected = alpha_previous.sqrt() * clean + (1.0 - alpha_previous).sqrt() * predicted_noise

    assert torch.allclose(reconstructed, expected, atol=1e-6, rtol=1e-6)


def test_ddim_linear_coefficients_reconstruct_final_clean_prediction() -> None:
    alpha_t = torch.tensor(0.36)
    sample = torch.tensor([[[0.8, -1.2]]])
    predicted_noise = torch.tensor([[[0.1, 0.4]]])

    state_coefficient, noise_coefficient = ddim_linear_coefficients(alpha_t, None)
    reconstructed = state_coefficient * sample + noise_coefficient * predicted_noise
    expected = (sample - (1.0 - alpha_t).sqrt() * predicted_noise) / alpha_t.sqrt()

    assert torch.allclose(reconstructed, expected, atol=1e-6, rtol=1e-6)
