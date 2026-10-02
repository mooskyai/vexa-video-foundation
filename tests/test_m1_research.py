from __future__ import annotations

import math

import pytest
import torch

from vexa_video.diffusion import LinearNoiseSchedule
from vexa_video.training.m1_research import (
    causal_latent_delta_loss,
    constraint_rank_score,
    gradient_cosine_matrix,
    latent_svd_diagnostics,
    latent_target_ranking_loss,
    merge_primary_and_protected_gradients,
    min_snr_epsilon_weights,
    sinkhorn_causal_gate,
    sinkhorn_shape_geometry_loss,
)


def test_min_snr_epsilon_weights_downweight_high_snr_steps() -> None:
    schedule = LinearNoiseSchedule(1000, 0.0001, 0.02)
    timesteps = torch.tensor([0, 999], dtype=torch.long)
    weights = min_snr_epsilon_weights(
        schedule,
        timesteps,
        gamma=5.0,
        dtype=torch.float32,
    )

    assert torch.isfinite(weights).all()
    assert bool((weights > 0.0).all().item())
    assert float(weights[0].item()) < float(weights[1].item())
    assert float(weights[1].item()) <= 1.0


def test_causal_latent_delta_loss_rewards_correct_shape_change() -> None:
    square = torch.tensor([[[[[2.0, 0.0]]]]])
    circle = torch.tensor([[[[[0.0, 1.0]]]]])

    exact_loss, exact_cosine, exact_ratio = causal_latent_delta_loss(
        square,
        circle,
        square,
        circle,
    )
    collapsed = torch.zeros_like(square)
    collapsed_loss, collapsed_cosine, collapsed_ratio = causal_latent_delta_loss(
        collapsed,
        collapsed,
        square,
        circle,
    )

    assert float(exact_loss.item()) == pytest.approx(0.0, abs=1e-6)
    assert float(exact_cosine.item()) == pytest.approx(1.0, abs=1e-6)
    assert float(exact_ratio.item()) == pytest.approx(1.0, abs=1e-6)
    assert float(collapsed_loss.item()) > float(exact_loss.item())
    assert float(collapsed_cosine.item()) == pytest.approx(0.0, abs=1e-6)
    assert float(collapsed_ratio.item()) == pytest.approx(0.0, abs=1e-6)


def test_latent_target_ranking_rejects_collapsed_shape_pair() -> None:
    square = torch.tensor([[[[[2.0, 0.0]]]]])
    circle = torch.tensor([[[[[0.0, 1.0]]]]])

    exact_loss, exact_distance, exact_margin = latent_target_ranking_loss(
        square,
        circle,
        square,
        circle,
        margin=0.25,
    )
    midpoint = 0.5 * (square + circle)
    collapsed_loss, collapsed_distance, collapsed_margin = latent_target_ranking_loss(
        midpoint,
        midpoint,
        square,
        circle,
        margin=0.25,
    )

    assert float(exact_loss.item()) == pytest.approx(0.0, abs=1e-6)
    assert float(exact_distance.item()) == pytest.approx(0.0, abs=1e-6)
    assert float(exact_margin.item()) > 0.25
    assert float(collapsed_loss.item()) > float(exact_loss.item())
    assert float(collapsed_distance.item()) > 0.0
    assert float(collapsed_margin.item()) == pytest.approx(0.0, abs=1e-6)


def test_sinkhorn_causal_gate_delays_geometry_until_latent_separates() -> None:
    low = sinkhorn_causal_gate(torch.tensor(0.05), start=0.10, full=0.30)
    middle = sinkhorn_causal_gate(torch.tensor(0.20), start=0.10, full=0.30)
    high = sinkhorn_causal_gate(torch.tensor(0.40), start=0.10, full=0.30)

    assert float(low.item()) == pytest.approx(0.0, abs=1e-6)
    assert float(middle.item()) == pytest.approx(0.5, abs=1e-6)
    assert float(high.item()) == pytest.approx(1.0, abs=1e-6)


def test_shape_protected_gradient_projection_removes_only_conflicting_component() -> None:
    primary = (torch.tensor([1.0, -1.0]),)
    protected = (torch.tensor([-1.0, 1.0]),)

    merged, cosine, active = merge_primary_and_protected_gradients(primary, protected)

    assert active is True
    assert cosine == pytest.approx(-1.0, abs=1e-6)
    assert merged[0] is not None
    assert torch.allclose(merged[0], protected[0], atol=1e-6)


def test_shape_protected_gradient_projection_preserves_aligned_primary_update() -> None:
    primary = (torch.tensor([2.0, 0.0]),)
    protected = (torch.tensor([1.0, 0.0]),)

    merged, cosine, active = merge_primary_and_protected_gradients(primary, protected)

    assert active is False
    assert cosine == pytest.approx(1.0, abs=1e-6)
    assert merged[0] is not None
    assert torch.allclose(merged[0], torch.tensor([3.0, 0.0]), atol=1e-6)


def test_latent_svd_diagnostics_identify_low_rank_shape_subspace() -> None:
    coefficients = torch.arange(1.0, 9.0).reshape(8, 1)
    direction = torch.tensor([[1.0, -2.0, 0.5, 3.0]])
    deltas = coefficients * direction
    metrics = latent_svd_diagnostics(deltas)

    assert metrics["shape_svd_top1_energy"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["shape_svd_top4_energy"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["shape_svd_effective_rank"] == pytest.approx(1.0, abs=1e-5)


def test_gradient_cosine_matrix_detects_opposed_tasks() -> None:
    parameter = torch.tensor([1.0, -2.0], requires_grad=True)
    positive = parameter.square().sum()
    negative = -parameter.square().sum()
    aligned = 3.0 * parameter.square().sum()

    metrics = gradient_cosine_matrix(
        {"positive": positive, "negative": negative, "aligned": aligned},
        [parameter],
    )

    assert metrics["grad_cos_positive_negative"] == pytest.approx(-1.0, abs=1e-6)
    assert metrics["grad_cos_positive_aligned"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["grad_cos_negative_aligned"] == pytest.approx(-1.0, abs=1e-6)


def test_constraint_rank_score_prioritizes_gate_violation() -> None:
    feasible = constraint_rank_score(0.0, 1_000_000.0)
    violating = constraint_rank_score(0.001, 0.0)
    same_violation_better_tie = constraint_rank_score(0.25, 1.0)
    same_violation_worse_tie = constraint_rank_score(0.25, 10.0)

    assert feasible < violating
    assert same_violation_better_tie < same_violation_worse_tie


def test_sinkhorn_shape_geometry_prefers_matching_square() -> None:
    pytest.importorskip("geomloss")
    video = torch.full((1, 3, 2, 32, 32), -1.0)
    video[:, :, :, 12:20, 12:20] = 1.0

    loss, correct, margin = sinkhorn_shape_geometry_loss(
        video,
        ["a red square moves right at medium speed"],
        size=32,
        blur=0.12,
        margin=0.02,
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(correct)
    assert torch.isfinite(margin)
    assert float(margin.item()) > 0.0
    assert math.isfinite(float(loss.item()))
